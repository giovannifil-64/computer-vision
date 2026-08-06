"""
evaluate_detail
===============
The part of the evaluation that goes beyond the three headline metrics:
the paper's collision measure, a significance test of CLIK against the "no
movement" reference, a breakdown by tooth type, and an account of the teeth that
are excluded from the metrics.

Functions
---------
- `collision(meshes, samples)`: Mean penetration depth between neighbouring teeth.
- `dentition(sid, stage, converted_root, output_root)`: The teeth of one dentition state.
- `analyse_collision(sids, ...)`: Collision for initial / prediction / ground truth.
- `analyse_significance(csv_path)`: Paired tests of CLIK vs the baseline.
- `analyse_by_tooth_type(...)`: Errors split by incisor / canine / premolar / molar.
- `analyse_excluded(...)`: What happens to extracted teeth and third molars.
- `main()`: Run the requested analyses and write a JSON summary.

Example
-------
```bash
python evaluate_detail.py --converted ../data/Data_prepost --output ../output/Output_prepost \\
    --report ../report --collision-subjects 40
```

Notes
-----
- The crowns of this dataset are watertight, so signed distances (and therefore
  penetration depth) are well defined.
- Collision is reported for the ground truth as well, which sets the level a
  clinically designed setup actually achieves.
"""
import os
import csv
import json
import glob
import argparse
import numpy as np
import trimesh

from meshes import load_mesh
from scipy import stats

TOOTH_TYPE = {
    'incisivo': {7, 8, 9, 10, 23, 24, 25, 26},
    'canino': {6, 11, 22, 27},
    'premolare': {4, 5, 12, 13, 20, 21, 28, 29},
    'molare': {2, 3, 14, 15, 18, 19, 30, 31},
}
THIRD_MOLARS = {1, 16, 17, 32}


def dentition(sid, stage, converted_root, output_root):
    """
    Load one dentition state as a dict of meshes.

    Parameters
    ----------
    - `sid (str)`: Subject id.
    - `stage (str)`: `"initial"`, `"final"` or `"pred"`.
    - `converted_root (str)`: Converter output root.
    - `output_root (str)`: CLIK output root.

    Returns
    -------
    - `dict`: `{tooth_id: trimesh.Trimesh}`.
    """
    pat = (os.path.join(output_root, sid, 'results', '*.ply') if stage == 'pred'
           else os.path.join(converted_root, sid, stage, '*.stl'))
    out = {}
    for f in glob.glob(pat):
        name = os.path.basename(f).rsplit('.', 1)[0]
        if name.isdigit():
            out[int(name)] = load_mesh(f)
    return out


def collision(meshes, samples=600, neighbour_gap=1):
    """
    Mean penetration depth between neighbouring teeth, in millimetres.

    Parameters
    ----------
    - `meshes (dict)`: `{tooth_id: trimesh.Trimesh}` of one dentition state.
    - `samples (int, optional)`: Surface points sampled per tooth. Default `600`.
    - `neighbour_gap (int, optional)`: How many ids away still counts as adjacent. Default `1`.

    Returns
    -------
    - `tuple`: `(mean_depth_mm, penetrating_fraction)`; `(0.0, 0.0)` when nothing overlaps.

    Notes
    -----
    - Follows the paper's definition: points falling inside a neighbouring tooth
      have a negative signed distance, and the metric averages the absolute value
      over those points only.
    """
    depths, total = [], 0
    ids = sorted(meshes)
    for t in ids:
        pts = meshes[t].sample(samples)
        total += len(pts)
        for n in (t - neighbour_gap, t + neighbour_gap):
            if n not in meshes:
                continue
            sd = trimesh.proximity.signed_distance(meshes[n], pts)  # >0 inside neighbour
            inside = sd[sd > 0]
            if len(inside):
                depths.append(inside)
    if not depths:
        return 0.0, 0.0
    d = np.concatenate(depths)
    return float(d.mean()), float(len(d) / max(total, 1))


def analyse_collision(sids, converted_root, output_root, samples=600):
    """
    Collision for the initial, predicted and ground-truth dentitions.

    Parameters
    ----------
    - `sids (list)`: Subject ids to process.
    - `converted_root (str)`: Converter output root.
    - `output_root (str)`: CLIK output root.
    - `samples (int, optional)`: Surface points per tooth. Default `600`.

    Returns
    -------
    - `dict`: Median depth and penetrating fraction per state.
    """
    res = {k: {'depth': [], 'frac': []} for k in ('initial', 'pred', 'final')}
    for i, sid in enumerate(sids, 1):
        for stage in res:
            m = dentition(sid, stage, converted_root, output_root)
            if len(m) < 2:
                continue
            d, f = collision(m, samples)
            res[stage]['depth'].append(d); res[stage]['frac'].append(f)
        print(f'  [{i}/{len(sids)}] {sid}: '
              + '  '.join(f"{k} {res[k]['depth'][-1]:.3f} mm" for k in res if res[k]['depth']))
    return {k: {'median_depth_mm': float(np.median(v['depth'])),
                'mean_depth_mm': float(np.mean(v['depth'])),
                'penetrating_fraction': float(np.mean(v['frac'])),
                'n': len(v['depth'])} for k, v in res.items() if v['depth']}


def analyse_significance(csv_path):
    """
    Paired tests of CLIK against the "no movement" reference, per subject.

    Parameters
    ----------
    - `csv_path (str)`: `alignment_metrics.csv`.

    Returns
    -------
    - `dict`: For each metric, the two means, the t-test and the Wilcoxon p-values.
    """
    rows = list(csv.DictReader(open(csv_path)))
    out = {}
    for k, kb, lab in (('rot_err', 'rot_baseline', 'rotazione'),
                       ('trans_err', 'trans_baseline', 'traslazione'),
                       ('pcd_err', 'pcd_baseline', 'point_cloud')):
        a = np.array([float(r[k]) for r in rows])
        b = np.array([float(r[kb]) for r in rows])
        t = stats.ttest_rel(a, b)
        w = stats.wilcoxon(a, b)
        out[lab] = {'clik_mean': float(a.mean()), 'baseline_mean': float(b.mean()),
                    'p_ttest': float(t.pvalue), 'p_wilcoxon': float(w.pvalue),
                    'clik_better': bool(a.mean() < b.mean()), 'n_subjects': len(a)}
    return out


def analyse_by_tooth_type(sids, converted_root, output_root):
    """
    Alignment error split by tooth type.

    Parameters
    ----------
    - `sids (list)`: Subject ids.
    - `converted_root (str)`: Converter output root.
    - `output_root (str)`: CLIK output root.

    Returns
    -------
    - `dict`: Median rotation / point-cloud error and baseline, per tooth type.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluate_alignment import evaluate_subject

    acc = {k: {'rot': [], 'pcd': [], 'rot_b': [], 'pcd_b': []} for k in TOOTH_TYPE}
    for sid in sids:
        rep = evaluate_subject(sid, converted_root, output_root)
        if not rep:
            continue
        for r in rep['teeth']:
            for name, ids in TOOTH_TYPE.items():
                if r['tooth'] in ids:
                    acc[name]['rot'].append(r['rot_err']); acc[name]['pcd'].append(r['pcd_err'])
                    acc[name]['rot_b'].append(r['rot_moved']); acc[name]['pcd_b'].append(r['pcd_moved'])
    return {k: {'n': len(v['rot']),
                'rot_median': float(np.median(v['rot'])), 'rot_baseline': float(np.median(v['rot_b'])),
                'pcd_median': float(np.median(v['pcd'])), 'pcd_baseline': float(np.median(v['pcd_b']))}
            for k, v in acc.items() if v['rot']}


def analyse_excluded(converted_root, output_root):
    """
    Account for the teeth left out of the metrics.

    Parameters
    ----------
    - `converted_root (str)`: Converter output root.
    - `output_root (str)`: CLIK output root.

    Returns
    -------
    - `dict`: Counts and, for teeth extracted during treatment, how far CLIK moved them.

    Notes
    -----
    - CLIK's scheme covers 28 teeth, so third molars are never predicted; teeth
      extracted during treatment are absent from the ground truth but CLIK, not
      knowing about the extraction, still proposes a position for them.
    """
    n_third = n_extracted = 0
    moved = []
    for meta_f in glob.glob(os.path.join(converted_root, '*', 'center.json')):
        sid = os.path.basename(os.path.dirname(meta_f))
        meta = json.load(open(meta_f))
        extracted = set(meta.get('extracted', []))
        n_third += len([t for t in meta.get('teeth_ori', []) if t in THIRD_MOLARS])
        n_extracted += len(extracted)
        tf = os.path.join(output_root, sid, 'results', 'transformation.json')
        if not (extracted and os.path.exists(tf)):
            continue
        tr = {int(k.split('-')[1]): np.asarray(v, float) for k, v in json.load(open(tf)).items()}
        for t in extracted:
            if t in tr:
                moved.append(float(np.linalg.norm(tr[t][:3, 3])))
    return {'third_molars_never_predicted': n_third,
            'teeth_extracted_during_treatment': n_extracted,
            'displacement_applied_to_extracted_teeth_mm':
                {'median': float(np.median(moved)), 'n': len(moved)} if moved else None}


def main():
    """Run the analyses and write `metrics_detail.json` next to the other metrics."""
    ap = argparse.ArgumentParser(description="Collision, significance, tooth type and excluded teeth.")
    ap.add_argument('--converted', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--collision-subjects', type=int, default=40,
                    help='subjects used for the collision measure; 0 skips it (it is the slow part)')
    args = ap.parse_args()

    sids = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(args.converted, '*', 'center.json')))
    res = {}

    print('Significativita statistica...')
    res['significance'] = analyse_significance(os.path.join(args.output, 'alignment_metrics.csv'))
    for k, v in res['significance'].items():
        print(f"  {k:12s} CLIK {v['clik_mean']:6.2f} vs baseline {v['baseline_mean']:6.2f}  "
              f"p(t)={v['p_ttest']:.2e}  p(wilcoxon)={v['p_wilcoxon']:.2e}")

    print('\nDenti esclusi...')
    res['excluded'] = analyse_excluded(args.converted, args.output)
    print('  ' + json.dumps(res['excluded'], ensure_ascii=False))

    print('\nDettaglio per tipo di dente...')
    res['by_tooth_type'] = analyse_by_tooth_type(sids, args.converted, args.output)
    for k, v in res['by_tooth_type'].items():
        print(f"  {k:10s} n={v['n']:5d}  rot {v['rot_median']:5.2f} (base {v['rot_baseline']:5.2f})  "
              f"pcd {v['pcd_median']:5.2f} (base {v['pcd_baseline']:5.2f})")

    if args.collision_subjects:
        print(f'\nCollisione su {args.collision_subjects} soggetti (lento)...')
        res['collision'] = analyse_collision(sids[:args.collision_subjects], args.converted, args.output)
        for k, v in res['collision'].items():
            print(f"  {k:8s} profondita mediana {v['median_depth_mm']:.3f} mm  "
                  f"punti compenetrati {100*v['penetrating_fraction']:.2f}%")
    else:
        print('\nCollisione saltata (--collision-subjects 0); il valore gia calcolato viene conservato')
        old = os.path.join(args.report, 'metrics_detail.json')
        if os.path.exists(old):
            res['collision'] = json.load(open(old)).get('collision', {})

    out = os.path.join(args.report, 'metrics_detail.json')
    json.dump(res, open(out, 'w'), indent=2, ensure_ascii=False)
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
