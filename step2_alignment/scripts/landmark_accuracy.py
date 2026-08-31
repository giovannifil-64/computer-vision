"""
landmark_accuracy
=================
Check whether CLIK's landmarks land on anatomically meaningful points of this
dataset, by comparing them with the landmark annotations shipped with it. This
complements the repeatability test: repeatability proves the detector is
*consistent*, this asks whether it is *correct*.

For every annotated landmark the distance to the nearest CLIK landmark is
measured, together with a random baseline (the same number of points sampled on
the same crowns), which sets the chance level for the same measurement.

Functions
---------
- `load_gt(sid, dataset_root, converted_root)`: Dataset landmarks, in the converted frame.
- `evaluate_subject(models, sid, dataset_root, converted_root)`: Distances per landmark class.
- `main()`: Aggregate over the annotated subjects.

Example
-------
```bash
python landmark_accuracy.py --converted ../Data_prepost \
    --dataset ../../PrePostOrthodontic --limit 70
```

Notes
-----
- The dataset's scheme (3-4 points per tooth: Pt0, Pt2, Pt3, Pt6) differs from
  CLIK's (8-11 per tooth), so this is a coverage measure ("did CLIK put some
  landmark on this true anatomical point?") rather than a per-class identity match.
- Annotations live in the raw scan frame, so the converter's offset and rotation
  (stored in `center.json`) are applied to bring them into the same frame.
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
from meshes import load_mesh

from scipy.spatial import KDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepost_to_clik import FDI2U
from landmark_repeatability import detect_stage, load_detection, CKPTS


def load_gt(sid, dataset_root, converted_root, stage='ori'):
    """
    Load the dataset's own landmark annotations, mapped into the converted frame.

    Parameters
    ----------
    - `sid (str)`: Subject id.
    - `dataset_root (str)`: Root holding `Landmark_annotation/`.
    - `converted_root (str)`: Converter output root (for `center.json`).
    - `stage (str, optional)`: `"ori"` (pre-treatment) or `"final"` (post). Default `"ori"`.

    Returns
    -------
    - `dict`: `{universal_tooth_id: {class_name: np.ndarray(3)}}`, or `{}` if absent.
    """
    meta = json.load(open(os.path.join(converted_root, sid, 'center.json')))
    center = np.asarray(meta['center'], float)
    R = np.asarray(meta['frame_R'], float)

    out = {}
    tag = 'Ori' if stage == 'ori' else 'Final'
    for arch in ('L', 'U'):
        p = os.path.join(dataset_root, 'Landmark_annotation', sid, stage,
                         f'{arch}_{tag}_landmarks.json')
        if not os.path.exists(p):
            continue
        for entry in json.load(open(p))['landmarks'][1:]:      # [0] is jaw metadata
            for fdi_str, pts in entry.items():
                if not fdi_str.isdigit() or int(fdi_str) not in FDI2U:
                    continue
                uid = FDI2U[int(fdi_str)]
                out.setdefault(uid, {})
                for name, coord in pts.items():
                    out[uid][name] = R @ (np.asarray(coord, float) - center)
    return out


def evaluate_subject(models, sid, dataset_root, converted_root, trials=10, seed=0, stage='ori'):
    """
    Compare CLIK's landmarks with the dataset annotations for one subject.

    Parameters
    ----------
    - `models (tuple)`: The four Stage-1 detectors.
    - `sid (str)`: Subject id.
    - `dataset_root (str)`: Root holding `Landmark_annotation/`.
    - `converted_root (str)`: Converter output root.
    - `trials (int, optional)`: Random-baseline repetitions. Default `10`.
    - `seed (int, optional)`: RNG seed. Default `0`.

    Returns
    -------
    - `dict` or `None`: `{'sid', 'per_class': {name: [distances]}, 'baseline': [distances]}`.
    """
    gt = load_gt(sid, dataset_root, converted_root, stage)
    if not gt:
        return None
    folder = 'initial' if stage == 'ori' else 'final'
    det = detect_stage(models, converted_root, sid, folder, seed=1)
    if not det:
        return None

    clik_pts = np.array([v for d in det.values() for v in d.values()])
    surface = np.concatenate([load_mesh(f).vertices for f in
                              glob.glob(os.path.join(converted_root, sid, folder, '*.stl'))], 0)
    tree = KDTree(clik_pts)

    per_class, gt_all = {}, []
    for uid, pts in gt.items():
        for name, p in pts.items():
            per_class.setdefault(name, []).append(float(tree.query(p)[0]))
            gt_all.append(p)
    gt_all = np.array(gt_all)

    rng = np.random.default_rng(seed)
    base = []
    for _ in range(trials):
        idx = rng.choice(len(surface), min(len(clik_pts), len(surface)), replace=False)
        base.append(np.median(KDTree(surface[idx]).query(gt_all)[0]))
    return {'sid': sid, 'per_class': per_class, 'baseline': float(np.mean(base))}


def main():
    """Aggregate landmark accuracy over the annotated subjects."""
    ap = argparse.ArgumentParser(description="CLIK landmarks vs the dataset's own annotations.")
    ap.add_argument('--converted', required=True)
    ap.add_argument('--dataset', required=True, help='root holding Landmark_annotation/')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stage', default='ori', choices=['ori', 'final'],
                    help='pre-treatment (ori) or post-treatment (final) crowns')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sids = sorted(os.path.basename(d) for d in
                  glob.glob(os.path.join(args.dataset, 'Landmark_annotation', '*'))
                  if os.path.isdir(d) and os.path.exists(os.path.join(args.converted, os.path.basename(d), 'center.json')))
    if args.limit:
        sids = sids[:args.limit]

    print('Loading detectors...')
    models = load_detection(**CKPTS)

    agg, bases = {}, []
    for i, sid in enumerate(sids, 1):
        r = evaluate_subject(models, sid, args.dataset, args.converted, stage=args.stage)
        if not r:
            continue
        for k, v in r['per_class'].items():
            agg.setdefault(k, []).extend(v)
        bases.append(r['baseline'])
        allv = [x for v in r['per_class'].values() for x in v]
        print(f"  [{i}/{len(sids)}] {sid}: median {np.median(allv):5.2f} mm (baseline {r['baseline']:5.2f})")

    print(f'\n=== {len(bases)} subjects, stage {args.stage} ===')
    base = float(np.mean(bases))
    print(f"  {'class':10s} {'n':>6s} {'CLIK(mm)':>10s} {'baseline':>10s} {'skill':>7s}")
    allv = []
    for k in sorted(agg):
        v = np.array(agg[k]); allv.append(v)
        print(f'  {k:10s} {len(v):6d} {np.median(v):10.2f} {base:10.2f} {1 - np.median(v)/base:7.2f}')
    v = np.concatenate(allv)
    print(f"  {'OVERALL':10s} {len(v):6d} {np.median(v):10.2f} {base:10.2f} {1 - np.median(v)/base:7.2f}")
    if args.out:
        json.dump({'per_class': {k: float(np.median(agg[k])) for k in agg},
                   'overall_median_mm': float(np.median(v)), 'baseline_mm': base},
                  open(args.out, 'w'), indent=2)
        print(f'  -> {args.out}')


if __name__ == '__main__':
    main()
