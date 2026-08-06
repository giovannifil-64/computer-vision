"""
seed_variability
================
Quantify how much of CLIK's alignment error is stochastic. The diffusion model is
generative, so each run draws a different sample; the evaluation so far used a
single seed. Re-running the same subjects with several seeds separates the part
of the error that moves with the random draw from the part that is systematic.

Functions
---------
- `metrics_for(sid, converted_root, output_root)`: Alignment metrics for one subject and run.
- `main()`: Aggregate the spread across seeds and compare it with the baseline gap.

Example
-------
```bash
python seed_variability.py --converted ../data/Data_prepost \\
    --runs ../output/Output_prepost ../output/seed_2 ../output/seed_3
```

Notes
-----
- The spread is reported both per subject (how much one prediction wobbles) and
  on the aggregate (whether the headline numbers would change with another seed).
- If the spread is small relative to the distance from the "no movement"
  reference, the conclusion drawn from a single run is safe.
"""
import os
import glob
import json
import argparse
import numpy as np


def metrics_for(sid, converted_root, output_root):
    """
    Alignment metrics of one subject for one inference run.

    Parameters
    ----------
    - `sid (str)`: Subject id.
    - `converted_root (str)`: Converter output root.
    - `output_root (str)`: Folder of the run to score.

    Returns
    -------
    - `dict` or `None`: `{'rot', 'trans', 'pcd', 'rot_b', 'pcd_b'}` medians, or `None`.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluate_alignment import evaluate_subject
    rep = evaluate_subject(sid, converted_root, output_root)
    if not rep:
        return None
    s = rep['summary']
    return {'rot': s['rot_err'], 'trans': s['trans_err'], 'pcd': s['pcd_err'],
            'rot_b': s['rot_baseline'], 'pcd_b': s['pcd_baseline']}


def main():
    """Compare the same subjects across several inference runs (seeds)."""
    ap = argparse.ArgumentParser(description="How stable is CLIK's prediction across random seeds?")
    ap.add_argument('--converted', required=True)
    ap.add_argument('--runs', nargs='+', required=True, help='output folders, one per seed')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    # subjects present in every run
    common = None
    for r in args.runs:
        have = {os.path.basename(os.path.dirname(os.path.dirname(p)))
                for p in glob.glob(os.path.join(r, '*', 'results', 'transformation.json'))}
        common = have if common is None else (common & have)
    common = sorted(common or [])
    print(f'{len(args.runs)} run, {len(common)} soggetti presenti in tutti\n')

    per_subject = {}
    for sid in common:
        vals = [metrics_for(sid, args.converted, r) for r in args.runs]
        vals = [v for v in vals if v]
        if len(vals) < 2:
            continue
        per_subject[sid] = {
            'pcd': [v['pcd'] for v in vals], 'rot': [v['rot'] for v in vals],
            'pcd_b': vals[0]['pcd_b'], 'rot_b': vals[0]['rot_b'],
        }

    pcd = np.array([v['pcd'] for v in per_subject.values()])      # (subjects, seeds)
    rot = np.array([v['rot'] for v in per_subject.values()])
    pcd_b = np.array([v['pcd_b'] for v in per_subject.values()])
    rot_b = np.array([v['rot_b'] for v in per_subject.values()])

    print('=== dispersione fra seed (per soggetto) ===')
    print(f'  point cloud: dev.std media {pcd.std(axis=1).mean():.3f} mm  '
          f'(escursione media {np.ptp(pcd, axis=1).mean():.3f} mm)')
    print(f'  rotazione  : dev.std media {rot.std(axis=1).mean():.3f} gradi  '
          f'(escursione media {np.ptp(rot, axis=1).mean():.3f} gradi)')

    print('\n=== effetto sul risultato aggregato ===')
    for i, r in enumerate(args.runs):
        print(f'  {os.path.basename(r):18s} point cloud mediana {np.median(pcd[:, i]):.3f} mm | '
              f'rotazione mediana {np.median(rot[:, i]):.3f} gradi')
    print(f'  {"baseline":18s} point cloud mediana {np.median(pcd_b):.3f} mm | '
          f'rotazione mediana {np.median(rot_b):.3f} gradi')

    gap = np.median(pcd, axis=1) - pcd_b
    print(f'\n  distanza dal baseline (point cloud): {np.median(gap):+.3f} mm')
    print(f'  dispersione fra seed                : {pcd.std(axis=1).mean():.3f} mm')
    ratio = abs(np.median(gap)) / max(pcd.std(axis=1).mean(), 1e-9)
    print(f'  -> il divario e {ratio:.1f}x la dispersione '
          f'({"conclusione stabile" if ratio > 3 else "attenzione: rumore non trascurabile"})')

    if args.out:
        json.dump({'n_subjects': len(per_subject), 'runs': args.runs,
                   'pcd_std_mm': float(pcd.std(axis=1).mean()),
                   'rot_std_deg': float(rot.std(axis=1).mean()),
                   'pcd_median_per_run': [float(np.median(pcd[:, i])) for i in range(pcd.shape[1])],
                   'pcd_baseline_median': float(np.median(pcd_b)),
                   'per_subject': {k: v['pcd'] for k, v in per_subject.items()}},
                  open(args.out, 'w'), indent=2)
        print(f'  -> {args.out}')


if __name__ == '__main__':
    main()
