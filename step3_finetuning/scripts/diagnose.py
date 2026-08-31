"""
diagnose
========
Ask of a trained model not whether its median error went down, but what kind of
mistake it is making. The alignment metrics say how far off a prediction is; they
do not say whether the model is timid, misdirected, or simply noisy, and those
call for different fixes.

Three questions are answered here. How often does the model actually beat leaving
the teeth alone, per patient rather than on a median that can hide half the cases.
How much of its landmark error survives the rigid fit, which tells apart noisy
landmarks from a genuinely wrong motion. And how big are the motions it predicts
compared with the real ones, which is where systematic timidity shows up: a
squared-error loss on an ambiguous target pulls a model towards the average of the
plausible answers, and an average of motions is always smaller than the motions
themselves.

Functions
---------
- `beats_baseline(folder)`: Share of patients where the model beats no movement.
- `motion_stats(predicted, tensors, limit)`: Error decomposition and motion sizes.
- `main()`: Report both for one run.

Example
-------
```bash
python diagnose.py --predicted ../output/individual_s200_pred.npz \\
    --scored ../output/individual_s200 --tensors ../data/test_tensors
```

Notes
-----
- The rigid motion compared here is the one the pipeline actually applies, solved
  from the predicted landmarks, against the one solved from the exact targets. It
  is therefore the error of the transform, not of the landmarks that produced it.
"""
import os
import sys
import csv
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(BASE, 'step2_alignment', 'scripts'))

from stage_a_prepare import TEETH, tooth_landmark_indices, SCALE
from evaluate_alignment import kabsch, rotation_error

METRICS = (('rot_err', 'rot_baseline', 'rotation'),
           ('trans_err', 'trans_baseline', 'translation'),
           ('pcd_err', 'pcd_baseline', 'point cloud'))


def beats_baseline(folder):
    """
    How often the model is better than not moving the teeth, patient by patient.

    Parameters
    ----------
    - `folder (str)`: A scored run, holding `alignment_metrics.csv`.

    Returns
    -------
    - `dict`: Metric label to `(median, baseline_median, share_better, n)`.

    Notes
    -----
    - The share matters more than the median. A model can have the better median
      and still make things worse for half the patients, and for a clinical tool
      that is the number to quote.
    """
    with open(os.path.join(folder, 'alignment_metrics.csv')) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for key, base_key, label in METRICS:
        y = np.array([float(r[key]) for r in rows])
        x = np.array([float(r[base_key]) for r in rows])
        out[label] = (np.median(y), np.median(x), float((y < x).mean()), len(y))
    return out


def motion_stats(predicted, tensors, limit=100):
    """
    Decompose the landmark error and compare predicted motions with real ones.

    Parameters
    ----------
    - `predicted (str)`: `.npz` written by stage C.
    - `tensors (str)`: Folder of prepared tensors, for the exact targets.
    - `limit (int, optional)`: Subjects to include. Default `100`.

    Returns
    -------
    - `dict`: Millimetre and degree figures, plus the ratios and correlations.
    """
    data = np.load(predicted)
    ids, pred = [str(s) for s in data['ids']], data['landmarks']
    total, non_rigid, applied, identity = [], [], [], []
    size_pred, size_true, ang_pred, ang_true = [], [], [], []

    for i, sid in enumerate(ids[:limit]):
        path = os.path.join(tensors, f'{sid}.npz')
        if not os.path.exists(path):
            continue
        z = np.load(path)
        mask, target, cond, p_all = z['mask'], z['target'], z['cond'][:, :3], pred[i].T
        for j, _ in enumerate(TEETH):
            if not mask[j]:
                continue
            idx = tooth_landmark_indices(j)
            p, t, c = p_all[idx] * SCALE, target[idx] * SCALE, cond[idx] * SCALE
            total.append(np.median(np.linalg.norm(p - t, axis=1)))
            r, tr, _ = kabsch(p, t)
            non_rigid.append(np.median(np.linalg.norm((r @ p.T).T + tr - t, axis=1)))

            rp, tp, _ = kabsch(c, p)          # the motion the pipeline applies
            rt, tt, _ = kabsch(c, t)          # the motion that actually happened
            truth = (rt @ c.T).T + tt
            applied.append(np.median(np.linalg.norm((rp @ c.T).T + tp - truth, axis=1)))
            identity.append(np.median(np.linalg.norm(c - truth, axis=1)))
            size_pred.append(np.linalg.norm(p.mean(0) - c.mean(0)))
            size_true.append(np.linalg.norm(t.mean(0) - c.mean(0)))
            ang_pred.append(rotation_error(np.eye(3), rp))
            ang_true.append(rotation_error(np.eye(3), rt))

    med = np.median
    return {'teeth': len(total),
            'landmark_error': med(total),
            'non_rigid_part': med(non_rigid),
            'motion_error': med(applied),
            'no_movement': med(identity),
            'translation_ratio': med(size_pred) / med(size_true),
            'rotation_ratio': med(ang_pred) / med(ang_true),
            'translation_corr': np.corrcoef(size_pred, size_true)[0, 1],
            'rotation_corr': np.corrcoef(ang_pred, ang_true)[0, 1]}


def main():
    """Report what kind of mistake one run is making."""
    ap = argparse.ArgumentParser(description='Diagnose a run beyond its median error.')
    ap.add_argument('--predicted', required=True, help='.npz written by stage C')
    ap.add_argument('--scored', required=True, help='folder holding alignment_metrics.csv')
    ap.add_argument('--tensors', required=True)
    ap.add_argument('--limit', type=int, default=100, help='subjects for the motion analysis')
    args = ap.parse_args()

    print('\nhow often the model beats leaving the teeth alone')
    for label, (median, base, share, n) in beats_baseline(args.scored).items():
        print(f'  {label:14s} {median:6.2f} vs {base:5.2f}   {share * 100:3.0f}% of {n} patients')

    s = motion_stats(args.predicted, args.tensors, args.limit)
    print(f'\nwhere the error lives, per tooth over {s["teeth"]} teeth')
    print(f'  landmark error                    {s["landmark_error"]:.2f} mm')
    print(f'  of which not a rigid motion       {s["non_rigid_part"]:.2f} mm')
    print(f'  error of the motion applied       {s["motion_error"]:.2f} mm')
    print(f'  error of not moving at all        {s["no_movement"]:.2f} mm')

    print('\nis the model timid?')
    print(f'  predicted translation / real      {s["translation_ratio"]:.2f}   '
          f'(correlation {s["translation_corr"]:.2f})')
    print(f'  predicted rotation / real         {s["rotation_ratio"]:.2f}   '
          f'(correlation {s["rotation_corr"]:.2f})')
    print('  a ratio well under 1 with a decent correlation means the model has found '
          'the\n  right direction and is holding back, which is what a squared-error '
          'loss does\n  when several answers are plausible.\n')


if __name__ == '__main__':
    main()
