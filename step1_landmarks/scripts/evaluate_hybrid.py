"""
evaluate_hybrid
===============
Score the hybrid landmarks class by class on every annotated patient.

`evaluate.py` measures coverage: for each annotated landmark, the distance to the
nearest CLIK landmark of any type. That says whether CLIK puts a landmark near every
annotated point, not whether the landmark it puts there is the right one. The hybrid
of `hybrid.py` assigns a 3DTeethLand class to every point, the four classes CLIK
covers through a map learned on one patient and the two it does not cover from the
crown geometry, so here each annotated landmark is compared only with hybrid
landmarks of its own class.

Functions
---------
- `score_patient(pid, ...)`: Class-specific distances and random baseline for one patient.
- `main()`: Learn the map, score every other patient and write the summary.

Example
-------
```bash
python evaluate_hybrid.py --src ../data/Teeth3DS_input --gt <osfstorage-archive> \\
    --converted ../data/Data_teeth3ds_gt --output ../output/Output_teeth3ds_gt
```

Notes
-----
- The map is learned on the first annotated patient, the same rule `run_all.py`
  uses, and that patient is left out of the summary so every figure is measured on
  patients the map has not seen.
- The skill score has the same definition as in `evaluate.py`: one minus the ratio
  between the hybrid distance and the distance obtained with as many random points
  on the same crowns, computed per patient and summarised by the median.
"""
import os
import glob
import json
import argparse
import numpy as np

from meshes import load_mesh
from common import load_center, load_gt_landmarks, find_gt_files
from evaluate import nearest_distances, random_baseline, skill_score
from hybrid import learn_class_map, build_hybrid

CLASSES = ('Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint', 'FacialPoint')


def score_patient(pid, converted_root, output_root, gt_root, class_map, trials=20):
    """
    Class-specific distances between annotations and hybrid landmarks for one patient.

    Parameters
    ----------
    - `pid (str)`: Patient id.
    - `converted_root (str)`: Converter output (crowns and `center.json`).
    - `output_root (str)`: CLIK output (detected landmarks).
    - `gt_root (str)`: Root of `osfstorage-archive`.
    - `class_map (dict)`: Output of `learn_class_map`.
    - `trials (int, optional)`: Random-baseline trials. Default `20`.

    Returns
    -------
    - `dict` or `None`: Per class `{'n', 'median', 'random_median', 'skill'}`, or
      `None` when the patient has no annotation.
    """
    center = load_center(converted_root, pid)
    gt = load_gt_landmarks(gt_root, pid, frame='center', center=center)
    if gt is None:
        return None
    gt_pts, gt_cls = gt
    hyb = build_hybrid(converted_root, output_root, pid, class_map)
    surface = np.concatenate(
        [load_mesh(f).vertices
         for f in glob.glob(os.path.join(converted_root, pid, 'initial', '*.stl'))], 0)

    out = {}
    for cl in CLASSES:
        rows = [i for i, c in enumerate(gt_cls) if c == cl]
        pts = np.array([p for _, c, p in hyb if c == cl])
        if not rows or not len(pts):
            continue
        d = nearest_distances(gt_pts[rows], pts)
        med = float(np.median(d))
        base = random_baseline(gt_pts[rows], surface, len(pts), trials)
        out[cl] = {'n': len(rows), 'median': med, 'random_median': base,
                   'skill': skill_score(med, base)}
    return out


def main():
    """Learn the class map on the first annotated patient and score all the others."""
    ap = argparse.ArgumentParser(description='Class-specific score of the hybrid landmarks.')
    ap.add_argument('--src', required=True, help='Teeth3DS source folder (patient subfolders)')
    ap.add_argument('--gt', required=True, help='osfstorage-archive root')
    ap.add_argument('--converted', required=True, help='converter output')
    ap.add_argument('--output', required=True, help='CLIK output')
    args = ap.parse_args()

    pids = sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.src, '*'))
                  if os.path.isdir(d))
    map_pid = next(p for p in pids if find_gt_files(args.gt, p))
    class_map = learn_class_map(args.converted, args.output, args.gt, map_pid)

    per_patient = {}
    for pid in pids:
        if pid == map_pid:
            continue
        rep = score_patient(pid, args.converted, args.output, args.gt, class_map)
        if rep:
            per_patient[pid] = rep

    summary = {'map_patient': map_pid, 'patients': len(per_patient), 'per_class': {}}
    print(f'map learned on {map_pid}, scored on {len(per_patient)} other patients\n')
    print(f'  {"class":12s} {"median mm":>10s} {"random mm":>10s} {"skill":>7s} {"sd":>6s}')
    for cl in CLASSES:
        reps = [r[cl] for r in per_patient.values() if cl in r]
        med = float(np.median([r['median'] for r in reps]))
        base = float(np.median([r['random_median'] for r in reps]))
        skills = np.array([r['skill'] for r in reps])
        summary['per_class'][cl] = {'patients': len(reps), 'median_mm': med,
                                    'random_median_mm': base,
                                    'skill_median': float(np.median(skills)),
                                    'skill_sd': float(skills.std(ddof=1))}
        print(f'  {cl:12s} {med:10.2f} {base:10.2f} {np.median(skills):+7.2f} '
              f'{skills.std(ddof=1):6.2f}')

    out = os.path.join(args.output, 'hybrid_summary.json')
    json.dump({'summary': summary, 'per_patient': per_patient}, open(out, 'w'), indent=2)
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
