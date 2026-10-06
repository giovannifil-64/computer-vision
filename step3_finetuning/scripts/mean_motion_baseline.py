"""
mean_motion_baseline
====================
A reference that predicts, for every tooth, the average motion of that tooth
position in the training cases.

The no-movement reference asks whether a model does better than leaving the teeth
where they are. A model that ignores its noisy input behaves as a regression from
the initial dentition to one treated dentition, so a second, stronger question is
whether it does better than a population average that ignores the patient
altogether. This script builds that average from the training cases and writes it in
CLIK's own output format, so that the step-2 evaluation scores it unchanged.

Functions
---------
- `tooth_motions(sid, converted)`: Rotation and centroid displacement of every tooth of one case.
- `average_motions(sids, converted, workers)`: Mean rotation and displacement per tooth position.
- `write_prediction(sid, converted, mean, out)`: The average motion applied to one test case.
- `main()`: Build the average on the training split and write it for the test cases.

Example
-------
```bash
python mean_motion_baseline.py --split ../output/abl_base/split.json \\
    --test-ids ../output/asis_split --out ../output/mean_motion
python ../../step2_alignment/scripts/evaluate_alignment.py \\
    --converted ../../step2_alignment/data/Data_prepost --output ../output/mean_motion
```

Notes
-----
- The motion of a tooth is split into a rotation about its own centroid and a
  displacement of that centroid, so the average does not depend on where the tooth
  sits in the dentition. The mean rotation is the average of the rotation matrices
  projected back onto the rotation group, which is adequate for rotations of a few
  degrees such as these.
- Only teeth whose pretreatment and target meshes share their vertices are used,
  the same rule the evaluation applies, and third molars are left out because CLIK
  does not cover them.
"""
import os
import json
import glob
import argparse
import numpy as np
from multiprocessing import Pool

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                'step2_alignment', 'scripts'))
from meshes import load_mesh                                          # noqa: E402

THIRD_MOLARS = {1, 16, 17, 32}
STEP2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'step2_alignment')


def _kabsch(a, b):
    """Rotation and translation carrying the points `a` onto `b` in least squares."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cb - r @ ca


def tooth_motions(sid, converted):
    """
    Rotation and centroid displacement of every usable tooth of one case.

    Parameters
    ----------
    - `sid (str)`: Case id.
    - `converted (str)`: Converter output root, with `initial/` and `final/` meshes.

    Returns
    -------
    - `dict`: `{tooth_id: (rotation (3, 3), displacement (3,))}`.
    """
    out = {}
    for f in glob.glob(os.path.join(converted, sid, 'initial', '*.stl')):
        tid = int(os.path.basename(f)[:-4])
        g = os.path.join(converted, sid, 'final', f'{tid}.stl')
        if tid in THIRD_MOLARS or not os.path.exists(g):
            continue
        a = load_mesh(f, process=False).vertices
        b = load_mesh(g, process=False).vertices
        if a.shape != b.shape:
            continue
        r, _ = _kabsch(a, b)
        out[tid] = (r, b.mean(0) - a.mean(0))
    return out


def _motions_job(args):
    return tooth_motions(*args)


def average_motions(sids, converted, workers=4):
    """
    Mean rotation and mean centroid displacement of every tooth position.

    Parameters
    ----------
    - `sids (list)`: Training case ids.
    - `converted (str)`: Converter output root.
    - `workers (int, optional)`: Parallel processes. Default `4`.

    Returns
    -------
    - `dict`: `{tooth_id: (mean_rotation, mean_displacement, count)}`.
    """
    with Pool(workers) as pool:
        per_case = pool.map(_motions_job, [(s, converted) for s in sids])
    rots, disps = {}, {}
    for motions in per_case:
        for tid, (r, d) in motions.items():
            rots.setdefault(tid, []).append(r)
            disps.setdefault(tid, []).append(d)
    mean = {}
    for tid in rots:
        u, _, vt = np.linalg.svd(np.mean(rots[tid], axis=0))
        r = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        mean[tid] = (r, np.mean(disps[tid], axis=0), len(rots[tid]))
    return mean


def write_prediction(sid, converted, mean, out):
    """
    Apply the average motion of each tooth position to one test case.

    Parameters
    ----------
    - `sid (str)`: Test case id.
    - `converted (str)`: Converter output root.
    - `mean (dict)`: Output of `average_motions`.
    - `out (str)`: Root of the run folder to write.

    Returns
    -------
    - `int`: Number of teeth written.

    Notes
    -----
    - The file mirrors CLIK's `transformation.json`: one 4x4 matrix per tooth,
      acting on the converted pretreatment mesh.
    """
    transforms = {}
    for f in glob.glob(os.path.join(converted, sid, 'initial', '*.stl')):
        tid = int(os.path.basename(f)[:-4])
        if tid not in mean:
            continue
        r, d, _ = mean[tid]
        c = load_mesh(f, process=False).vertices.mean(0)
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = c + d - r @ c
        transforms[f'Tooth-{tid}'] = m.tolist()
    folder = os.path.join(out, sid, 'results')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'transformation.json'), 'w') as fh:
        json.dump(transforms, fh, indent=4)
    return len(transforms)


def main():
    """Average the training motions and write them for every test case."""
    ap = argparse.ArgumentParser(description='Average-motion reference for the test cases.')
    ap.add_argument('--split', required=True, help='split.json of a training run')
    ap.add_argument('--test-ids', required=True,
                    help='a scored run whose case folders define the test cases')
    ap.add_argument('--out', required=True, help='run folder to write')
    ap.add_argument('--converted', default=os.path.join(STEP2, 'data', 'Data_prepost'))
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    train = json.load(open(args.split))['train']
    mean = average_motions(train, args.converted, args.workers)
    print(f'average motion from {len(train)} training cases, {len(mean)} tooth positions')
    for tid in sorted(mean):
        r, d, n = mean[tid]
        angle = np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
        print(f'  tooth {tid:2d}: {n:4d} teeth, rotation {angle:4.2f} deg, '
              f'displacement {np.linalg.norm(d):4.2f} mm')

    tests = sorted(d for d in os.listdir(args.test_ids) if d.isdigit())
    teeth = sum(write_prediction(s, args.converted, mean, args.out) for s in tests)
    print(f'wrote {teeth} teeth for {len(tests)} test cases into {args.out}')


if __name__ == '__main__':
    main()
