"""
landmark_consistency
====================
Ask whether CLIK's landmarks mean the same thing on different people.

A landmark earns the name only if it marks the same anatomical spot from one
patient to the next. The evaluation in step 1 answered a related question by
comparing against expert annotations, but that needs annotated scans and only 85
were available. This asks the question from the data alone, on every patient
there is.

For one tooth type, the landmark sets of many patients are superimposed by
removing position, orientation and size, which leaves only differences of shape.
Whatever spread survives is either real anatomical variation or the detector
placing the point somewhere else. The tooth centroid gives something to read that
against, because CLIK does not predict it: it is the mean of the tooth's vertices,
so its spread comes from how much the crown varies and not from any choice the
network made.

It is a reference and not a lower bound. Measured over 250 patients, 61 of the 228
predicted landmarks are steadier than their own centroid, which is what a cusp tip
should be: a sharper feature than the average of a whole crown, whose extent
depends on where the segmentation cut it. The informative tail is the other one.
Thirty one landmarks scatter more than twice their centroid, and those are not
marking an anatomical point at all.

Functions
---------
- `collect(tensor_dir, limit)`: Landmark sets per tooth, across patients.
- `superimpose(sets, iterations)`: Remove position, orientation and size.
- `spread(sets)`: Millimetres each landmark moves between patients.
- `main()`: Report the spread per tooth, next to the centroid.

Example
-------
```bash
python landmark_consistency.py --tensors ../data/train_tensors --limit 250
```

Notes
-----
- Coordinates come from the prepared tensors, so this needs no meshes and no
  annotations, and it runs in seconds.
- The spread is a mean distance from the average shape, in millimetres, after the
  shapes have been scaled back to the average tooth size.
"""
import os
import sys
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stage_a_prepare import TEETH, tooth_landmark_indices, SCALE

# Landmark ids per tooth type, mirroring the tables in `s1_LandmarkDetection`.
# Id 0 is the centroid, which the detector does not predict.
LANDMARK_IDS = {
    'incisor': ['0', '3', '4', '13', '15', '16', '34', '35'],
    'canine': ['0', '3', '4', '5', '19', '20', '34', '35'],
    'premolar': ['0', '4', '6', '7', '21', '22', '23', '34', '35'],
    'molar': ['0', '8', '9', '10', '11', '21', '22', '25', '30', '34', '35'],
}
TOOTH_TYPE = {}
for _ids, _name in (({7, 8, 9, 10, 23, 24, 25, 26}, 'incisor'),
                    ({6, 11, 22, 27}, 'canine'),
                    ({4, 5, 12, 13, 20, 21, 28, 29}, 'premolar'),
                    ({2, 3, 14, 15, 18, 19, 30, 31}, 'molar')):
    TOOTH_TYPE.update({t: _name for t in _ids})


def collect(tensor_dir, limit=0):
    """
    Gather each tooth's landmark set from every patient that has it.

    Parameters
    ----------
    - `tensor_dir (str)`: Folder of `.npz` files written by stage A.
    - `limit (int, optional)`: Patients to read. Default `0`, meaning all.

    Returns
    -------
    - `dict`: Tooth id to a list of `(n_landmarks, 3)` arrays in millimetres.
    """
    files = sorted(glob.glob(os.path.join(tensor_dir, '*.npz')))
    if limit:
        files = files[:limit]
    out = {}
    for path in files:
        data = np.load(path)
        points, mask = data['cond'][:, :3] * SCALE, data['mask']
        for i, tid in enumerate(TEETH):
            if mask[i]:
                out.setdefault(tid, []).append(points[tooth_landmark_indices(i)])
    return out


def superimpose(sets, iterations=5):
    """
    Superimpose landmark sets, leaving only differences of shape.

    Parameters
    ----------
    - `sets (list)`: One `(n, 3)` array per patient, all with the same `n`.
    - `iterations (int, optional)`: Refinements of the average shape. Default `5`.

    Returns
    -------
    - `tuple`: `(aligned, mean_size)` with `aligned` of shape `(patients, n, 3)`
      at unit size, and `mean_size` the average size that was divided out.

    Notes
    -----
    - Each set is centred and scaled to unit size, then rotated onto the running
      average; the average is recomputed and the pass repeated. Rotations are
      solved by singular value decomposition, with the reflection case rejected so
      that a left tooth is never matched onto a mirrored right one.
    """
    x = np.stack(sets)
    x = x - x.mean(axis=1, keepdims=True)
    sizes = np.linalg.norm(x, axis=(1, 2), keepdims=True)
    x = x / sizes

    reference = x[0]
    for _ in range(iterations):
        rotated = []
        for shape in x:
            u, _, vt = np.linalg.svd(shape.T @ reference)
            if np.linalg.det(u @ vt) < 0:
                u[:, -1] *= -1
            rotated.append(shape @ (u @ vt))
        x = np.stack(rotated)
        reference = x.mean(axis=0)
        reference /= np.linalg.norm(reference)
    return x, float(sizes.mean())


def spread(sets):
    """
    How far each landmark moves between patients, in millimetres.

    Parameters
    ----------
    - `sets (list)`: One `(n, 3)` array per patient.

    Returns
    -------
    - `tuple`: `(per_landmark, radius)`, both in millimetres. `radius` is the
      average distance of the landmarks from the tooth's centre, which gives the
      spread a scale to be read against.
    """
    aligned, size = superimpose(sets)
    shapes = aligned * size
    average = shapes.mean(axis=0)
    return (np.linalg.norm(shapes - average, axis=2).mean(axis=0),
            float(np.linalg.norm(average, axis=1).mean()))


def main():
    """Report, per tooth, how far each landmark moves between patients."""
    ap = argparse.ArgumentParser(
        description='Do CLIK\'s landmarks mark the same spot on different people?')
    ap.add_argument('--tensors', required=True, help='folder of prepared .npz files')
    ap.add_argument('--limit', type=int, default=250, help='patients to read')
    ap.add_argument('--min-patients', type=int, default=30,
                    help='skip a tooth seen in fewer patients than this')
    args = ap.parse_args()

    per_tooth = collect(args.tensors, args.limit)
    print(f'\n{len(per_tooth)} teeth, from up to {args.limit} patients')
    print('spread of each landmark between patients, in mm, after position, '
          'orientation\nand size have been removed. Landmark 0 is the centroid, '
          'which the detector does not\npredict; it is a reference, not a floor, '
          'and a sharp feature can beat it.\n')

    for tid in sorted(per_tooth):
        sets = per_tooth[tid]
        if len(sets) < args.min_patients:
            continue
        values, radius = spread(sets)
        labels = LANDMARK_IDS[TOOTH_TYPE[tid]]
        worst = max(values[1:]) / values[0] if values[0] else float('nan')
        print(f'  tooth {tid:2d} ({TOOTH_TYPE[tid]:<9s}) {len(sets):4d} patients, '
              f'radius {radius:.1f} mm, worst landmark {worst:.1f}x the centroid')
        print('    ' + '  '.join(f'{lab}={val:.1f}' for lab, val in zip(labels, values)))
    print()


if __name__ == '__main__':
    main()
