"""
split_overlap
=============
Look for test cases whose crowns also appear in the training data.

PrePostOrthodontic treats every phase of an aligner treatment as a separate case and
does not distribute patient identifiers, so the official split cannot be checked for
patients on both sides of it. What can be checked is whether the same crowns appear
on both sides. A crown keeps its shape whatever pose the conversion gives it, so each
tooth is summarised by descriptors that do not depend on pose, the three principal
extents of points sampled on its surface and its area, and two cases share a crown
when these coincide. The descriptors of the same mesh coincide exactly, while two
different scans of the same patient differ like the scans of two patients, so this
finds reused scans and not every pair of phases of one patient.

Functions
---------
- `crown_descriptors(sid, stage, converted)`: Pose-independent descriptors of every crown of one case.
- `shared_crowns(a, b)`: How many crowns two descriptor sets have in common.
- `main()`: Compare every test case with every training case and report the overlaps.

Example
-------
```bash
python split_overlap.py --split ../output/abl_base/split.json \\
    --test-ids ../output/asis_split --out ../output/split_overlap.json
```

Notes
-----
- Third molars are left out, since CLIK does not use them.
- A pair is reported when at least half of the crowns of the test case have an
  identical counterpart in the training case.
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
STEP2 = os.path.join(HERE, '..', '..', 'step2_alignment')
sys.path.insert(0, os.path.join(STEP2, 'scripts'))
from meshes import load_mesh                                          # noqa: E402

THIRD_MOLARS = {1, 16, 17, 32}


def crown_descriptors(sid, stage, converted, samples=1500):
    """
    Pose-independent descriptors of every crown of one case.

    Parameters
    ----------
    - `sid (str)`: Case id.
    - `stage (str)`: `"initial"` or `"final"`.
    - `converted (str)`: Converter output root.
    - `samples (int, optional)`: Surface points per crown. Default `1500`.

    Returns
    -------
    - `dict`: `{tooth_id: np.ndarray}` with the three principal extents and the area.

    Notes
    -----
    - The surface is sampled with a fixed seed, so the same mesh always yields the
      same points, only moved by the rigid pose of the case, and therefore the same
      descriptors.
    """
    out = {}
    for f in glob.glob(os.path.join(converted, sid, stage, '*.stl')):
        tid = int(os.path.basename(f)[:-4])
        if tid in THIRD_MOLARS:
            continue
        mesh = load_mesh(f)
        pts = trimesh.sample.sample_surface(mesh, samples, seed=0)[0]
        extents = np.sqrt(np.sort(np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T)))[::-1])
        out[tid] = np.r_[extents, mesh.area]
    return out


def shared_crowns(a, b):
    """
    Count the crowns of `a` that have an identical counterpart in `b`.

    Parameters
    ----------
    - `a (dict)`: Descriptors of one case.
    - `b (dict)`: Descriptors of another case.

    Returns
    -------
    - `tuple`: `(identical, compared)` over the tooth positions present in both.
    """
    common = set(a) & set(b)
    return sum(np.allclose(a[t], b[t], rtol=1e-6) for t in common), len(common)


def main():
    """Compare every test case with every training case and report shared crowns."""
    ap = argparse.ArgumentParser(description='Crowns shared between test and training cases.')
    ap.add_argument('--split', required=True, help='split.json of a training run')
    ap.add_argument('--test-ids', required=True,
                    help='a scored run whose case folders define the test cases')
    ap.add_argument('--out', required=True, help='JSON file to write')
    ap.add_argument('--converted', default=os.path.join(STEP2, 'data', 'Data_prepost'))
    args = ap.parse_args()

    split = json.load(open(args.split))
    train = [s for key in ('train', 'val', 'validation') for s in split.get(key, [])]
    tests = sorted(d for d in os.listdir(args.test_ids) if d.isdigit())
    desc = {s: crown_descriptors(s, 'initial', args.converted) for s in train + tests}

    pairs = []
    for t in tests:
        for s in train:
            same, compared = shared_crowns(desc[t], desc[s])
            if compared and same >= compared / 2:
                same_f, compared_f = shared_crowns(crown_descriptors(t, 'final', args.converted),
                                                   crown_descriptors(s, 'final', args.converted))
                pairs.append({'test': t, 'train': s, 'initial': [same, compared],
                              'target': [same_f, compared_f]})
                print(f'  test {t} / training {s}: {same} of {compared} initial crowns '
                      f'and {same_f} of {compared_f} target crowns identical')
    print(f'{len(pairs)} of {len(tests)} test cases share their crowns with a training case')
    json.dump(pairs, open(args.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
