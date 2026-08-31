"""
stage_d_transform
=================
Stage D of the fine-tuning pipeline: turn the landmarks predicted in stage C into
the aligned dentition, writing exactly what CLIK's own inference writes.

The rigid solve and the export are not reimplemented here: they call CLIK's
`s3_SolveMatrix` directly, so the output is identical in format and in numbers to
what `infer_crown.py` produces. That is what lets the evaluation, the figures and
the spreadsheet built for step 2 be reused untouched, and what makes the split
pipeline comparable with the single-process one.

This stage runs locally: it needs the meshes, and the solve is done in float64,
which several GPUs handle poorly.

Functions
---------
- `transform_subject(sid, cond, pred, converted_root, out_root)`: Align one dentition.
- `main()`: Apply the predicted landmarks of every subject and write the results.

Example
-------
```bash
python stage_d_transform.py --predicted ../output/base_loss/predicted_landmarks.npz \\
    --tensors ../data/test_tensors --out ../output/base_loss/inference
```

Notes
-----
- Landmarks arrive in normalised units; the scaling back to millimetres is handled
  inside CLIK's solver, which multiplies the translation by the same `scale`.
- Teeth absent from a dentition are skipped, exactly as in the original pipeline.
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
STEP2 = os.path.join(BASE, 'step2_alignment')
sys.path.insert(0, os.path.join(STEP2, 'scripts'))


def _find_clik_root():
    """Locate the CLIK-Diffusion checkout (env `CLIK_ROOT`, else walk up)."""
    env = os.environ.get('CLIK_ROOT')
    if env:
        return env
    here = HERE
    while here != os.path.dirname(here):
        cand = os.path.join(here, 'CLIK-Diffusion')
        if os.path.isdir(os.path.join(cand, 'Code')):
            return cand
        here = os.path.dirname(here)
    raise RuntimeError('CLIK-Diffusion checkout not found; set CLIK_ROOT')


ROOT = _find_clik_root()
sys.path.insert(0, os.path.join(ROOT, 'Code'))

# CLIK's own modules only resolve once its Code folder is on the path above
from s3_SolveMatrix import solve_and_trans_mesh, save_mesh, save_transformation
from meshes import load_mesh


def transform_subject(sid, cond, pred, converted_root, out_root):
    """
    Apply one subject's predicted landmarks to its teeth.

    Parameters
    ----------
    - `sid (str)`: Subject id.
    - `cond (np.ndarray)`: `(5, 256)` conditioning landmarks (the initial ones, with ids).
    - `pred (np.ndarray)`: `(3, 256)` predicted target landmarks.
    - `converted_root (str)`: Folder holding `<sid>/initial/*.stl`.
    - `out_root (str)`: Where to write the aligned meshes and the transforms.

    Returns
    -------
    - `int`: Number of teeth transformed, or `0` if the subject has no meshes.

    Notes
    -----
    - The solver expects batched tensors, hence the leading dimension added here;
      it is CLIK's own routine, so rotations and translations match the original
      pipeline exactly.
    """
    ini_dir = os.path.join(converted_root, sid, 'initial')
    meshes = {}
    for f in glob.glob(os.path.join(ini_dir, '*.stl')):
        name = os.path.basename(f)[:-4]
        if name.isdigit():
            meshes[int(name)] = load_mesh(f)
    if not meshes:
        return 0

    initial = torch.from_numpy(cond).float().unsqueeze(0)
    target = torch.from_numpy(pred).float().unsqueeze(0)
    trans_mesh, transforms = solve_and_trans_mesh(target, initial, meshes)
    save_mesh(trans_mesh, sid, out_root)
    save_transformation(transforms, sid, out_root)
    return len(trans_mesh)


def main():
    """Transform every subject present in the prediction file."""
    ap = argparse.ArgumentParser(description='Stage D: predicted landmarks -> aligned dentition.')
    ap.add_argument('--predicted', required=True, help='.npz written by stage C')
    ap.add_argument('--tensors', required=True, help='folder of prepared .npz (for the conditioning)')
    ap.add_argument('--out', required=True, help='folder for the aligned meshes')
    ap.add_argument('--converted', default=os.path.join(STEP2, 'data', 'Data_prepost'))
    ap.add_argument('--subjects', nargs='*',
                    help='only these subjects; the exported meshes are large and only a '
                         'handful are ever rendered, so figures need not pay for all of them')
    args = ap.parse_args()

    for k in ('predicted', 'tensors', 'out', 'converted'):
        setattr(args, k, os.path.abspath(getattr(args, k)))

    data = np.load(args.predicted)
    ids, pred = [str(s) for s in data['ids']], data['landmarks']
    wanted = set(args.subjects) if args.subjects else None
    todo = [(i, s) for i, s in enumerate(ids) if wanted is None or s in wanted]
    print(f'{len(todo)} subjects to transform', flush=True)

    ok = 0
    for k, (i, sid) in enumerate(todo, 1):
        tensor = os.path.join(args.tensors, f'{sid}.npz')
        if not os.path.exists(tensor):
            print(f'  [{k}/{len(todo)}] {sid}: tensors missing, skipped')
            continue
        cond = np.load(tensor)['cond'].T
        n = transform_subject(sid, cond, pred[i], args.converted, args.out)
        if n:
            ok += 1
        print(f'  [{k}/{len(todo)}] {sid}: {n} teeth', flush=True)

    print(f'\nTransformed {ok}/{len(todo)} subjects into {args.out}')


if __name__ == '__main__':
    main()
