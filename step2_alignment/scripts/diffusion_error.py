"""
diffusion_error
===============
Localise where CLIK's alignment error is produced, by measuring the diffusion
stage on its own. Because the ground-truth per-tooth transform is known exactly,
the *ideal* post-treatment landmarks are simply the detected pre-treatment
landmarks carried through that transform. Comparing them with what the diffusion
actually predicts gives the Stage-2 error directly, in millimetres.

The residual of a rigid fit is reported alongside: Stage 3 can only apply a rigid
motion, so any part of the predicted landmark cloud that is not a rigid motion of
the input is information the pipeline necessarily discards.

Functions
---------
- `run_stages_12(sid, converted_root, models, network)`: Detected and predicted landmarks for one subject.
- `evaluate_subject(sid, converted_root, models, network)`: Stage-2 error for one subject.
- `main()`: Run over a sample of subjects and report the aggregate.

Example
-------
```bash
python diffusion_error.py --converted ../Data_prepost --limit 40
```

Notes
-----
- CLIK's own Stage-1 and Stage-2 routines are called unchanged; only the
  intermediate diffusion output is captured, which does not alter behaviour.
- Landmarks live in CLIK's normalised units inside the network, so they are
  multiplied back by `SCALE` (46) to be reported in millimetres.
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import trimesh
from meshes import load_mesh

def _find_clik_root():
    """Locate the CLIK-Diffusion checkout (env `CLIK_ROOT`, else walk up to a sibling)."""
    env = os.environ.get('CLIK_ROOT')
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    while here != os.path.dirname(here):
        cand = os.path.join(here, 'CLIK-Diffusion')
        if os.path.isdir(os.path.join(cand, 'Code')):
            return cand
        here = os.path.dirname(here)
    raise RuntimeError('CLIK-Diffusion checkout not found; set CLIK_ROOT')

ROOT = _find_clik_root()
sys.path.insert(0, os.path.join(ROOT, 'Code'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model.core_util as util
from model.core_util import set_seed
from s1_LandmarkDetection import load_detection, detect_one_patient
from s2_LandmarkDiffusion import load_diffusion, organize_input, query_points, diffusion_one_patient
from evaluate_alignment import kabsch                      

SCALE = 46.0
CKPT = os.path.join(ROOT, 'Code', 'checkpoint')
CKPTS = dict(
    incisor_ckpt=os.path.join(CKPT, '[Crown]incisor-e837.pt'),
    cuspid_ckpt=os.path.join(CKPT, '[Crown]cuspid-e879.pt'),
    premolar_ckpt=os.path.join(CKPT, '[Crown]premolar-e443.pt'),
    molar_ckpt=os.path.join(CKPT, '[Crown]molar-e178.pt'),
)
TEETH = [*range(2, 16), *range(18, 32)]


def run_stages_12(sid, converted_root, models, network):
    """
    Run Stage 1 and Stage 2 for one subject and return both landmark sets.

    Parameters
    ----------
    - `sid (str)`: Subject id.
    - `converted_root (str)`: Converter output root.
    - `models (tuple)`: The four Stage-1 detectors.
    - `network`: The loaded diffusion network.

    Returns
    -------
    - `tuple`: `(initial (3, 256), predicted (3, 256), meshes)` in normalised units.
    """
    set_seed(1)
    lm, meshes = detect_one_patient(sid, os.path.join(converted_root, sid),
                                    *models, num_samples=2048, save_dir=None)
    inp = organize_input(lm, meshes)
    desc = query_points(inp, meshes)
    initial, predicted = diffusion_one_patient(inp, desc, network)
    return initial[0, :3].cpu().numpy(), predicted[0].cpu().numpy(), meshes


def evaluate_subject(sid, converted_root, models, network):
    """
    Measure the diffusion error against the ideal target landmarks.

    Parameters
    ----------
    - `sid (str)`: Subject id.
    - `converted_root (str)`: Converter output root.
    - `models (tuple)`: The four Stage-1 detectors.
    - `network`: The loaded diffusion network.

    Returns
    -------
    - `dict` or `None`: `{'sid', 'diff_err', 'rigid_residual', 'n_teeth'}`, distances in mm.

    Notes
    -----
    - `diff_err` is the distance between the predicted landmark and where that same
      landmark should have ended up under the ground-truth motion. `rigid_residual`
      is what remains after fitting the best rigid motion to the predicted
      landmarks, i.e. the part Stage 3 cannot represent.
    """
    initial, predicted, _ = run_stages_12(sid, converted_root, models, network)

    diff, resid = [], []
    for i, tid in enumerate(TEETH):
        f_ini = os.path.join(converted_root, sid, 'initial', f'{tid}.stl')
        f_fin = os.path.join(converted_root, sid, 'final', f'{tid}.stl')
        if not (os.path.exists(f_ini) and os.path.exists(f_fin)):
            continue
        A = load_mesh(f_ini, process=False).vertices
        B = load_mesh(f_fin, process=False).vertices
        if A.shape != B.shape:
            continue
        R, t, _ = kabsch(A, B)                       # ground-truth motion, mm

        s, e = util.landmark_slices[i], util.landmark_slices[i + 1]
        idx = [i] + list(range(s, e))                # centroid + this tooth's landmarks
        L_ini = initial[:, idx].T * SCALE            # mm
        L_pred = predicted[:, idx].T * SCALE         # mm
        L_ideal = (R @ L_ini.T).T + t                # where they should have gone

        diff.append(np.linalg.norm(L_pred - L_ideal, axis=1))
        Rp, tp, rms = kabsch(L_ini, L_pred)          # best rigid fit of the prediction
        resid.append(rms)

    if not diff:
        return None
    return {'sid': sid, 'diff_err': np.concatenate(diff),
            'rigid_residual': np.array(resid), 'n_teeth': len(resid)}


def main():
    """Run the Stage-2 diagnostic over a sample of subjects and print the aggregate."""
    ap = argparse.ArgumentParser(description="Isolate the diffusion stage error, in landmark space.")
    ap.add_argument('--converted', required=True)
    ap.add_argument('--limit', type=int, default=40)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sids = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(args.converted, '*', 'center.json')))[:args.limit]

    print('Loading networks...')
    models = load_detection(**CKPTS)
    network = load_diffusion(os.path.join(CKPT, 'diffusion-e20000.pth'))

    errs, resids, rows = [], [], []
    for i, sid in enumerate(sids, 1):
        r = evaluate_subject(sid, args.converted, models, network)
        if not r:
            continue
        errs.append(r['diff_err']); resids.append(r['rigid_residual'])
        rows.append({'sid': sid, 'diff_median_mm': float(np.median(r['diff_err'])),
                     'rigid_residual_mm': float(np.median(r['rigid_residual']))})
        print(f"  [{i}/{len(sids)}] {sid}: diffusion error {np.median(r['diff_err']):5.2f} mm | "
              f"non-rigid residual {np.median(r['rigid_residual']):5.2f} mm")

    e = np.concatenate(errs); rs = np.concatenate(resids)
    print(f'\n=== {len(rows)} subjects ===')
    print(f'  diffusion landmark error : median {np.median(e):5.2f} mm  mean {e.mean():5.2f} +/- {e.std():.2f}')
    print(f'  non-rigid residual       : median {np.median(rs):5.2f} mm  (what Stage 3 must discard)')
    if args.out:
        json.dump({'per_subject': rows, 'diffusion_median_mm': float(np.median(e)),
                   'rigid_residual_median_mm': float(np.median(rs))}, open(args.out, 'w'), indent=2)
        print(f'  -> {args.out}')


if __name__ == '__main__':
    main()
