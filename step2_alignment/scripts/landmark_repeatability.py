"""
landmark_repeatability
======================
Measure the quality of CLIK's Stage-1 landmark detector on this dataset without
needing any landmark ground truth, by exploiting a property of the data: the
`final` teeth are the *same* meshes as the `ori` teeth, only rigidly moved. A
sound detector must therefore put its landmarks on the same anatomical points, so
the landmarks found on `final` should equal the landmarks found on `ori` mapped
through the known ground-truth transform. Whatever is left over is detector error.

A noise floor is measured alongside: the detector is run twice on the *same* mesh
with different random seeds, which isolates the jitter caused by the random start
of the farthest-point sampling from genuine pose sensitivity.

Functions
---------
- `detect_stage(models, converted_root, sid, stage, seed)`: Stage-1 landmarks for one stage.
- `evaluate_subject(models, converted_root, sid)`: Repeatability and noise floor for one subject.
- `main()`: Run over a set of subjects and report the aggregate.

Example
-------
```bash
python landmark_repeatability.py --converted ../Data_prepost --limit 60
```

Notes
-----
- Only Stage 1 runs here (no diffusion), so this is cheap compared to a full
  inference. CLIK's own detection routine is called unchanged; the `final` meshes
  are exposed to it through a temporary `initial/` folder because that is the
  layout it expects.
"""
import os
import sys
import glob
import json
import shutil
import tempfile
import argparse
import numpy as np
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

from model.core_util import set_seed
from s1_LandmarkDetection import load_detection, detect_one_patient
from evaluate_alignment import kabsch

CKPT = os.path.join(ROOT, 'Code', 'checkpoint')
CKPTS = dict(
    incisor_ckpt=os.path.join(CKPT, '[Crown]incisor-e837.pt'),
    cuspid_ckpt=os.path.join(CKPT, '[Crown]cuspid-e879.pt'),
    premolar_ckpt=os.path.join(CKPT, '[Crown]premolar-e443.pt'),
    molar_ckpt=os.path.join(CKPT, '[Crown]molar-e178.pt'),
)


def detect_stage(models, converted_root, sid, stage, seed=1):
    """
    Run CLIK's Stage-1 detector on one stage of one subject.

    Parameters
    ----------
    - `models (tuple)`: The four detection networks from `load_detection`.
    - `converted_root (str)`: Converter output root.
    - `sid (str)`: Subject id.
    - `stage (str)`: `"initial"` or `"final"`.
    - `seed (int, optional)`: Random seed (affects the farthest-point sampling). Default `1`.

    Returns
    -------
    - `dict`: `{tooth_id: {landmark_id: np.ndarray(3)}}` in the converted frame.

    Notes
    -----
    - CLIK's `detect_one_patient` expects the meshes under an `initial/` folder, so
      a temporary directory with that name is pointed at the requested stage.
    """
    set_seed(seed)
    tmp = tempfile.mkdtemp()
    try:
        os.symlink(os.path.abspath(os.path.join(converted_root, sid, stage)),
                   os.path.join(tmp, 'initial'))
        lm, _ = detect_one_patient(sid, tmp, *models, num_samples=2048, save_dir=None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {t: {k: np.asarray(v, float) for k, v in d.items()} for t, d in lm.items()}


def evaluate_subject(models, converted_root, sid):
    """
    Compute repeatability (pose sensitivity) and the sampling noise floor.

    Parameters
    ----------
    - `models (tuple)`: The four detection networks.
    - `converted_root (str)`: Converter output root.
    - `sid (str)`: Subject id.

    Returns
    -------
    - `dict` or `None`: `{'sid', 'pose_err', 'noise_err', 'n'}` with per-landmark
      distances in mm, or `None` if the subject has no comparable teeth.

    Notes
    -----
    - `pose_err` compares landmarks detected on `final` against landmarks detected
      on `ori` carried through the ground-truth transform; `noise_err` compares two
      detections of the same `ori` meshes under different seeds.
    """
    lm_ini = detect_stage(models, converted_root, sid, 'initial', seed=1)
    lm_fin = detect_stage(models, converted_root, sid, 'final', seed=1)
    lm_ini2 = detect_stage(models, converted_root, sid, 'initial', seed=7)

    pose, noise = [], []
    for tid, d_ini in lm_ini.items():
        f_ini = os.path.join(converted_root, sid, 'initial', f'{tid}.stl')
        f_fin = os.path.join(converted_root, sid, 'final', f'{tid}.stl')
        if tid not in lm_fin or not os.path.exists(f_fin):
            continue
        A = load_mesh(f_ini, process=False).vertices
        B = load_mesh(f_fin, process=False).vertices
        if A.shape != B.shape:
            continue
        R, t, _ = kabsch(A, B)

        for k, p_ini in d_ini.items():
            if k in lm_fin[tid]:
                pose.append(np.linalg.norm((R @ p_ini + t) - lm_fin[tid][k]))
            if tid in lm_ini2 and k in lm_ini2[tid]:
                noise.append(np.linalg.norm(p_ini - lm_ini2[tid][k]))

    if not pose:
        return None
    return {'sid': sid, 'pose_err': np.array(pose), 'noise_err': np.array(noise), 'n': len(pose)}


def main():
    """Run the repeatability test over the converted subjects and print the aggregate."""
    ap = argparse.ArgumentParser(description="Stage-1 landmark repeatability under rigid motion.")
    ap.add_argument('--converted', required=True)
    ap.add_argument('--limit', type=int, default=0, help='max subjects (0 = all)')
    ap.add_argument('--out', default=None, help='optional JSON summary path')
    args = ap.parse_args()

    sids = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(args.converted, '*', 'center.json')))
    if args.limit:
        sids = sids[:args.limit]

    print('Loading detectors...')
    models = load_detection(**CKPTS)

    pose_all, noise_all, per_subject = [], [], []
    for i, sid in enumerate(sids, 1):
        r = evaluate_subject(models, args.converted, sid)
        if not r:
            continue
        pose_all.append(r['pose_err']); noise_all.append(r['noise_err'])
        per_subject.append({'sid': sid, 'n': r['n'],
                            'pose_median': float(np.median(r['pose_err'])),
                            'noise_median': float(np.median(r['noise_err'])) if len(r['noise_err']) else None})
        print(f"  [{i}/{len(sids)}] {sid}: pose {np.median(r['pose_err']):5.2f} mm | "
              f"noise {np.median(r['noise_err']):5.2f} mm ({r['n']} landmarks)")

    pose = np.concatenate(pose_all); noise = np.concatenate(noise_all)
    print(f'\n=== {len(per_subject)} subjects, {len(pose)} landmark pairs ===')
    print(f'  pose repeatability : median {np.median(pose):5.2f} mm   mean {pose.mean():5.2f} +/- {pose.std():.2f}')
    print(f'  sampling noise floor: median {np.median(noise):5.2f} mm   mean {noise.mean():5.2f} +/- {noise.std():.2f}')
    print(f'  landmarks within 1 mm: pose {100*(pose<1).mean():.0f}%   noise {100*(noise<1).mean():.0f}%')
    if args.out:
        json.dump({'per_subject': per_subject,
                   'pose_median_mm': float(np.median(pose)),
                   'noise_median_mm': float(np.median(noise))}, open(args.out, 'w'), indent=2)
        print(f'  -> {args.out}')


if __name__ == '__main__':
    main()
