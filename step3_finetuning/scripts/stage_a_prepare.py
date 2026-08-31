"""
stage_a_prepare
===============
Stage A of the fine-tuning pipeline: turn converted subjects into the tensors the
diffusion model trains on, so that
training never has to touch the meshes again. For each subject this runs CLIK's
Stage-1 detector once and stores the conditioning landmarks, the dentition
descriptor, the target landmarks and a validity mask.

The target is not obtained by detecting landmarks a second time on the
post-treatment crowns. Because `final` holds the same meshes as `ori`, only
rigidly moved, the exact per-tooth transform is recoverable and can be applied to
the landmarks already detected on `ori`: the target is then exact by construction
and corresponds one-to-one with the input, which a second detection could not
guarantee.

The detector is stochastic: it draws its own random point samples inside every
encoder stage. Reproducing what `infer_crown` sees therefore means reproducing the
whole sequence of draws, which has two consequences that look like inefficiencies
and are not. The detectors are rebuilt for every subject, because `infer_crown`
seeds and only then constructs them, and constructing a `PointMLP` initialises its
weights from the same generator. Building them once up front would leave every
subject on a different random stream, shifting landmarks by millimetres. And the
28 teeth are detected one at a time rather than as a batch, because the number of
draws depends on how many teeth travel together; batching moves 161 of 228
landmarks to a different vertex. Speed comes from `--workers` instead, which gives
each process its own faithful copy of the sequence.

Functions
---------
- `tooth_landmark_indices(i)`: Positions of one tooth's landmarks inside the 256-vector.
- `build_subject(sid, converted_root)`: Tensors for one subject.
- `main()`: Process every converted subject and write one `.npz` each.

Example
-------
```bash
python stage_a_prepare.py --converted ../../step2_alignment/data/Data_prepost \\
    --out ../data/train_tensors --ids ../../datasets/step2_prepost/train_ids.txt
```

Notes
-----
- Coordinates are stored in CLIK's normalised units (millimetres divided by
  `SCALE`), which is what the network consumes.
- The mask is False for teeth missing from the scan and for teeth extracted during
  treatment, which have an input but no target; the training loss must ignore them.
- Subjects are independent, so `--workers` scales nearly linearly: four processes
  take a subject from about 10 s to about 6 s.
"""
import os
import sys
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


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
STEP2 = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'step2_alignment', 'scripts')
sys.path.insert(0, os.path.join(ROOT, 'Code'))
sys.path.insert(0, STEP2)


import model.core_util as util

from model.core_util import set_seed
from s1_LandmarkDetection import load_detection, detect_one_patient
from s2_LandmarkDiffusion import organize_input, query_points
from evaluate_alignment import kabsch
from meshes import load_mesh

SCALE = 46.0
TEETH = [*range(2, 16), *range(18, 32)]
CKPT = os.path.join(ROOT, 'Code', 'checkpoint')
CKPTS = dict(
    incisor_ckpt=os.path.join(CKPT, '[Crown]incisor-e837.pt'),
    cuspid_ckpt=os.path.join(CKPT, '[Crown]cuspid-e879.pt'),
    premolar_ckpt=os.path.join(CKPT, '[Crown]premolar-e443.pt'),
    molar_ckpt=os.path.join(CKPT, '[Crown]molar-e178.pt'),
)


def tooth_landmark_indices(i):
    """
    Positions occupied by one tooth inside the 256-landmark vector.

    Parameters
    ----------
    - `i (int)`: Index of the tooth in `TEETH` (0..27).

    Returns
    -------
    - `list`: `[centroid_index] + [its other landmark indices]`.

    Notes
    -----
    - The vector holds the 28 centroids first, then the remaining landmarks
      grouped per tooth according to `core_util.landmark_slices`.
    """
    return [i] + list(range(util.landmark_slices[i], util.landmark_slices[i + 1]))


def build_subject(sid, converted_root, seed=1):
    """
    Build the training tensors for one subject.

    Parameters
    ----------
    - `sid (str)`: Subject id.
    - `converted_root (str)`: Folder holding `<sid>/initial` and `<sid>/final`.
    - `seed (int, optional)`: Seed for the detector's sampling. Default `1`.

    Returns
    -------
    - `dict` or `None`: `{'cond', 'desc', 'target', 'mask', 'teeth'}`, or `None` when
      the subject yields no usable tooth.

    Notes
    -----
    - `cond` is `(256, 5)` (coordinates plus landmark and tooth ids), `desc` is
      `(256, 384)`, `target` is `(256, 3)` and `mask` is `(256,)` booleans.
    - Seeding then loading, in that order, is what `infer_crown` does, and the
      order matters because building the detectors consumes random numbers. The
      two seconds it costs per subject buy tensors identical to the ones the
      original pipeline works from.
    """
    set_seed(seed)
    models = load_detection(**CKPTS)
    subject_dir = os.path.join(converted_root, sid)
    landmarks, meshes = detect_one_patient(sid, subject_dir, *models,
                                           num_samples=2048, save_dir=None)
    if not landmarks:
        return None

    cond = organize_input(landmarks, meshes)
    desc = query_points(cond, meshes)

    target = np.zeros((256, 3), dtype=np.float32)
    mask = np.zeros(256, dtype=bool)
    used = []
    for i, tid in enumerate(TEETH):
        f_ini = os.path.join(subject_dir, 'initial', f'{tid}.stl')
        f_fin = os.path.join(subject_dir, 'final', f'{tid}.stl')
        if not (os.path.exists(f_ini) and os.path.exists(f_fin)):
            continue                                   # missing, or extracted during treatment
        A = load_mesh(f_ini, process=False).vertices
        B = load_mesh(f_fin, process=False).vertices
        if A.shape != B.shape:
            continue                                   # re-tessellated: no correspondence
        R, t, _ = kabsch(A, B)

        idx = tooth_landmark_indices(i)
        pts_mm = cond[idx, :3] * SCALE
        target[idx] = ((R @ pts_mm.T).T + t) / SCALE
        mask[idx] = True
        used.append(tid)

    if not used:
        return None
    return {'cond': cond.astype(np.float32), 'desc': desc.astype(np.float32),
            'target': target, 'mask': mask, 'teeth': np.array(used)}


def _worker(todo, converted, out, tag):
    """Prepare a slice of the subject list; used as the body of one worker process."""
    for i, sid in enumerate(todo, 1):
        try:
            data = build_subject(sid, converted)
        except Exception as exc:
            print(f'  [{tag}] {sid}: error ({exc})', flush=True)
            continue
        if data is None:
            print(f'  [{tag}] {sid}: skipped (no usable tooth)', flush=True)
            continue
        np.savez_compressed(os.path.join(out, f'{sid}.npz'), **data)
        print(f'  [{tag} {i}/{len(todo)}] {sid}: {len(data["teeth"])} teeth, '
              f'{int(data["mask"].sum())}/256 landmarks with a target', flush=True)


def main():
    """Process every requested subject and write one `.npz` per subject."""
    ap = argparse.ArgumentParser(description="Build the diffusion training tensors.")
    ap.add_argument('--converted', required=True, help='converter output (initial/ and final/)')
    ap.add_argument('--out', required=True, help='folder for the .npz files')
    ap.add_argument('--ids', help='optional text file restricting the subjects')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1,
                    help='parallel processes; subjects are independent, so this scales '
                         'almost linearly without touching the detector numerics')
    args = ap.parse_args()

    available = sorted(os.path.basename(os.path.dirname(p))
                       for p in glob.glob(os.path.join(args.converted, '*', 'center.json')))
    if args.ids:
        wanted = {l.strip() for l in open(args.ids) if l.strip()}
        available = [s for s in available if s in wanted]
    todo = [s for s in available if not os.path.exists(os.path.join(args.out, f'{s}.npz'))]
    if args.limit:
        todo = todo[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    print(f'{len(available)} subjects available, {len(todo)} to prepare, '
          f'{args.workers} processes', flush=True)
    if not todo:
        return

    if args.workers <= 1:
        _worker(todo, args.converted, args.out, 'w0')
    else:
        import multiprocessing as mp
        chunks = [todo[i::args.workers] for i in range(args.workers)]
        procs = [mp.Process(target=_worker, args=(c, args.converted, args.out, f'w{i}'))
                 for i, c in enumerate(chunks) if c]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    done = len(glob.glob(os.path.join(args.out, '*.npz')))
    print(f'\nPrepared {done} subjects in {args.out}')


if __name__ == '__main__':
    main()
