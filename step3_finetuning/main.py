"""
main
====
Drive the fine-tuning pipeline end to end. The work is split into four stages,
each a single self-contained script, because they have very different needs: two
of them must stay next to the meshes, the other two are pure neural-network work
that can run on any device or on a rented GPU.

Stage A  prepare    meshes -> tensors            local, needs the 25 GB dataset
Stage B  train      tensors -> checkpoint        any device (CUDA / MPS / CPU)
Stage C  predict    tensors + ckpt -> landmarks  any device, batched
Stage D  transform  landmarks -> aligned teeth   local, needs the meshes, float64

Stages B and C exchange a few hundred megabytes in and well under one megabyte
out, which is what makes offloading them cheap. Nothing here requires a rented
machine: the default device is whatever the local one offers.

Functions
---------
- `run(name, fn)`: Execute one stage, timing it and reporting failures without stopping.
- `stage_a(cfg)` … `stage_d(cfg)`: The four stages.
- `evaluate(cfg)`: Score the aligned dentition against the post-treatment scan.
- `main()`: Parse the arguments and run the requested stages.

Example
-------
```bash
python main.py --stages c,d,evaluate --ckpt output/base_loss/best.pt
python main.py --stages a --split test          # prepare the test tensors
python main.py --stages b --epochs 100          # fine-tune
```

Notes
-----
- Every stage skips work whose output already exists, so a run can be resumed.
- Force a device with the `CLIK_DEVICE` environment variable (`cpu`, `mps`, `cuda`).
"""
import os
import sys
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, 'scripts')
BASE = os.path.dirname(HERE)
STEP2 = os.path.join(BASE, 'step2_alignment')

ALL_STAGES = ['a', 'b', 'c', 'd', 'evaluate']
TERMS = ('arch', 'contact', 'individual')      # the clinical constraints, see losses.py


def run(name, fn):
    """
    Execute one stage, timing it and reporting a failure without aborting the run.

    Parameters
    ----------
    - `name (str)`: Stage name, printed as a banner.
    - `fn (callable)`: The stage itself.

    Returns
    -------
    - `bool`: `True` if the stage completed without raising.
    """
    print(f'\n{"=" * 62}\n== {name}\n{"=" * 62}', flush=True)
    t0 = time.time()
    try:
        fn()
    except Exception as exc:
        print(f'!! {name} failed: {exc}')
        return False
    print(f'-- {name} finished in {time.time() - t0:.0f}s')
    return True


def _py(script, *args):
    """Run one of the stage scripts as a subprocess, unbuffered."""
    subprocess.run([sys.executable, '-u', os.path.join(SCRIPTS, script), *args], check=True)


def stage_a(cfg):
    """Meshes to tensors: detection, descriptors and targets for the chosen split."""
    ids = os.path.join(BASE, 'datasets', 'step2_prepost', f'{cfg.split}_ids.txt')
    _py('stage_a_prepare.py', '--converted', cfg.converted, '--out', cfg.tensors,
        '--ids', ids, '--workers', str(cfg.workers))


def stage_b(cfg):
    """Fine-tune the diffusion model, with whichever clinical constraints are enabled."""
    extra = []
    for term in TERMS:
        weight = getattr(cfg, term)
        if weight:
            extra += [f'--{term}', str(weight)]
    if cfg.clinical:
        extra += ['--clinical', cfg.clinical]
    _py('stage_b_train.py', '--tensors', cfg.train_tensors, '--out', cfg.run_dir,
        '--epochs', str(cfg.epochs), '--batch-size', str(cfg.batch_size), *extra)


def stage_c(cfg):
    """Sample the target landmarks for the test subjects, in batches."""
    _py('stage_c_predict.py', '--tensors', cfg.tensors, '--ckpt', cfg.ckpt,
        '--out', cfg.predicted, '--batch-size', str(cfg.predict_batch),
        '--steps', str(cfg.steps))


def stage_d(cfg):
    """Apply the predicted landmarks to the meshes, in CLIK's own output format."""
    _py('stage_d_transform.py', '--predicted', cfg.predicted, '--tensors', cfg.tensors,
        '--out', cfg.inference, '--converted', cfg.converted)


def evaluate(cfg):
    """Score the aligned dentition with the step-2 evaluation, unchanged."""
    subprocess.run([sys.executable, '-u', os.path.join(STEP2, 'scripts', 'evaluate_alignment.py'),
                    '--converted', cfg.converted, '--output', cfg.inference], check=True)


def main():
    """Parse arguments and run the requested stages in order."""
    ap = argparse.ArgumentParser(
        description='Fine-tune CLIK on the PrePostOrthodontic dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--stages', default=','.join(ALL_STAGES),
                    help='comma-separated: ' + ', '.join(ALL_STAGES))
    ap.add_argument('--run', default='base_loss', help='name of this experiment')
    ap.add_argument('--split', default='test', choices=['train', 'test'],
                    help='which split stage A prepares')
    ap.add_argument('--converted', default=os.path.join(STEP2, 'data', 'Data_prepost'))
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=48)
    ap.add_argument('--predict-batch', type=int, default=16)
    ap.add_argument('--steps', type=int, default=0,
                    help='denoising steps in stage C; 0 runs the full schedule. The '
                         'network ignores its noisy input, so 1 gives the same answer '
                         'two thousand times faster')
    ap.add_argument('--workers', type=int, default=4, help='parallel processes for stage A')
    ap.add_argument('--ckpt', default=None, help='checkpoint for stages C/D (default: this run\'s)')
    for term in TERMS:
        ap.add_argument(f'--{term}', type=float, default=0.0,
                        help=f'weight of the {term} clinical constraint in stage B')
    ap.add_argument('--clinical', choices=['paper'],
                    help="use the paper's three weights together (measured as harmful "
                         'when fine-tuning; see the step-3 report)')
    cfg = ap.parse_args()

    cfg.converted = os.path.abspath(cfg.converted)
    cfg.run_dir = os.path.join(HERE, 'output', cfg.run)
    cfg.train_tensors = os.path.join(HERE, 'data', 'train_tensors')
    cfg.tensors = os.path.join(HERE, 'data', f'{cfg.split}_tensors')
    cfg.predicted = os.path.join(cfg.run_dir, 'predicted_landmarks.npz')
    cfg.inference = os.path.join(cfg.run_dir, 'inference')
    cfg.ckpt = os.path.abspath(cfg.ckpt) if cfg.ckpt else \
        os.path.join(cfg.run_dir, 'best.pt')
    os.makedirs(cfg.run_dir, exist_ok=True)

    wanted = [s.strip().lower() for s in cfg.stages.split(',') if s.strip()]
    table = {'a': stage_a, 'b': stage_b, 'c': stage_c, 'd': stage_d, 'evaluate': evaluate}
    unknown = [s for s in wanted if s not in table]
    if unknown:
        ap.error(f'unknown stages: {unknown}; available: {list(table)}')

    print(f'experiment : {cfg.run}\nconverted  : {cfg.converted}\n'
          f'tensors    : {cfg.tensors}\ncheckpoint : {cfg.ckpt}\nstages     : {wanted}')

    t0 = time.time()
    ok = [run(f'stage {name}', lambda f=table[name]: f(cfg)) for name in wanted]
    print(f'\n{sum(ok)}/{len(ok)} stages finished in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
