"""
main
====
Run the whole step-1 evaluation with one command: gather the Teeth3DS subjects
that carry landmark annotations, convert them to CLIK's layout, run the detector,
and score its landmarks against the 3DTeethLand ground truth, producing the
renders along the way. Stages skip themselves when their output already exists.

Functions
---------
- `run(name, fn)`: Execute one stage, timing it and reporting failures without stopping.
- `gather(cfg)`: Collect the annotated subjects out of the `data_part_*` folders.
- `pipeline(cfg)`: Convert, run CLIK, render and evaluate every subject.
- `main()`: Parse the arguments and run the requested stages.

Example
-------
```bash
python main.py                  # gather + full pipeline
python main.py --stages pipeline    # only re-run the pipeline
python main.py --limit 10       # quick pass on ten subjects
```

Notes
-----
- Only the subjects with ground-truth landmarks on both arches are used, since
  they are the only ones on which the detector can actually be scored; the list
  lives in `scripts/gt_patient_ids.txt`.
"""
import os
import sys
import glob
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, 'scripts')

ALL_STAGES = ['gather', 'pipeline']


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
    print(f'\n{"=" * 62}\n== {name}\n{"=" * 62}')
    t0 = time.time()
    try:
        fn()
    except Exception as exc:
        print(f'!! {name} interrotto: {exc}')
        return False
    print(f'-- {name} completato in {time.time() - t0:.0f}s')
    return True


def gather(cfg):
    """Copy the annotated subjects out of the `data_part_*` folders into one place."""
    cmd = [sys.executable, os.path.join(SCRIPTS, 'setup_input.py'),
           '--parts', cfg.dataset, '--out', cfg.src,
           '--ids', os.path.join(SCRIPTS, 'gt_patient_ids.txt'), '--copy']
    if cfg.limit:
        cmd += ['--limit', str(cfg.limit)]
    subprocess.run(cmd, check=True)


def pipeline(cfg):
    """Convert, run CLIK, render with a shared palette and evaluate against the ground truth."""
    subprocess.run([sys.executable, os.path.join(SCRIPTS, 'run_all.py'),
                    '--src', cfg.src, '--gt', cfg.gt,
                    '--converted', cfg.converted, '--output', cfg.output], check=True)


def main():
    """Parse arguments and run the requested stages in order."""
    ap = argparse.ArgumentParser(
        description='Valutazione del rilevamento landmark di CLIK su scansioni Teeth3DS.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--dataset', default=os.path.join(HERE, '..', 'datasets', 'step1_teeth3ds'),
                    help='folder containing the data_part_* directories')
    ap.add_argument('--gt', default=os.path.join(HERE, '..', 'datasets', 'step1_teeth3ds',
                                                 'osfstorage-archive'))
    ap.add_argument('--src', default=os.path.join(HERE, 'data', 'Teeth3DS_input'))
    ap.add_argument('--converted', default=os.path.join(HERE, 'data', 'Data_teeth3ds_gt'))
    ap.add_argument('--output', default=os.path.join(HERE, 'output', 'Output_teeth3ds_gt'))
    ap.add_argument('--stages', default=','.join(ALL_STAGES),
                    help='comma-separated: ' + ', '.join(ALL_STAGES))
    ap.add_argument('--limit', type=int, default=0, help='max subjects (0 = all annotated ones)')
    cfg = ap.parse_args()

    for k in ('dataset', 'gt', 'src', 'converted', 'output'):
        setattr(cfg, k, os.path.abspath(getattr(cfg, k)))

    wanted = [s.strip() for s in cfg.stages.split(',') if s.strip()]
    table = {'gather': gather, 'pipeline': pipeline}
    unknown = [s for s in wanted if s not in table]
    if unknown:
        ap.error(f'stadi sconosciuti: {unknown}; disponibili: {list(table)}')

    print(f'dataset   : {cfg.dataset}\nground tr.: {cfg.gt}\nsoggetti  : {cfg.src}\n'
          f'output    : {cfg.output}\nstadi     : {wanted}')

    t0 = time.time()
    ok = [run(name, lambda f=table[name]: f(cfg)) for name in wanted]
    print(f'\n{sum(ok)}/{len(ok)} stadi completati in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
