"""
main
====
Run the whole step-2 evaluation with one command: convert the dataset, run CLIK
as-is on every subject, score the prediction against the post-treatment scan,
produce the diagnostics that localise the error, and write the figures and the
spreadsheet. Every stage is skipped automatically when its output already exists,
so the script can be re-run to resume or to refresh a single part.

Functions
---------
- `run(name, fn)`: Execute one stage, timing it and reporting failures without stopping.
- `convert(cfg)`: Dataset -> CLIK per-tooth layout.
- `infer(cfg)`: CLIK crown-only inference on every converted subject.
- `evaluate(cfg)`: Alignment metrics against the ground truth.
- `diagnostics(cfg)`: Stage-by-stage error localisation.
- `seeds(cfg)`: Repeat the inference with several seeds to separate noise from error.
- `figures(cfg)`: Qualitative comparisons and the summary chart.
- `spreadsheet(cfg)`: Collect everything into one workbook.
- `main()`: Parse the arguments and run the requested stages.

Example
-------
```bash
python main.py                           # everything except the seed study
python main.py --stages evaluate,excel   # only re-score and refresh the workbook
python main.py --with-seeds --limit 20   # include the seed study, on 20 subjects
```

Notes
-----
- CLIK resolves its checkpoints relative to the working directory, so inference is
  launched from inside the CLIK-Diffusion folder.
- Long stages print progress; the whole run on 250 subjects takes a few hours,
  almost all of it in the diffusion sampling.
"""
import os
import sys
import glob
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, 'scripts')
REPORTING = os.path.join(SCRIPTS, 'reporting')
sys.path.insert(0, SCRIPTS)

ALL_STAGES = ['convert', 'infer', 'evaluate', 'diagnostics', 'figures', 'excel']


def _clik_root():
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
    except Exception as exc:                        # keep going: later stages may still work
        print(f'!! {name} interrotto: {exc}')
        return False
    print(f'-- {name} completato in {time.time() - t0:.0f}s')
    return True


def convert(cfg):
    """Convert the raw dataset into CLIK's per-tooth layout (skips existing subjects)."""
    from prepost_to_clik import convert_subject
    dirs = sorted(d for d in glob.glob(os.path.join(cfg.dataset, 'Orthodontic_dental_dataset', '*'))
                  if os.path.isdir(d))
    if cfg.limit:
        dirs = dirs[:cfg.limit]
    todo = [d for d in dirs
            if not os.path.exists(os.path.join(cfg.converted, os.path.basename(d), 'center.json'))]
    print(f'{len(dirs)} soggetti, {len(todo)} da convertire')
    for d in todo:
        convert_subject(d, cfg.converted)


def infer(cfg):
    """Run CLIK crown-only inference on every converted subject that lacks a result."""
    root = _clik_root()
    sids = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(cfg.converted, '*', 'center.json')))
    todo = [s for s in sids
            if not os.path.exists(os.path.join(cfg.output, s, 'results', 'transformation.json'))]
    print(f'{len(sids)} soggetti, {len(todo)} da elaborare (~40 s ciascuno)')
    for i, sid in enumerate(todo, 1):
        print(f'  [{i}/{len(todo)}] {sid}', flush=True)
        subprocess.run([sys.executable, '-u', os.path.join(root, 'Code', 'infer_crown.py'),
                        '-i', os.path.join(cfg.converted, sid), '-o', cfg.output],
                       cwd=root, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def evaluate(cfg):
    """Score every prediction: the headline metrics and the detailed breakdown."""
    py = sys.executable   # '-u': stream the child's output instead of buffering it
    subprocess.run([py, '-u', os.path.join(SCRIPTS, 'evaluate_alignment.py'),
                    '--converted', cfg.converted, '--output', cfg.output], check=True)
    subprocess.run([py, '-u', os.path.join(SCRIPTS, 'evaluate_detail.py'),
                    '--converted', cfg.converted, '--output', cfg.output, '--report', cfg.data,
                    '--collision-subjects', str(cfg.collision_subjects)], check=True)


def diagnostics(cfg):
    """Localise the error: detector repeatability and accuracy, then the diffusion itself."""
    py = sys.executable   # '-u': stream the child's output instead of buffering it
    subprocess.run([py, '-u', os.path.join(SCRIPTS, 'landmark_repeatability.py'),
                    '--converted', cfg.converted, '--limit', str(cfg.diag_subjects),
                    '--out', os.path.join(cfg.data, 'landmark_repeatability.json')], check=True)
    for stage in ('ori', 'final'):
        suffix = '' if stage == 'ori' else '_final'
        subprocess.run([py, '-u', os.path.join(SCRIPTS, 'landmark_accuracy.py'),
                        '--converted', cfg.converted, '--dataset', cfg.dataset, '--stage', stage,
                        '--out', os.path.join(cfg.data, f'landmark_accuracy{suffix}.json')], check=True)
    subprocess.run([py, '-u', os.path.join(SCRIPTS, 'diffusion_error.py'),
                    '--converted', cfg.converted, '--limit', str(cfg.diag_subjects),
                    '--out', os.path.join(cfg.data, 'diffusion_error.json')], check=True)


def seeds(cfg):
    """Repeat the inference with extra seeds and measure how much the result moves."""
    root = _clik_root()
    sids = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(cfg.converted, '*', 'center.json')))[:cfg.diag_subjects]
    runs = [cfg.output]
    for seed in cfg.seeds:
        out = os.path.join(os.path.dirname(cfg.output), f'seed_{seed}')
        runs.append(out)
        for i, sid in enumerate(sids, 1):
            if os.path.exists(os.path.join(out, sid, 'results', 'transformation.json')):
                continue
            print(f'  seed {seed} [{i}/{len(sids)}] {sid}', flush=True)
            subprocess.run([sys.executable, '-u', os.path.join(root, 'Code', 'infer_crown.py'),
                            '--seed', str(seed), '-i', os.path.join(cfg.converted, sid), '-o', out],
                           cwd=root, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([sys.executable, '-u', os.path.join(SCRIPTS, 'seed_variability.py'),
                    '--converted', cfg.converted, '--runs', *runs,
                    '--out', os.path.join(cfg.data, 'seed_variability.json')], check=True)


def figures(cfg):
    """Render the qualitative comparisons and the summary chart."""
    subprocess.run([sys.executable, '-u', os.path.join(REPORTING, 'render_cases.py'),
                    '--converted', cfg.converted, '--output', cfg.output,
                    '--out-dir', cfg.figures], check=True)
    subprocess.run([sys.executable, '-u', os.path.join(REPORTING, 'make_chart.py'),
                    '--output', cfg.output, '--report', cfg.figures], check=True)


def spreadsheet(cfg):
    """Collect every result into the shareable workbook."""
    subprocess.run([sys.executable, '-u', os.path.join(REPORTING, 'make_excel.py'),
                    '--output', cfg.output, '--report', cfg.report, '--data', cfg.data], check=True)


def main():
    """Parse arguments and run the requested stages in order."""
    ap = argparse.ArgumentParser(
        description='Valutazione as-is di CLIK sul dataset PrePostOrthodontic.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--dataset', default=os.path.join(HERE, '..', 'datasets', 'step2_prepost'))
    ap.add_argument('--converted', default=os.path.join(HERE, 'data', 'Data_prepost'))
    ap.add_argument('--output', default=os.path.join(HERE, 'output', 'Output_prepost'))
    ap.add_argument('--report', default=os.path.join(HERE, 'report'))
    ap.add_argument('--stages', default=','.join(ALL_STAGES),
                    help='comma-separated: ' + ', '.join(ALL_STAGES))
    ap.add_argument('--with-seeds', action='store_true', help='include the seed study (slow)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[2, 3, 4, 5])
    ap.add_argument('--limit', type=int, default=0, help='max subjects to convert (0 = all)')
    ap.add_argument('--diag-subjects', type=int, default=40,
                    help='subjects used by the sampled diagnostics')
    ap.add_argument('--collision-subjects', type=int, default=40,
                    help='subjects for the collision measure; 0 reuses the previous value '
                         '(this is what makes the evaluate stage slow)')
    cfg = ap.parse_args()

    for k in ('dataset', 'converted', 'output', 'report'):
        setattr(cfg, k, os.path.abspath(getattr(cfg, k)))
    # the report folder holds only what gets shared; figures and the raw numbers
    # that feed them live one level down
    cfg.figures = os.path.join(cfg.report, 'figures')
    cfg.data = os.path.join(cfg.report, 'data')
    for d in (cfg.report, cfg.figures, cfg.data):
        os.makedirs(d, exist_ok=True)

    wanted = [s.strip() for s in cfg.stages.split(',') if s.strip()]
    table = {'convert': convert, 'infer': infer, 'evaluate': evaluate,
             'diagnostics': diagnostics, 'figures': figures, 'excel': spreadsheet}
    unknown = [s for s in wanted if s not in table]
    if unknown:
        ap.error(f'stadi sconosciuti: {unknown}; disponibili: {list(table)}')

    print(f'dataset   : {cfg.dataset}\nconvertiti: {cfg.converted}\noutput    : {cfg.output}\n'
          f'report    : {cfg.report}\nstadi     : {wanted}' + ('  + seeds' if cfg.with_seeds else ''))

    t0 = time.time()
    ok = [run(name, lambda f=table[name]: f(cfg)) for name in wanted]
    if cfg.with_seeds:
        ok.append(run('seeds', lambda: seeds(cfg)))
    print(f'\n{sum(ok)}/{len(ok)} stadi completati in {(time.time() - t0) / 60:.1f} min')
    print(f'risultati condivisibili in {cfg.report}')


if __name__ == '__main__':
    main()
