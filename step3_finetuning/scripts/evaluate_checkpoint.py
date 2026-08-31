"""
evaluate_checkpoint
===================
Score a fine-tuned diffusion checkpoint on the test split and put the numbers
next to the as-is baseline, which is the comparison the whole study is about.

Inference is run through CLIK's own `infer_crown.py`, pointed at the given
checkpoint via its `--diffusion_ckpt` option, so nothing in the method changes
apart from the weights being tested. Scoring reuses the step-2 evaluation, which
means the fine-tuned and the as-is numbers are produced by identical code.

For routine work the stage C/D path is the one to use: it is two and a half times
faster and produces the same transforms to four decimal places. This script earns
its keep as the *independent* check: it shares no code with stages C and D, so
when the two agree, they agree for a reason. That is how the split pipeline was
validated in the first place, and it is worth re-running after any change to the
sampling or the rigid solve.

Functions
---------
- `test_subjects(converted_root, ids_file)`: The test-split subjects available locally.
- `run_inference(ckpt, sids, converted_root, out_root)`: Inference with the given weights.
- `compare(as_is_csv, tuned_csv, sids)`: Side-by-side medians on the same subjects.
- `main()`: Run inference, score, and print the comparison.

Example
-------
```bash
python evaluate_checkpoint.py --ckpt ../output/base_loss/best.pt \\
    --out ../output/base_loss/inference --limit 40
```

Notes
-----
- The comparison is restricted to the subjects present in both runs, so a partial
  evaluation stays honest: with `--limit` the as-is medians are recomputed on the
  same subset rather than taken from the full 250.
- Inference costs roughly 40 s per subject, almost all of it diffusion sampling.
"""
import os
import sys
import csv
import glob
import time
import argparse
import subprocess
import numpy as np

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
METRICS = ('rot_err', 'trans_err', 'pcd_err')
LABELS = {'rot_err': 'rotation (degrees)', 'trans_err': 'translation',
          'pcd_err': 'point cloud (mm)'}


def test_subjects(converted_root, ids_file, limit=0):
    """
    The test-split subjects that have been converted locally.

    Parameters
    ----------
    - `converted_root (str)`: Folder with the converted subjects.
    - `ids_file (str)`: Text file listing the test-split ids.
    - `limit (int, optional)`: Keep only the first N. Default `0` (all).

    Returns
    -------
    - `list`: Subject ids.
    """
    wanted = {l.strip() for l in open(ids_file) if l.strip()}
    have = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(converted_root, '*', 'center.json')))
    sids = [s for s in have if s in wanted]
    return sids[:limit] if limit else sids


def run_inference(ckpt, sids, converted_root, out_root):
    """
    Run CLIK inference on the given subjects with the supplied diffusion weights.

    Parameters
    ----------
    - `ckpt (str)`: Path to the diffusion checkpoint to test.
    - `sids (list)`: Subject ids.
    - `converted_root (str)`: Folder with the converted subjects.
    - `out_root (str)`: Where to write the predictions.

    Returns
    -------
    - `int`: How many subjects were processed in this call (already-done ones are skipped).

    Raises
    ------
    - `subprocess.CalledProcessError`: If an inference process exits non-zero.
    """
    todo = [s for s in sids
            if not os.path.exists(os.path.join(out_root, s, 'results', 'transformation.json'))]
    print(f'{len(sids)} subjects, {len(todo)} to process (~40 s ciascuno)', flush=True)
    for i, sid in enumerate(todo, 1):
        print(f'  [{i}/{len(todo)}] {sid}', flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, 'Code', 'infer_crown.py'),
                        '-i', os.path.join(converted_root, sid), '-o', out_root,
                        '--diffusion_ckpt', ckpt],
                       cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return len(todo)


def compare(as_is_csv, tuned_csv, sids):
    """
    Median metrics of both runs, restricted to the same subjects.

    Parameters
    ----------
    - `as_is_csv (str)`: `alignment_metrics.csv` of the pretrained run.
    - `tuned_csv (str)`: `alignment_metrics.csv` of the fine-tuned run.
    - `sids (list)`: Subjects to include.

    Returns
    -------
    - `dict`: Per metric, the as-is median, the fine-tuned median, the baseline and
      the relative change.
    """
    def load(path):
        return {r['subject']: r for r in csv.DictReader(open(path))}

    a, b = load(as_is_csv), load(tuned_csv)
    common = [s for s in sids if s in a and s in b]
    out = {'n': len(common)}
    for k in METRICS:
        va = np.median([float(a[s][k]) for s in common])
        vb = np.median([float(b[s][k]) for s in common])
        base = np.median([float(a[s][k.replace('_err', '_baseline')]) for s in common])
        out[k] = {'as_is': va, 'tuned': vb, 'baseline': base,
                  'change_pct': 100.0 * (vb - va) / va if va else float('nan')}
    return out


def main():
    """Run inference with the given checkpoint, score it and print the comparison."""
    ap = argparse.ArgumentParser(description="Score a fine-tuned checkpoint against the as-is run.")
    ap.add_argument('--ckpt', required=True, help='diffusion checkpoint to test')
    ap.add_argument('--out', required=True, help='folder for the predictions of this checkpoint')
    ap.add_argument('--converted', default=os.path.join(STEP2, 'data', 'Data_prepost'))
    ap.add_argument('--as-is', default=os.path.join(STEP2, 'output', 'Output_prepost'),
                    help='folder holding the pretrained run to compare against')
    ap.add_argument('--ids', default=os.path.join(BASE, 'datasets', 'step2_prepost', 'test_ids.txt'))
    ap.add_argument('--limit', type=int, default=0, help='evaluate only the first N subjects')
    args = ap.parse_args()

    # infer_crown runs with CLIK's working directory, so every path handed to a
    # subprocess must be absolute or it resolves against the wrong folder
    for k in ('ckpt', 'out', 'converted', 'as_is', 'ids'):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    sids = test_subjects(args.converted, args.ids, args.limit)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    run_inference(args.ckpt, sids, args.converted, args.out)
    print(f'inference finished in {(time.time() - t0) / 60:.0f} min', flush=True)

    subprocess.run([sys.executable, '-u', os.path.join(STEP2, 'scripts', 'evaluate_alignment.py'),
                    '--converted', args.converted, '--output', args.out], check=True)

    res = compare(os.path.join(args.as_is, 'alignment_metrics.csv'),
                  os.path.join(args.out, 'alignment_metrics.csv'), sids)
    print(f'\n=== comparison over {res["n"]} subjects (medians) ===')
    print(f'  {"metric":20s} {"as-is":>10s} {"fine-tuned":>12s} {"change":>12s} {"no movement":>13s}')
    for k in METRICS:
        v = res[k]
        arrow = 'better' if v['change_pct'] < 0 else 'worse'
        print(f'  {LABELS[k]:20s} {v["as_is"]:10.2f} {v["tuned"]:12.2f} '
              f'{v["change_pct"]:+11.1f}% {v["baseline"]:13.2f}   {arrow}')


if __name__ == '__main__':
    main()
