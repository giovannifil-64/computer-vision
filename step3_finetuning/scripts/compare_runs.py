"""
compare_runs
============
Put several scored runs side by side and say which differences are real.

Every run in step 3 ends in an `alignment_metrics.csv` written by the step-2
evaluation, so comparing models is a matter of reading those files rather than
recomputing anything. Two things make the comparison worth automating: the runs
must be restricted to the subjects they all have, and a difference in medians
means nothing on its own.

The significance test is paired, and legitimately so: every run here is scored on
the same subjects, from the same stage-A tensors, so the two arms differ only in
the diffusion weights. That is a much stronger design than comparing independent
runs, and it is what lets a change of a tenth of a millimetre be called real.

Functions
---------
- `load(folder)`: The metrics of one scored run, keyed by subject.
- `common_subjects(runs)`: Subjects present in every run.
- `summarise(rows, subjects)`: Median error and no-movement baseline per metric.
- `compare(runs, reference, subjects)`: Each run against the reference, with p-values.
- `main()`: Print the comparison for the runs named on the command line.

Example
-------
```bash
python compare_runs.py --runs "as-is=../output/asis_split" "fine-tuned=../output/tuned_split"
python compare_runs.py --reference "base=../output/abl_base_eval" \\
    --runs "+ arch=../output/abl_arch_eval" "+ contact=../output/abl_contact_eval"
```

Notes
-----
- The no-movement baseline is the error left by not moving the teeth at all, so a
  model is only useful once it beats that column, not merely once it improves.
- Medians are quoted rather than means: the per-subject errors are skewed by a few
  patients whose treatment moved the teeth a long way.
"""
import os
import csv
import argparse
import numpy as np
from scipy.stats import wilcoxon

METRICS = (('rot_err', 'rot_baseline', 'rotation (degrees)'),
           ('trans_err', 'trans_baseline', 'translation'),
           ('pcd_err', 'pcd_baseline', 'point cloud (mm)'))


def load(folder):
    """
    Read one scored run.

    Parameters
    ----------
    - `folder (str)`: Directory holding `alignment_metrics.csv`.

    Returns
    -------
    - `dict`: Subject id to its row of metrics.

    Raises
    ------
    - `FileNotFoundError`: If the run has not been scored.
    """
    path = os.path.join(folder, 'alignment_metrics.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{folder}: alignment_metrics.csv is missing')
    with open(path) as fh:
        return {row['subject']: row for row in csv.DictReader(fh)}


def common_subjects(runs):
    """
    The subjects every run has in common.

    Parameters
    ----------
    - `runs (dict)`: Label to loaded run.

    Returns
    -------
    - `list`: Sorted subject ids.
    """
    return sorted(set.intersection(*[set(r) for r in runs.values()]))


def summarise(rows, subjects):
    """
    Median error and no-movement baseline for one run.

    Parameters
    ----------
    - `rows (dict)`: A loaded run.
    - `subjects (list)`: Subjects to include.

    Returns
    -------
    - `dict`: Metric key to `(median_error, median_baseline)`.
    """
    out = {}
    for key, base_key, _ in METRICS:
        err = np.median([float(rows[s][key]) for s in subjects])
        base = np.median([float(rows[s][base_key]) for s in subjects])
        out[key] = (err, base)
    return out


def compare(runs, reference, subjects):
    """
    Each run against the reference, on the subjects they share.

    Parameters
    ----------
    - `runs (dict)`: Label to loaded run, in the order to report them.
    - `reference (str)`: Label of the run everything is measured against.
    - `subjects (list)`: Subjects to include.

    Returns
    -------
    - `dict`: Label to per-metric `{median, baseline, change_pct, p, better}`.

    Notes
    -----
    - `p` is a paired Wilcoxon signed-rank test, which suits per-subject errors
      that are not normally distributed. It is `nan` for the reference itself.
    """
    ref = runs[reference]
    result = {}
    for label, rows in runs.items():
        entry = {}
        for key, base_key, _ in METRICS:
            a = np.array([float(ref[s][key]) for s in subjects])
            b = np.array([float(rows[s][key]) for s in subjects])
            med_a, med_b = np.median(a), np.median(b)
            p = float('nan') if label == reference else wilcoxon(a, b).pvalue
            entry[key] = {'median': med_b,
                          'baseline': np.median([float(rows[s][base_key]) for s in subjects]),
                          'change_pct': 100.0 * (med_b - med_a) / med_a if med_a else float('nan'),
                          'p': p,
                          'better': int((b < a).sum())}
        result[label] = entry
    return result


def _stars(p):
    """A compact significance marker, or a dash for the reference row."""
    if np.isnan(p):
        return '--'
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))


def main():
    """Print the comparison for the runs given on the command line."""
    ap = argparse.ArgumentParser(
        description='Compare several scored runs, with a paired test.')
    ap.add_argument('--runs', nargs='+', required=True,
                    help='one or more "etichetta=cartella"; the first is the reference '
                         'unless --reference is given')
    ap.add_argument('--reference', help='"etichetta=cartella" to measure everything against')
    args = ap.parse_args()

    def parse(spec):
        label, _, path = spec.partition('=')
        return (label, path) if path else (os.path.basename(path or label), label)

    entries = [parse(s) for s in args.runs]
    if args.reference:
        entries.insert(0, parse(args.reference))
    runs = {label: load(path) for label, path in entries}
    reference = entries[0][0]

    subjects = common_subjects(runs)
    result = compare(runs, reference, subjects)

    print(f'\n{len(subjects)} subjects in common, reference: {reference}\n')
    for key, _, label in METRICS:
        base = result[reference][key]['baseline']
        print(f'  {label}   (no movement: {base:.2f})')
        print('  %-22s %10s %12s %8s %14s' % ('', 'median', 'change', 'p', 'improved'))
        for name in runs:
            v = result[name][key]
            change = '' if name == reference else f'{v["change_pct"]:+.1f}%'
            better = '' if name == reference else f'{v["better"]}/{len(subjects)}'
            print('  %-22s %10.2f %12s %8s %14s' %
                  (name, v['median'], change, _stars(v['p']), better))
        print()


if __name__ == '__main__':
    main()
