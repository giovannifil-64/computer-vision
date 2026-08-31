"""
make_charts
===========
The three charts step 3 needs, in the visual language step 2 already established:
a light surface, recessive grid, no top or right spine, and the same
scatter-against-the-baseline idea.

Each chart answers one question and is chosen for that job rather than for
variety. The scatter asks whether a model beats leaving the teeth alone, and does
it per subject rather than by a median, because a median can hide a model that
helps half the patients and harms the other half. The ablation chart compares one
measure across variants, so it is bars on a single hue, since five colours would imply
five identities where there is only one quantity. The training chart is the only
thing here that is a time series, so it is lines.

Colours are Okabe-Ito, which is published as colour-vision-deficiency safe, so the
two-series charts stay readable without relying on hue alone: both also carry a
legend, and the scatter separates the series by position as much as by colour.

Functions
---------
- `scatter(runs, out_png)`: Per-subject error against the no-movement baseline.
- `ablation(runs, out_png)`: Each clinical constraint against the base loss.
- `training(runs, out_png)`: Training and validation loss per epoch.
- `timidity(predicted, tensors, out_png)`: Predicted motion against the real one.
- `landmark_spread(tensors, out_png)`: How far each landmark moves between patients.
- `main()`: Write whichever charts the given folders allow.

Example
-------
```bash
python make_charts.py --as-is ../../output/asis_split --tuned ../../output/individual_split \\
    --ablation ../../output --out-dir ../../report/figures
```

Notes
-----
- Medians, not means: the per-subject errors are skewed by the few patients whose
  treatment moved the teeth a long way.
"""
import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
# pyplot has to come after the backend is chosen, or it picks its own
import matplotlib.pyplot as plt

SURFACE = '#ffffff'
INK = '#1a1a1a'
INK_SOFT = '#595959'
GRID = '#e6e6e3'
BLUE = '#0072B2'                          # Okabe-Ito, published as CVD safe
VERMILLION = '#D55E00'
PANELS = [('rot_err', 'rot_baseline', 'Tooth rotation', 'degrees'),
          ('pcd_err', 'pcd_baseline', 'Tooth position', 'mm')]

# sized for the width of a report page rather than rendered large and shrunk, so
# the type here is the type the reader sees
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    # Arial and DejaVu ship as plain .ttf files. The macOS Helvetica faces are
    # .ttc collections, which matplotlib renders with uneven stroke weights, so
    # they are deliberately not in this list.
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,                   # real glyphs, not outlines, so text stays selectable
    'ps.fonttype': 42,
    'axes.linewidth': .6,
})


def _save(fig, name):
    """
    Write the figure as a PDF.

    Charts are vector by nature and end up in a typeset report, so a PDF stays
    sharp at any size and weighs a quarter of the equivalent PNG. The mesh renders
    are the exception and stay raster, because that is what they are.
    """
    out = os.path.splitext(name)[0] + '.pdf'
    fig.savefig(out, facecolor=SURFACE, bbox_inches='tight', pad_inches=.02)
    plt.close(fig)
    return out


def _read(folder):
    """Rows of one scored run, keyed by subject."""
    path = os.path.join(folder, 'alignment_metrics.csv')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return {r['subject']: r for r in csv.DictReader(fh)}


def _style(ax, title, xlabel, ylabel):
    """Apply the shared chart styling to one axis."""
    ax.set_title(title, color=INK, pad=6, loc='left')
    ax.set_xlabel(xlabel, color=INK_SOFT, labelpad=4)
    ax.set_ylabel(ylabel, color=INK_SOFT, labelpad=4)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=3, width=.6)


def scatter(runs, out_png):
    """
    Per-subject error against the no-movement baseline, for two models.

    Parameters
    ----------
    - `runs (list)`: `(label, rows, colour)` triples; the first is drawn behind.
    - `out_png (str)`: Where to write the PNG.

    Returns
    -------
    - `str`: `out_png`.

    Notes
    -----
    - The diagonal is the only reference that matters: a point above it means the
      prediction ended up further from the post-treatment scan than the untouched
      dentition already was.
    """
    subjects = sorted(set.intersection(*[set(r) for _, r, _ in runs]))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), facecolor=SURFACE)

    for ax, (key, base_key, title, unit) in zip(axes, PANELS):
        lim = 0
        for z, (label, rows, colour) in enumerate(runs):
            x = np.array([float(rows[s][base_key]) for s in subjects])
            y = np.array([float(rows[s][key]) for s in subjects])
            lim = max(lim, x.max(), y.max())
            ax.scatter(x, y, s=9, c=colour, alpha=.65, linewidths=0, zorder=3 + z,
                       label=f'{label}: {(y > x).mean() * 100:.0f}% above')
        lim *= 1.04
        ax.plot([0, lim], [0, lim], color=INK_SOFT, lw=.9, ls=(0, (4, 3)), zorder=2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        _style(ax, title, f'error if nothing moves ({unit})', f'error of the model ({unit})')
        leg = ax.legend(loc='upper left', frameon=False, handletextpad=.4,
                        borderpad=.1, labelspacing=.35)
        for text in leg.get_texts():
            text.set_color(INK_SOFT)

    fig.suptitle(f'Each model against leaving the teeth alone, on {len(subjects)} test subjects',
                 fontsize=10, color=INK, x=.012, ha='left', y=1.02)
    fig.text(.012, -.04, 'Above the diagonal: the prediction is further from the post-treatment '
                         'scan than the initial dentition already was.',
             fontsize=7, color=INK_SOFT)
    fig.tight_layout(w_pad=2.2)
    return _save(fig, out_png)


def ablation(variants, out_png):
    """
    Each clinical constraint against the base loss.

    Parameters
    ----------
    - `variants (list)`: `(label, rows)` pairs, the first being the reference.
    - `out_png (str)`: Where to write the PNG.

    Returns
    -------
    - `str`: `out_png`.

    Notes
    -----
    - One hue for every bar: the variants are values of a single measure, not
      separate identities, and the axis already names them. The reference line is
      the base loss, so a bar reaching past it is a constraint that made things
      worse.
    """
    subjects = sorted(set.intersection(*[set(r) for _, r in variants]))
    labels = [label for label, _ in variants]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), facecolor=SURFACE)

    for ax, (key, base_key, title, unit) in zip(axes, PANELS):
        values = [np.median([float(rows[s][key]) for s in subjects]) for _, rows in variants]
        no_move = np.median([float(variants[0][1][s][base_key]) for s in subjects])
        pos = np.arange(len(values))[::-1]
        top = max(max(values), no_move) * 1.28
        # Reference lines sit at their own value, so which of the two comes first
        # is itself the result: on rotation the base loss beats no movement, on
        # position it does not. Each label therefore goes on the side of its line
        # that faces away from the other, which keeps them apart whatever the order.
        refs = sorted(((values[0], INK_SOFT, 'base loss'),
                       (no_move, VERMILLION, 'no movement')), key=lambda r: r[0])
        for outer, (x, colour, name) in zip(('right', 'left'), refs):
            ax.axvline(x, color=colour, lw=.8, ls=(0, (2, 2)), zorder=1)
            offset = top * (-.01 if outer == 'right' else .01)
            ax.text(x + offset, len(values) - .45, name, fontsize=6.5, color=colour,
                    va='bottom', ha=outer, clip_on=False)
        ax.barh(pos, values, height=.58, color=BLUE, zorder=3)
        for p, v in zip(pos, values):
            ax.text(v + top * .015, p, f'{v:.2f}', va='center', fontsize=7, color=INK_SOFT)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.set_ylim(-.6, len(values) + .05)
        ax.set_xlim(0, top)
        _style(ax, title, unit, '')
        ax.grid(axis='y', visible=False)
        ax.tick_params(axis='y', length=0)

    fig.suptitle(f'Clinical constraints, each against the base loss, on {len(subjects)} subjects',
                 fontsize=10, color=INK, x=.012, ha='left', y=1.04)
    fig.text(.012, -.06, 'Lower is better. A bar past the dashed line is a constraint that made '
                         'the model worse.', fontsize=7, color=INK_SOFT)
    fig.tight_layout(w_pad=2.6)
    return _save(fig, out_png)


def training(logs, out_png):
    """
    Training and validation loss per epoch, one panel per run.

    Parameters
    ----------
    - `logs (list)`: `(label, path_to_log_csv)` pairs.
    - `out_png (str)`: Where to write the PNG.

    Returns
    -------
    - `str`: `out_png`.

    Notes
    -----
    - Both panels share a y-axis, which is the whole point: the gap between the two
      lines is what separates a run that is fitting from one that is memorising.
    """
    fig, axes = plt.subplots(1, len(logs), figsize=(7.0, 2.9), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)

    for first, (ax, (label, path)) in enumerate(zip(axes, logs)):
        first = first == 0
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        epochs = [int(r['epoch']) + 1 for r in rows]
        train = [float(r['train_loss']) for r in rows]
        val = [(int(r['epoch']) + 1, float(r['val_base'])) for r in rows
               if r.get('val_base') and r['val_base'] != 'nan']
        ax.plot(epochs, train, color=BLUE, lw=1.2, label='training')
        ax.plot([e for e, _ in val], [v for _, v in val], color=VERMILLION, lw=1.2,
                marker='o', markersize=2.6, markeredgewidth=0, label='validation')
        gap = val[-1][1] / train[-1] if train[-1] else float('nan')
        # bottom left: the training curve ends bottom right, so that corner is taken
        ax.text(.03, .04, f'final gap {gap:.1f}x', transform=ax.transAxes, ha='left',
                va='bottom', fontsize=7.5, color=INK_SOFT)
        # the panels share a y axis, so only the first one carries its label
        _style(ax, label, 'epoch', 'masked MSE (normalised units)' if first else '')
        ax.set_yscale('log')
        leg = ax.legend(loc='upper right', frameon=False, handletextpad=.5,
                        borderpad=.1, labelspacing=.3)
        for text in leg.get_texts():
            text.set_color(INK_SOFT)

    fig.suptitle('Fine-tuning converges well before the epoch budget runs out',
                 fontsize=10, color=INK, x=.012, ha='left', y=1.04)
    fig.text(.012, -.06, 'Validation is the base reconstruction term in both panels, so the two '
                         'runs are measured by the same yardstick.', fontsize=7, color=INK_SOFT)
    fig.tight_layout(w_pad=2.2)
    return _save(fig, out_png)


def timidity(predicted, tensors, out_png, limit=100):
    """
    How big the motions the model predicts are, against the ones that happened.

    Parameters
    ----------
    - `predicted (str)`: `.npz` written by stage C.
    - `tensors (str)`: Folder of prepared tensors, for the exact targets.
    - `out_png (str)`: Where to write the PNG.
    - `limit (int, optional)`: Subjects to include. Default `100`.

    Returns
    -------
    - `str`: `out_png`.

    Notes
    -----
    - One point per tooth, and the diagonal is what a perfectly calibrated model
      would sit on. A cloud that follows the diagonal but flattened towards the
      horizontal is a model that has found the right direction and is holding
      back, which is what a squared-error loss does when the target is ambiguous.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from diagnose import TEETH, tooth_landmark_indices, SCALE, kabsch, rotation_error

    data = np.load(predicted)
    ids, pred = [str(s) for s in data['ids']], data['landmarks']
    tp, tt, rp, rt = [], [], [], []
    for i, sid in enumerate(ids[:limit]):
        path = os.path.join(tensors, f'{sid}.npz')
        if not os.path.exists(path):
            continue
        z = np.load(path)
        mask, target, cond, p_all = z['mask'], z['target'], z['cond'][:, :3], pred[i].T
        for j, _ in enumerate(TEETH):
            if not mask[j]:
                continue
            idx = tooth_landmark_indices(j)
            p, t, c = p_all[idx] * SCALE, target[idx] * SCALE, cond[idx] * SCALE
            rot_p, _, _ = kabsch(c, p)
            rot_t, _, _ = kabsch(c, t)
            tp.append(np.linalg.norm(p.mean(0) - c.mean(0)))
            tt.append(np.linalg.norm(t.mean(0) - c.mean(0)))
            rp.append(rotation_error(np.eye(3), rot_p))
            rt.append(rotation_error(np.eye(3), rot_t))

    panels = [(np.array(tt), np.array(tp), 'Translation', 'mm'),
              (np.array(rt), np.array(rp), 'Rotation', 'degrees')]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), facecolor=SURFACE)
    for ax, (x, y, title, unit) in zip(axes, panels):
        lim = np.percentile(np.concatenate([x, y]), 99) * 1.05
        ax.plot([0, lim], [0, lim], color=INK_SOFT, lw=.9, ls=(0, (4, 3)), zorder=2)
        ax.scatter(x, y, s=5, c=BLUE, alpha=.28, linewidths=0, zorder=3)
        slope = np.median(y) / np.median(x)
        ax.plot([0, lim], [0, lim * slope], color=VERMILLION, lw=1.4, zorder=4)
        ax.text(.04, .95, f'predicted / real = {slope:.2f}\ncorrelation {np.corrcoef(x, y)[0, 1]:.2f}',
                transform=ax.transAxes, va='top', fontsize=7.5, color=INK_SOFT)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        _style(ax, title, f'motion that happened ({unit})', f'motion predicted ({unit})')

    fig.suptitle(f'The model finds the direction and holds back, over {len(tp)} teeth',
                 fontsize=10, color=INK, x=.012, ha='left', y=1.02)
    fig.text(.012, -.04, 'Dashed: a perfectly calibrated model. Solid: the ratio of the medians.',
             fontsize=7, color=INK_SOFT)
    fig.tight_layout(w_pad=2.2)
    return _save(fig, out_png)


def landmark_spread(tensors, out_png, limit=250):
    """
    How far each landmark moves between patients, one row per tooth position.

    Parameters
    ----------
    - `tensors (str)`: Folder of prepared `.npz` files.
    - `out_png (str)`: Where to write the figure.
    - `limit (int, optional)`: Patients to read. Default `250`.

    Returns
    -------
    - `str`: The written path.

    Notes
    -----
    - The centroid is drawn apart because it does a different job: CLIK does not
      predict it, so it says how much the crown itself varies. It is a reference
      rather than a bound, and a sharp feature like a cusp tip can be steadier.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from landmark_consistency import collect, spread, LANDMARK_IDS, TOOTH_TYPE

    per_tooth = collect(tensors, limit)
    teeth = [t for t in sorted(per_tooth) if len(per_tooth[t]) >= 30]
    fig, ax = plt.subplots(figsize=(7.0, 6.4), facecolor=SURFACE)

    for row, tid in enumerate(reversed(teeth)):
        values, _ = spread(per_tooth[tid])
        ax.scatter(values[1:], [row] * (len(values) - 1), s=14, c=BLUE, alpha=.75,
                   linewidths=0, zorder=3, label='predicted landmark' if row == 0 else None)
        ax.scatter(values[:1], [row], s=26, c=VERMILLION, marker='|', linewidths=1.6,
                   zorder=4, label='centroid, not predicted' if row == 0 else None)

    ax.set_yticks(range(len(teeth)))
    ax.set_yticklabels([f'{t}  {TOOTH_TYPE[t][:3]}' for t in reversed(teeth)], fontsize=6.5)
    ax.set_xlim(0, None)
    _style(ax, '', 'movement between patients (mm)', 'tooth position')
    ax.grid(axis='y', visible=False)
    ax.tick_params(axis='y', length=0)
    leg = ax.legend(loc='lower right', frameon=False, handletextpad=.4, borderpad=.3)
    for text in leg.get_texts():
        text.set_color(INK_SOFT)

    fig.suptitle(f'A landmark should mark the same spot on everyone, and some do not',
                 fontsize=10, color=INK, x=.012, ha='left', y=.98)
    fig.text(.012, -.03, 'Every landmark of one tooth, over up to 250 patients, after position, '
                         'orientation and size have been\nremoved. The centroid is not predicted '
                         'by the network, so it says how much the crown itself varies.',
             fontsize=7, color=INK_SOFT)
    fig.tight_layout()
    return _save(fig, out_png)


def main():
    """Write whichever of the charts the given folders allow."""
    ap = argparse.ArgumentParser(description='Charts for the step-3 report.')
    ap.add_argument('--as-is', required=True, help='scored run of the released model')
    ap.add_argument('--tuned', required=True, help='scored run of the fine-tuned model')
    ap.add_argument('--ablation', help='folder holding the abl_* runs')
    ap.add_argument('--predicted', help='.npz of the tuned run, for the calibration chart')
    ap.add_argument('--tensors', help='prepared tensors matching --predicted, for the '
                                      'calibration chart')
    ap.add_argument('--consistency', help='any folder of prepared tensors, for the landmark '
                                          'spread chart; the training ones give the most patients')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    as_is, tuned = _read(args.as_is), _read(args.tuned)
    if as_is and tuned:
        print('  ' + scatter([('CLIK as released', as_is, BLUE),
                              ('CLIK fine-tuned', tuned, VERMILLION)],
                             os.path.join(args.out_dir, 'models_vs_baseline.png')))

    if args.consistency:
        print('  ' + landmark_spread(args.consistency,
                                     os.path.join(args.out_dir, 'landmark_spread.png')))

    if args.predicted and args.tensors:
        print('  ' + timidity(args.predicted, args.tensors,
                              os.path.join(args.out_dir, 'calibration.png')))

    if args.ablation:
        wanted = [('base loss only', 'abl_base_eval'), ('+ arch 0.1', 'abl_arch_eval'),
                  ('+ arch 0.001', 'abl_arch_low_eval'), ('+ contact', 'abl_contact_eval'),
                  ('+ orientation', 'abl_individual_eval'), ('all three (paper)', 'abl_paper_eval')]
        variants = [(label, _read(os.path.join(args.ablation, folder))) for label, folder in wanted]
        variants = [(label, rows) for label, rows in variants if rows]
        if len(variants) > 1:
            print('  ' + ablation(variants, os.path.join(args.out_dir, 'ablation.png')))

        logs = [(label, os.path.join(args.ablation, run, 'log.csv'))
                for label, run in (('base loss only', 'abl_base'),
                                   ('+ orientation', 'abl_individual'))]
        logs = [(label, path) for label, path in logs if os.path.exists(path)]
        if logs:
            print('  ' + training(logs, os.path.join(args.out_dir, 'training.png')))


if __name__ == '__main__':
    main()
