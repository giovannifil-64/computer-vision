"""
make_chart
==========
Draw the summary figure of the as-is evaluation: one point per subject, CLIK's
error against the error of leaving the teeth untouched, with the parity diagonal.
Everything above the diagonal is a subject where the prediction ended up further
from the post-treatment scan than the untreated dentition.

Functions
---------
- `draw(csv_path, out_png)`: Render the two-panel figure from the metrics CSV.
- `main()`: CLI entry point.

Example
-------
```bash
python make_chart.py --output ../output/Output_prepost --report ../report
```

Notes
-----
- Rotation and position are shown as separate panels because they carry different
  units; a single series per panel means no legend is needed.
- Colours come from a palette validated for colour-vision deficiency and for
  contrast against the chart surface.
"""
import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SOFT = '#52514e'
SERIES = '#2a78d6'
GRID = '#d9d8d4'

PANELS = [('rot_err', 'rot_baseline', 'Rotazione dei denti', 'gradi'),
          ('pcd_err', 'pcd_baseline', 'Posizione dei denti', 'mm')]


def draw(csv_path, out_png):
    """
    Render the CLIK-vs-baseline scatter figure.

    Parameters
    ----------
    - `csv_path (str)`: `alignment_metrics.csv` produced by the evaluation.
    - `out_png (str)`: Where to write the PNG.

    Returns
    -------
    - `str`: `out_png`.
    """
    rows = list(csv.DictReader(open(csv_path)))
    col = lambda k: np.array([float(r[k]) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=SURFACE)
    for ax, (k, kb, title, unit) in zip(axes, PANELS):
        x, y = col(kb), col(k)
        lim = max(x.max(), y.max()) * 1.05
        ax.plot([0, lim], [0, lim], color=INK_SOFT, lw=1.2, ls='--', zorder=1)
        ax.scatter(x, y, s=26, c=SERIES, alpha=.75, edgecolors=SURFACE, linewidths=.6, zorder=3)
        ax.text(.04, .96, f'{(y > x).mean() * 100:.0f}% dei soggetti sopra la linea\n'
                          '(CLIK peggio del non muovere nulla)',
                transform=ax.transAxes, va='top', ha='left', fontsize=9, color=INK_SOFT)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
        ax.set_xlabel(f'errore non muovendo nulla  ({unit})', fontsize=9.5, color=INK_SOFT)
        ax.set_ylabel(f'errore di CLIK  ({unit})', fontsize=9.5, color=INK_SOFT)
        ax.set_title(title, fontsize=11.5, color=INK, pad=10, loc='left')
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=.7); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=9)

    fig.suptitle(f'CLIK as-is rispetto al lasciare i denti fermi, {len(rows)} soggetti di test',
                 fontsize=13, color=INK, x=.02, ha='left', y=.99)
    fig.text(.02, .02, 'Sopra la diagonale: la predizione e piu lontana dalla scansione '
                       'post-trattamento di quanto lo fosse la dentatura iniziale.',
             fontsize=8.5, color=INK_SOFT)
    plt.tight_layout(rect=[0, .04, 1, .95])
    plt.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def main():
    """CLI entry point: build the chart from the evaluation output."""
    ap = argparse.ArgumentParser(description="Summary chart of the as-is evaluation.")
    ap.add_argument('--output', required=True, help='folder with alignment_metrics.csv')
    ap.add_argument('--report', required=True, help='where to write the figure')
    args = ap.parse_args()
    out = draw(os.path.join(args.output, 'alignment_metrics.csv'),
               os.path.join(args.report, 'clik_vs_baseline.png'))
    print(f'scritto {out}')


if __name__ == '__main__':
    main()
