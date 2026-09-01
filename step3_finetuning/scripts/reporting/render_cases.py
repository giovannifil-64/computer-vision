"""
render_cases
============
Render the four-way comparison a reader needs in order to believe the tables:
the dentition before treatment, where CLIK as released puts it, where the
fine-tuned model puts it, and where it actually ended up.

Step 2 already renders a three-column version of this. That code is reused rather
than copied: the mesh loading, the per-arch camera and the tooth material all
come from `step2_alignment`, so the two figures are the same picture with one
column added and can sit next to each other in a report without looking like they
came from different projects.

Functions
---------
- `render_subject(sid, converted, runs, out_dir)`: One subject's comparison figure.
- `pick_cases(csv_path, n)`: Representative subjects, by point-cloud error.
- `main()`: Render the chosen subjects.

Example
-------
```bash
python render_cases.py --converted ../../../step2_alignment/data/Data_prepost \\
    --as-is ../../output/asis_split --tuned ../../output/individual_split \\
    --out-dir ../../report/figures
```

Notes
-----
- Rendering is off-screen, so it needs no display and runs over ssh.
- The camera is shared across a row and derived from the pre-treatment dentition,
  so the four columns are strictly comparable: any difference the eye sees is the
  teeth moving, never the viewpoint changing.
"""
import os
import sys
import csv
import argparse
import importlib.util
import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
STEP2 = os.path.join(BASE, 'step2_alignment', 'scripts')
sys.path.insert(0, STEP2)

# Step 2's renderer, reused wholesale. It is loaded by path and not by name
# because that file is also called render_cases, so a plain import finds this one.
_spec = importlib.util.spec_from_file_location(
    'step2_render_cases', os.path.join(STEP2, 'reporting', 'render_cases.py'))
_step2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_step2)
load_stage, _add, UPPER = _step2.load_stage, _step2._add, _step2.UPPER

LOWER = set(range(17, 33))


def render_subject(sid, converted, runs, out_dir, size=(3000, 1500)):
    """
    Write the four-column comparison figure for one subject.

    Parameters
    ----------
    - `sid (str)`: Subject id.
    - `converted (str)`: Converter output root, holding `initial/` and `final/`.
    - `runs (list)`: `(output_root, title)` pairs for the predictions to show.
    - `out_dir (str)`: Folder for the PNG.
    - `size (tuple, optional)`: Image size in pixels. Default `(3000, 1500)`.

    Returns
    -------
    - `str` or `None`: Path of the written PNG, or `None` if any stage was missing.
    """
    columns = [(load_stage(sid, 'initial', converted, None), 'Initial (pre-treatment)')]
    for root, title in runs:
        columns.append((load_stage(sid, 'pred', converted, root), title))
    columns.append((load_stage(sid, 'final', converted, None), 'Ground truth (post-treatment)'))
    if not all(teeth for teeth, _ in columns):
        return None

    os.makedirs(out_dir, exist_ok=True)
    pl = pv.Plotter(off_screen=True, shape=(2, len(columns)), window_size=size, border=False)
    pl.set_background('white')

    for row, (ids, arch) in enumerate([(UPPER, 'upper'), (LOWER, 'lower')]):
        # one camera per row, taken from the pre-treatment dentition, so the columns
        # differ only by where the teeth are
        ref = np.concatenate([m.vertices for t, m in columns[0][0].items() if t in ids]
                             or [np.zeros((1, 3))])
        centre = ref.mean(0)
        span = np.ptp(ref, 0).max() if len(ref) > 1 else 1.0
        sign = 1.0 if arch == 'upper' else -1.0            # look at the occlusal side
        cam = [tuple(centre + np.array([0, 0, sign * span * 2.2])), tuple(centre), (-1.0, 0.0, 0.0)]

        for col, (teeth, title) in enumerate(columns):
            pl.subplot(row, col)
            _add(pl, teeth, ids)
            pl.camera_position = cam
            pl.reset_camera()
            pl.camera.zoom(1.12)      # leaves a margin, so no arch is clipped
            if row == 0:
                pl.add_text(title, font_size=15, color='black', position='upper_edge')

    out = os.path.join(out_dir, f'{sid}_comparison.png')
    pl.screenshot(out)
    pl.close()
    return out


def pick_cases(csv_path, n=3):
    """
    Representative subjects: the best, the median and the worst by point-cloud error.

    Parameters
    ----------
    - `csv_path (str)`: `alignment_metrics.csv` of the run to rank by.
    - `n (int, optional)`: How many subjects. Default `3`.

    Returns
    -------
    - `list`: Subject ids.

    Notes
    -----
    - Showing only the good cases would be dishonest, and only the bad ones
      uninformative; the median is the one the reader should weigh most.
    """
    with open(csv_path) as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: float(r['pcd_err']))
    if not rows:
        return []
    picks = [0, len(rows) // 2, len(rows) - 1][:n]
    return [rows[i]['subject'] for i in picks]


def main():
    """Render the comparison figure for the chosen subjects."""
    ap = argparse.ArgumentParser(description='Four-way dentition comparison figures.')
    ap.add_argument('--converted', required=True, help='converter output root')
    ap.add_argument('--as-is', required=True, help='predictions of the released model')
    ap.add_argument('--tuned', required=True, help='predictions of the fine-tuned model')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--subjects', nargs='*', help='explicit ids (default: best, median, worst)')
    args = ap.parse_args()

    subjects = args.subjects or pick_cases(os.path.join(args.tuned, 'alignment_metrics.csv'))
    runs = [(args.as_is, 'CLIK as released'), (args.tuned, 'CLIK fine-tuned')]
    for sid in subjects:
        path = render_subject(sid, args.converted, runs, args.out_dir)
        print(f'  {sid}: {path or "skipped, data missing"}', flush=True)


if __name__ == '__main__':
    main()
