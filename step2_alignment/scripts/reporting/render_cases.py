"""
render_cases
============
Render qualitative comparisons in the style of the CLIK paper: for one subject,
the initial (pre-treatment) dentition, CLIK's prediction, and the ground-truth
post-treatment scan, shown side by side with an identical camera so the three
panels can be compared directly. Upper and lower arch are stacked as two rows.

Functions
---------
- `load_stage(sid, stage, converted_root, output_root)`: Load the teeth of one stage.
- `render_subject(sid, ...)`: Write the 2x3 comparison figure for one subject.
- `pick_cases(csv_path)`: Choose best / median / worst subjects by point-cloud error.
- `main()`: Render a set of representative subjects.

Example
-------
```bash
python render_cases.py --converted ../Data_prepost --output ../Output_prepost --out-dir ../report_prepost
```

Notes
-----
- Meshes are already in CLIK's canonical frame, so a fixed occlusal camera works
  for every subject; the same camera is reused across the three panels of a row.
- Teeth extracted during treatment are absent from the ground-truth panel, which
  is expected and visible in the figure.
"""
import os
import csv
import glob
import argparse
import numpy as np

from meshes import load_mesh
import pyvista as pv

pv.OFF_SCREEN = True

UPPER = set(range(1, 17))
TOOTH_COLOR = (0.88, 0.87, 0.85)


def load_stage(sid, stage, converted_root, output_root):
    """
    Load the per-tooth meshes of one stage for a subject.

    Parameters
    ----------
    - `sid (str)`: Subject id.
    - `stage (str)`: `"initial"`, `"final"` (both from the converter) or `"pred"` (CLIK output).
    - `converted_root (str)`: Converter output root.
    - `output_root (str)`: CLIK output root.

    Returns
    -------
    - `dict`: `{tooth_id (int): trimesh.Trimesh}`.
    """
    if stage == 'pred':
        pat = os.path.join(output_root, sid, 'results', '*.ply')
    else:
        pat = os.path.join(converted_root, sid, stage, '*.stl')
    out = {}
    for f in glob.glob(pat):
        name = os.path.basename(f).rsplit('.', 1)[0]
        if name.isdigit():
            out[int(name)] = load_mesh(f)
    return out


def _add(pl, teeth, ids):
    """Add every tooth of one arch to the current pyvista subplot."""
    for t, m in teeth.items():
        if t not in ids:
            continue
        faces = np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64).ravel()
        pl.add_mesh(pv.PolyData(m.vertices, faces), color=TOOTH_COLOR,
                    smooth_shading=True, specular=0.25, specular_power=15)


def render_subject(sid, converted_root, output_root, out_dir, size=(1650, 1100)):
    """
    Write the initial / prediction / ground-truth comparison figure for a subject.

    Parameters
    ----------
    - `sid (str)`: Subject id.
    - `converted_root (str)`: Converter output root.
    - `output_root (str)`: CLIK output root.
    - `out_dir (str)`: Folder for the PNG.
    - `size (tuple, optional)`: Image size in pixels. Default `(1650, 1100)`.

    Returns
    -------
    - `str` or `None`: Path of the written PNG, or `None` if data was missing.
    """
    stages = [('initial', 'Initial (pre-treatment)'),
              ('pred', 'CLIK prediction'),
              ('final', 'Ground truth (post-treatment)')]
    data = {k: load_stage(sid, k, converted_root, output_root) for k, _ in stages}
    if not all(data.values()):
        return None

    os.makedirs(out_dir, exist_ok=True)
    pl = pv.Plotter(off_screen=True, shape=(2, 3), window_size=size, border=False)
    pl.set_background('white')

    for row, (ids, arch) in enumerate([(UPPER, 'upper'), (set(range(17, 33)), 'lower')]):
        # one camera for the whole row, from the initial dentition
        ref = np.concatenate([m.vertices for t, m in data['initial'].items() if t in ids] or [np.zeros((1, 3))])
        ctr = ref.mean(0)
        span = np.ptp(ref, 0).max() if len(ref) > 1 else 1.0
        sign = 1.0 if arch == 'upper' else -1.0        # look at the occlusal side
        cam = [tuple(ctr + np.array([0, 0, sign * span * 2.2])), tuple(ctr), (-1.0, 0.0, 0.0)]

        for col, (key, title) in enumerate(stages):
            pl.subplot(row, col)
            _add(pl, data[key], ids)
            pl.camera_position = cam
            pl.reset_camera()
            pl.camera.zoom(1.35)
            if row == 0:
                pl.add_text(title, font_size=10, color='black', position='upper_edge')

    out = os.path.join(out_dir, f'{sid}_comparison.png')
    pl.screenshot(out)
    pl.close()
    return out


def pick_cases(csv_path, n=3):
    """
    Pick representative subjects (best, median, worst) by point-cloud error.

    Parameters
    ----------
    - `csv_path (str)`: `alignment_metrics.csv` produced by the evaluation.
    - `n (int, optional)`: Unused placeholder for future selections. Default `3`.

    Returns
    -------
    - `list`: `[(label, subject_id, pcd_err, pcd_baseline)]`.
    """
    rows = sorted(csv.DictReader(open(csv_path)), key=lambda r: float(r['pcd_err']))
    take = [('best', rows[0]), ('median', rows[len(rows) // 2]), ('worst', rows[-1])]
    return [(lab, r['subject'], float(r['pcd_err']), float(r['pcd_baseline'])) for lab, r in take]


def main():
    """Render the representative cases listed in the metrics CSV."""
    ap = argparse.ArgumentParser(description="Render initial / prediction / ground-truth comparisons.")
    ap.add_argument('--converted', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--subjects', nargs='*', help='explicit subject ids (default: best/median/worst)')
    args = ap.parse_args()

    if args.subjects:
        cases = [('chosen', s, float('nan'), float('nan')) for s in args.subjects]
    else:
        cases = pick_cases(os.path.join(args.output, 'alignment_metrics.csv'))

    for lab, sid, err, base in cases:
        p = render_subject(sid, args.converted, args.output, args.out_dir)
        extra = '' if np.isnan(err) else f'  (error {err:.2f} mm vs baseline {base:.2f} mm)'
        print(f'  {lab:7s} {sid}: {p or "SKIPPED"}{extra}')


if __name__ == '__main__':
    main()
