"""
evaluate_alignment
==================
Score CLIK's predicted post-orthodontic dentition against the real
post-treatment scan (`final`), using the metrics of the CLIK paper: per-tooth
rotation error, translation error and point-cloud error. A "no movement"
baseline (predicting that every tooth stays where it is) is computed alongside,
so the numbers can be read against the effort the task actually requires.

Functions
---------
- `kabsch(A, B)`: Rigid transform mapping point set `A` onto `B`.
- `rotation_error(R_gt, R_pred)`: Angle in degrees between two rotations.
- `evaluate_subject(sid, converted_root, output_root)`: Per-tooth metrics for one subject.
- `main()`: Evaluate every subject and write a CSV plus a summary.

Example
-------
```python
from evaluate_alignment import evaluate_subject
rep = evaluate_subject("0001", "Data_prepost", "Output_prepost")
print(rep['summary'])
```

Notes
-----
- Ground truth per-tooth transforms are recovered exactly, because `ori` and
  `final` contain the same crown meshes rigidly repositioned (verified: Kabsch
  residual ~1e-4 mm).
- Teeth extracted during treatment are absent from `final` and are excluded, as
  are teeth whose mesh was re-tessellated between stages (vertex count differs),
  since point-to-point correspondence no longer holds. Both counts are reported.
"""
import os
import csv
import json
import glob
import argparse
import numpy as np
import trimesh
from meshes import load_mesh


def kabsch(A, B):
    """
    Rigid transform (rotation + translation) mapping point set `A` onto `B`.

    Parameters
    ----------
    - `A (np.ndarray)`: `(N, 3)` source points.
    - `B (np.ndarray)`: `(N, 3)` target points, corresponding one-to-one with `A`.

    Returns
    -------
    - `tuple`: `(R, t, rmsd)` with `R` a `(3, 3)` rotation, `t` a `(3,)` translation
      such that `R @ a + t ≈ b`, and `rmsd` the residual in mm.
    """
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ ca
    rmsd = float(np.sqrt((((R @ A.T).T + t - B) ** 2).sum(1).mean()))
    return R, t, rmsd


def rotation_error(R_gt, R_pred):
    """
    Angular difference between two rotation matrices, in degrees.

    Parameters
    ----------
    - `R_gt (np.ndarray)`: `(3, 3)` ground-truth rotation.
    - `R_pred (np.ndarray)`: `(3, 3)` predicted rotation.

    Returns
    -------
    - `float`: `|alpha|` in degrees, from `2 cos(alpha) = trace(R_pred^-1 R_gt) - 1`.
    """
    c = (np.trace(R_pred.T @ R_gt) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def evaluate_subject(sid, converted_root, output_root):
    """
    Compute per-tooth alignment metrics for one subject.

    Parameters
    ----------
    - `sid (str)`: Subject id.
    - `converted_root (str)`: Converter output (holds `initial/` and `final/`).
    - `output_root (str)`: CLIK output (holds `<sid>/results/*.ply`).

    Returns
    -------
    - `dict` or `None`: `{'sid', 'teeth', 'summary'}` where `teeth` is a list of
      per-tooth records and `summary` holds the medians; `None` if nothing usable.

    Notes
    -----
    - For every tooth the ground-truth transform is `ori -> final` and the
      predicted one is `ori -> prediction`; the point-cloud error compares the
      predicted vertices with the ground-truth vertices directly.
    """
    ini_dir = os.path.join(converted_root, sid, 'initial')
    fin_dir = os.path.join(converted_root, sid, 'final')
    res_dir = os.path.join(output_root, sid, 'results')
    if not (os.path.isdir(ini_dir) and os.path.isdir(fin_dir) and os.path.isdir(res_dir)):
        return None

    # CLIK stores the rigid transform it applied to each tooth; using it avoids
    # relying on the exported mesh, whose vertices are merged on export and so no
    # longer correspond one-to-one with the input.
    tf_path = os.path.join(res_dir, 'transformation.json')
    if not os.path.exists(tf_path):
        return None
    transforms = {int(k.split('-')[1]): np.asarray(v, float)
                  for k, v in json.load(open(tf_path)).items()}

    rows, skipped_extracted, skipped_mismatch = [], 0, 0
    for f in sorted(glob.glob(os.path.join(ini_dir, '*.stl'))):
        tid = int(os.path.basename(f)[:-4])
        fin_f = os.path.join(fin_dir, f'{tid}.stl')
        if not os.path.exists(fin_f):
            skipped_extracted += 1
            continue
        if tid not in transforms:
            continue

        A = load_mesh(f, process=False).vertices
        B = load_mesh(fin_f, process=False).vertices
        if A.shape != B.shape:
            skipped_mismatch += 1
            continue

        M = transforms[tid]
        R_pr, t_pr = M[:3, :3], M[:3, 3]
        P = (R_pr @ A.T).T + t_pr          # CLIK's prediction, same tessellation as A
        R_gt, t_gt, res_gt = kabsch(A, B)
        rows.append({
            'tooth': tid,
            'rot_err': rotation_error(R_gt, R_pr),
            'trans_err': float(np.abs(t_gt - t_pr).sum() / 3.0),
            'pcd_err': float(np.linalg.norm(P - B, axis=1).mean()),
            # "no movement" baseline: predict the tooth stays in its initial pose
            'rot_moved': rotation_error(R_gt, np.eye(3)),
            'trans_moved': float(np.abs(t_gt).sum() / 3.0),
            'pcd_moved': float(np.linalg.norm(A - B, axis=1).mean()),
            'gt_residual': res_gt,
        })

    if not rows:
        return None
    med = lambda k: float(np.median([r[k] for r in rows]))
    return {
        'sid': sid,
        'teeth': rows,
        'summary': {
            'n_teeth': len(rows),
            'skipped_extracted': skipped_extracted,
            'skipped_mismatch': skipped_mismatch,
            'rot_err': med('rot_err'), 'rot_baseline': med('rot_moved'),
            'trans_err': med('trans_err'), 'trans_baseline': med('trans_moved'),
            'pcd_err': med('pcd_err'), 'pcd_baseline': med('pcd_moved'),
        },
    }


def main():
    """
    Evaluate every converted subject and write per-tooth CSV + JSON summary.

    Example
    -------
    ```bash
    python evaluate_alignment.py --converted ../Data_prepost --output ../Output_prepost
    ```
    """
    ap = argparse.ArgumentParser(description="Score CLIK's predicted alignment against the post-treatment scan.")
    ap.add_argument('--converted', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    sids = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(args.converted, '*', 'center.json')))
    reports = {}
    for sid in sids:
        rep = evaluate_subject(sid, args.converted, args.output)
        if rep:
            reports[sid] = rep['summary']
            s = rep['summary']
            print(f"  {sid}: {s['n_teeth']:2d} teeth | rot {s['rot_err']:5.2f}° (base {s['rot_baseline']:5.2f}°)"
                  f" | trans {s['trans_err']:5.2f} (base {s['trans_baseline']:5.2f})"
                  f" | pcd {s['pcd_err']:5.2f} (base {s['pcd_baseline']:5.2f}) mm")

    if not reports:
        print('No subject could be evaluated.')
        return
    out_csv = os.path.join(args.output, 'alignment_metrics.csv')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        cols = ['n_teeth', 'skipped_extracted', 'skipped_mismatch',
                'rot_err', 'rot_baseline', 'trans_err', 'trans_baseline', 'pcd_err', 'pcd_baseline']
        w.writerow(['subject'] + cols)
        for sid, s in reports.items():
            w.writerow([sid] + [f'{s[c]:.4f}' if isinstance(s[c], float) else s[c] for c in cols])
    json.dump(reports, open(os.path.join(args.output, 'alignment_summary.json'), 'w'), indent=2)

    agg = lambda k: float(np.median([s[k] for s in reports.values()]))
    print(f"\n=== {len(reports)} subjects ===")
    print(f"  rotation    CLIK {agg('rot_err'):6.2f}°   no-movement {agg('rot_baseline'):6.2f}°")
    print(f"  translation CLIK {agg('trans_err'):6.2f}    no-movement {agg('trans_baseline'):6.2f}")
    print(f"  point cloud CLIK {agg('pcd_err'):6.2f} mm  no-movement {agg('pcd_baseline'):6.2f} mm")
    print(f"  -> {out_csv}")


if __name__ == '__main__':
    main()
