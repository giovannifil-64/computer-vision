"""
run_all
=======
Driver for the Teeth3DS -> CLIK evaluation. Two passes: (1) convert every scan and run CLIK crown-only inference; (2) learn the CLIK-id -> 3DTeethLand-class map from a patient that has ground truth, then render every patient with the SAME 3DTeethLand colour legend (raw CLIK, hybrid, and GT) and evaluate where GT exists.

Functions
---------
- `run_clik(pid, converted_root, output_root, force)`: Run CLIK crown-only inference for one patient.
- `render_patient(pid, ..., class_map)`: Render raw-CLIK / hybrid / GT for one patient (shared palette).
- `main()`: Convert+infer all, learn the map, then render+evaluate all.

Example
-------
```bash
python run_all.py --src ~/Downloads/tmp --gt ~/Downloads/osfstorage-archive
```

Notes
-----
Run inside the `clik-tooth` conda env from any directory; paths resolve relative to the CLIK repo root. Inference runs as a subprocess so checkpoint paths match the original repo. All renders share one orientation and one colour palette, so the raw-CLIK / hybrid / GT images of a patient are directly comparable.
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    clik_root,
    load_center,
    load_clik_landmarks,
    load_gt_landmarks,
    load_arch_mesh,
    colors_by_nearest_class,
    arch_anterior_direction,
    find_gt_files,
    GT_CLASS_COLORS,
    ARCH_IDS,
)
from teeth3ds_to_clik import convert_patient
from evaluate import evaluate_patient, format_table
from visualize import render_arch
from hybrid import learn_class_map, build_hybrid

ROOT = clik_root()


def run_clik(pid, converted_root, output_root, force=False):
    """
    Run CLIK crown-only inference (`infer_crown.py -v`) for one patient.

    Parameters
    ----------
    - `pid (str)`: Patient id.
    - `converted_root (str)`: Converter output root (CLIK input).
    - `output_root (str)`: Where CLIK writes its results.
    - `force (bool, optional)`: Re-run even if output already exists. Default `False`.

    Returns
    -------
    - `bool`: `True` if CLIK output is present after the call.

    Raises
    ------
    - `subprocess.CalledProcessError`: If the inference process exits non-zero.
    """
    if not force and glob.glob(os.path.join(output_root, pid, "landmarks", "*.json")):
        print(f"  [skip CLIK] {pid}: output already present")
        return True
    cmd = [
        sys.executable,
        os.path.join(ROOT, "Code", "infer_crown.py"),
        "-i",
        os.path.join(converted_root, pid),
        "-o",
        output_root,
        "-v",
    ]
    print(f"  [CLIK] {pid} ...")
    subprocess.run(cmd, cwd=ROOT, check=True)
    return bool(glob.glob(os.path.join(output_root, pid, "landmarks", "*.json")))


def render_patient(pid, src_root, converted_root, output_root, gt_root, class_map):
    """
    Render raw-CLIK, hybrid, and (where available) GT landmarks for one patient.

    Parameters
    ----------
    - `pid (str)`: Patient id.
    - `src_root (str)`: Teeth3DS source folder.
    - `converted_root (str)`: Converter output root.
    - `output_root (str)`: CLIK output root (also where figures are written).
    - `gt_root (str)`: Ground-truth root.
    - `class_map (dict or None)`: CLIK-id -> class map (for hybrid + palette colouring).

    Returns
    -------
    - `None`

    Notes
    -----
    - The raw-CLIK image is coloured with the 3DTeethLand palette by nearest class:
      using GT where it exists, otherwise the hybrid landmarks. All three images of
      an arch share `orient_points` (the CLIK points) so the camera is identical.
    """
    center = load_center(converted_root, pid)
    out_dir = os.path.join(output_root, pid)
    hyb = build_hybrid(converted_root, output_root, pid, class_map) if class_map else []

    for jaw, ids in ARCH_IDS.items():
        rng = set(ids)
        mesh = load_arch_mesh(src_root, pid, jaw)
        if mesh is None:
            continue
        anterior = arch_anterior_direction(converted_root, pid, jaw)
        clik_pts, _ = load_clik_landmarks(
            output_root, pid, frame="original", converted_root=converted_root, jaw=jaw
        )
        gt = load_gt_landmarks(gt_root, pid, frame="original", jaw=jaw)
        hp = (
            np.array([p + center for t, c, p in hyb if t in rng])
            if hyb
            else np.empty((0, 3))
        )
        hc = [c for t, c, p in hyb if t in rng]

        # raw CLIK, coloured with the shared palette by nearest class (GT > hybrid)
        if gt is not None:
            ref_pts, ref_cls = gt
        elif len(hp):
            ref_pts, ref_cls = hp, hc
        else:
            ref_pts = None
        clik_colors = (
            colors_by_nearest_class(clik_pts, ref_pts, ref_cls)
            if ref_pts is not None and len(ref_pts)
            else ["orange"] * len(clik_pts)
        )
        render_arch(
            mesh,
            clik_pts,
            clik_colors,
            os.path.join(out_dir, f"{pid}_{jaw}_CLIK.png"),
            f"{pid} {jaw} - CLIK",
            orient_points=clik_pts,
            anterior=anterior,
        )

        if len(hp):
            render_arch(
                mesh,
                hp,
                [GT_CLASS_COLORS[c] for c in hc],
                os.path.join(out_dir, f"{pid}_{jaw}_HYBRID.png"),
                f"{pid} {jaw} - HYBRID",
                orient_points=clik_pts,
                anterior=anterior,
            )

        if gt is not None:
            gp, gc = gt
            render_arch(
                mesh,
                gp,
                [GT_CLASS_COLORS.get(c, "white") for c in gc],
                os.path.join(out_dir, f"{pid}_{jaw}_GT.png"),
                f"{pid} {jaw} - ground truth",
                orient_points=clik_pts,
                anterior=anterior,
            )


def main():
    """
    Convert + infer all patients, learn the class map from a GT patient, then
    render (shared palette) and evaluate; writes `evaluation_summary.json`.
    """
    ap = argparse.ArgumentParser(description="Teeth3DS -> CLIK evaluation driver.")
    ap.add_argument(
        "--src", required=True, help="Teeth3DS source folder (patient subfolders)"
    )
    ap.add_argument(
        "--gt", required=True, help="osfstorage-archive root (ground-truth landmarks)"
    )
    ap.add_argument("--converted", default=os.path.join(ROOT, "Data_teeth3ds"))
    ap.add_argument("--output", default=os.path.join(ROOT, "Output_teeth3ds"))
    args = ap.parse_args()

    pids = sorted(
        os.path.basename(d)
        for d in glob.glob(os.path.join(args.src, "*"))
        if os.path.isdir(d)
    )
    print(f"Patients: {pids}\n")

    # Pass 1: convert + infer
    for pid in pids:
        print(f"== {pid} ==")
        convert_patient(os.path.join(args.src, pid), args.converted)
        run_clik(pid, args.converted, args.output)

    # Learn the CLIK-id -> class map from a patient that has ground truth
    gt_pid = next((p for p in pids if find_gt_files(args.gt, p)), None)
    class_map = (
        learn_class_map(args.converted, args.output, args.gt, gt_pid)
        if gt_pid
        else None
    )
    print(
        f"\nClass map learned from: {gt_pid}\n"
        if gt_pid
        else "\nNo GT patient; hybrid/palette disabled\n"
    )

    # Pass 2: render (shared palette) + evaluate
    reports = {}
    for pid in pids:
        render_patient(pid, args.src, args.converted, args.output, args.gt, class_map)
        rep = evaluate_patient(pid, args.converted, args.output, args.gt, args.src)
        if rep:
            reports[pid] = rep
            print(format_table(rep))
        else:
            print(f"  [no GT] {pid}: rendered CLIK + hybrid only")
        print()

    summary = os.path.join(args.output, "evaluation_summary.json")
    json.dump(reports, open(summary, "w"), indent=2)
    print(
        f"Summary written to {summary}  ({len(reports)} patient(s) with ground truth)"
    )


if __name__ == "__main__":
    main()
