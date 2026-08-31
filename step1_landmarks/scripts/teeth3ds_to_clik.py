"""
teeth3ds_to_clik
================
Convert Teeth3DS intra-oral scans (per-arch OBJ + per-vertex FDI segmentation) into the per-tooth folder layout that CLIK's `infer_crown.py` expects. This is a pure preprocessing step; it does not modify CLIK.

Functions
---------
- `extract_teeth(obj_path, json_path)`: Slice one arch into per-tooth crown meshes (keyed by Universal id).
- `convert_patient(patient_dir, out_root)`: Convert one patient (upper+lower) and write the CLIK layout.
- `main()`: CLI entry point converting every patient folder in a directory.

Example
-------
```python
from teeth3ds_to_clik import convert_patient
convert_patient("/Users/me/Downloads/tmp/ZKJEPFDD", "Data_teeth3ds")
# -> Data_teeth3ds/ZKJEPFDD/initial/<id>.stl  and  Data_teeth3ds/ZKJEPFDD/center.json
```

Notes
-----
- Input is segmented per tooth via per-vertex FDI labels (0 = gingiva/base).
- Teeth are renumbered FDI -> Universal (the system CLIK uses) and the whole dentition is recentred at the origin; the offset is saved to `center.json` so ground-truth landmarks can later be brought into the same frame.
"""

import os
import json
import glob
import argparse
import numpy as np
from meshes import load_mesh

# FDI (ISO-3950) -> Universal (1..32), the numbering CLIK expects.
FDI2U = {
    18: 1, 17: 2, 16: 3, 15: 4, 14: 5, 13: 6, 12: 7, 11: 8,          # upper right
    21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,   # upper left
    38: 17, 37: 18, 36: 19, 35: 20, 34: 21, 33: 22, 32: 23, 31: 24,  # lower left
    41: 25, 42: 26, 43: 27, 44: 28, 45: 29, 46: 30, 47: 31, 48: 32,  # lower right
}


def extract_teeth(obj_path, json_path, min_faces=50):
    """
    Slice one arch mesh into per-tooth crown meshes using per-vertex FDI labels.

    Parameters
    ----------
    - `obj_path (str)`: Path to the arch mesh (`<pid>_<jaw>.obj`).
    - `json_path (str)`: Path to the segmentation JSON with a per-vertex `labels` list (FDI codes, 0 = gingiva).
    - `min_faces (int, optional)`: Drop teeth with fewer than this many faces (noise specks). Default `50`.

    Returns
    -------
    - `dict`: `{universal_id (int): trimesh.Trimesh}` for every tooth found in this arch.

    Raises
    ------
    - `AssertionError`: If the number of labels does not match the number of vertices.

    Notes
    -----
    - A face is assigned to a tooth when at least 2 of its 3 vertices carry that tooth's FDI label, which is robust to boundary vertices between teeth/gingiva.
    """
    mesh = load_mesh(obj_path, process=False)  # keeps vertex order == labels
    labels = np.asarray(json.load(open(json_path))["labels"])

    assert len(labels) == len(mesh.vertices), f"label/vertex mismatch in {obj_path}"

    face_lab = labels[mesh.faces]  # (F, 3) FDI per face-vertex
    teeth = {}

    for fdi in np.unique(labels):
        if fdi == 0 or int(fdi) not in FDI2U:  # 0 = gingiva/base
            continue
        mask = (face_lab == fdi).sum(axis=1) >= 2
        if mask.sum() < min_faces:
            continue
        teeth[FDI2U[int(fdi)]] = mesh.submesh([mask], append=True)

    return teeth


def convert_patient(patient_dir, out_root):
    """
    Convert one Teeth3DS patient (upper + lower) into the CLIK folder layout.

    Parameters
    ----------
    - `patient_dir (str)`: Folder holding `<pid>_upper/lower.obj` and matching `.json`.
    - `out_root (str)`: Output root; results go to `<out_root>/<pid>/initial/`.

    Returns
    -------
    - `str` or `None`: The patient id on success, or `None` if no teeth were extracted.

    Notes
    -----
    - Writes one `<universal_id>.stl` per tooth plus `center.json` (the offset subtracted to recentre the dentition, used later for frame alignment).
    """
    pid = os.path.basename(patient_dir.rstrip("/"))
    teeth = {}

    for jaw in ("upper", "lower"):
        obj = os.path.join(patient_dir, f"{pid}_{jaw}.obj")
        js = os.path.join(patient_dir, f"{pid}_{jaw}.json")
        if os.path.exists(obj) and os.path.exists(js):
            teeth.update(extract_teeth(obj, js))

    if not teeth:
        print(f"  [skip] {pid}: no teeth extracted")
        return None

    center = np.concatenate([t.vertices for t in teeth.values()], 0).mean(0)
    out_dir = os.path.join(out_root, pid, "initial")

    os.makedirs(out_dir, exist_ok=True)

    for uid, t in sorted(teeth.items()):
        t = t.copy()
        t.vertices = t.vertices - center
        t.export(os.path.join(out_dir, f"{uid}.stl"))

    json.dump(
        {"center": center.tolist(), "teeth": sorted(teeth.keys())},
        open(os.path.join(out_root, pid, "center.json"), "w"),
        indent=2,
    )

    print(f"  [ok] {pid}: {len(teeth)} teeth -> {out_dir}  (ids {sorted(teeth.keys())})")

    return pid


def main():
    """
    CLI entry point: convert every patient subfolder under `--in_root`.

    Example
    -------
    ```bash
    python teeth3ds_to_clik.py -i /Users/me/Downloads/tmp -o Data_teeth3ds
    ```
    """
    ap = argparse.ArgumentParser(description="Convert Teeth3DS scans to CLIK per-tooth layout.")
    ap.add_argument("-i", "--in_root", required=True, help="folder with Teeth3DS patient subfolders")
    ap.add_argument("-o", "--out_root", required=True, help="output root (CLIK-style)")

    args = ap.parse_args()
    dirs = [d for d in glob.glob(os.path.join(args.in_root, "*")) if os.path.isdir(d)]

    print(f"Found {len(dirs)} patient folders in {args.in_root}")

    for d in sorted(dirs):
        convert_patient(d, args.out_root)


if __name__ == "__main__":
    main()
