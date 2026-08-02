"""
common
======
Shared helpers for the Teeth3DS -> CLIK evaluation pipeline: repository paths,
coordinate-frame handling, and loaders for both CLIK-predicted landmarks and
3DTeethLand ground-truth landmarks.

Functions
---------
- `clik_root()`: Absolute path to the CLIK-Diffusion repository root.
- `load_center(converted_root, pid)`: Centering offset written by the converter.
- `load_clik_landmarks(output_root, pid, frame, converted_root)`: CLIK-detected landmarks.
- `load_gt_landmarks(gt_root, pid, frame, center)`: 3DTeethLand ground-truth landmarks.
- `load_ios_mesh(src_root, pid)`: Merged upper+lower raw IOS mesh (original coords).
- `clik_landmark_color(landmark_id)`: RGB colour for a CLIK landmark id.

Example
-------
```python
from common import load_center, load_clik_landmarks, load_gt_landmarks
center = load_center("Data_teeth3ds", "ZKJEPFDD")
clik, _ = load_clik_landmarks("Output_teeth3ds", "ZKJEPFDD", frame="center")
gt, cls = load_gt_landmarks(GT_ROOT, "ZKJEPFDD", frame="center", center=center)
```

Notes
-----
- CLIK saves Stage-1 landmarks in the coordinate frame of the meshes it was
  given. Our converter centers those meshes, so CLIK landmarks live in the
  *centered* frame; ground truth lives in the *original* scan frame. Use the
  `frame` argument to bring either set into a common frame before comparing.
"""

import os
import json
import glob
import numpy as np
import trimesh
from meshes import load_mesh

# 3DTeethLand landmark classes -> the EXACT colours used in the dataset's figure
# (Mesial red, Distal green, Cusp blue, Inner yellow, Outer cyan, Facial magenta).
GT_CLASS_COLORS = {
    "Mesial": "#FF0000",
    "Distal": "#00FF00",
    "Cusp": "#0000FF",
    "InnerPoint": "#FFFF00",
    "OuterPoint": "#00FFFF",
    "FacialPoint": "#FF00FF",
}

# Sub-datasets searched when looking up ground-truth landmark files.
_GT_SUBSETS = ("3DTeethLand_landmarks_train", "3DTeethLand_landmarks_test")

# Universal tooth-id ranges per arch (CLIK numbering).
ARCH_IDS = {"upper": range(1, 17), "lower": range(17, 33)}


def arch_anterior_direction(converted_root, pid, jaw):
    """
    Unit vector pointing toward the front (incisors) of an arch.

    Parameters
    ----------
    - `converted_root (str)`: Converter output root (per-tooth crowns).
    - `pid (str)`: Patient id.
    - `jaw (str)`: `"upper"` or `"lower"`.

    Returns
    -------
    - `np.ndarray` or `None`: Unit anterior direction, or `None` if incisors are missing.

    Notes
    -----
    - Computed as `incisor_centroid - arch_centroid`; used to orient renders so the
      incisors sit at the bottom of the image (as in the 3DTeethLand figure).
    """
    init = os.path.join(converted_root, pid, "initial")
    incisor_ids = {"lower": [24, 25, 23, 26], "upper": [8, 9, 7, 10]}[jaw]

    def centroid(ids):
        vs = [
            load_mesh(os.path.join(init, f"{u}.stl")).vertices
            for u in ids
            if os.path.exists(os.path.join(init, f"{u}.stl"))
        ]
        return np.concatenate(vs, 0).mean(0) if vs else None

    inc = centroid(incisor_ids)
    allc = centroid(list(ARCH_IDS[jaw]))
    if inc is None or allc is None:
        return None
    d = inc - allc
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else None


def clik_colors_by_nearest_gt(clik_pts, gt_pts, gt_cls):
    """
    Colour CLIK landmarks by the class of their nearest ground-truth landmark.

    Parameters
    ----------
    - `clik_pts (np.ndarray)`: `(N, 3)` CLIK landmarks (same frame as GT).
    - `gt_pts (np.ndarray)`: `(M, 3)` ground-truth landmarks.
    - `gt_cls (list)`: Length-`M` 3DTeethLand class names.

    Returns
    -------
    - `list`: Length-`N` colours (the same legend as `GT_CLASS_COLORS`), so the
      CLIK image is directly comparable to the GT image.

    Notes
    -----
    - This is a visualisation aid for the GT comparison only; it does not change
      the quantitative metric. Use it where ground truth exists.
    """
    return colors_by_nearest_class(clik_pts, gt_pts, gt_cls)


def colors_by_nearest_class(points, ref_points, ref_classes):
    """
    Colour each point by the class of its nearest reference point (3DTeethLand palette).

    Parameters
    ----------
    - `points (np.ndarray)`: `(N, 3)` points to colour (e.g. raw CLIK landmarks).
    - `ref_points (np.ndarray)`: `(M, 3)` reference points with known classes (GT or hybrid).
    - `ref_classes (list)`: Length-`M` 3DTeethLand class names.

    Returns
    -------
    - `list`: Length-`N` colours from `GT_CLASS_COLORS`, giving every render the same legend.
    """
    from scipy.spatial import cKDTree

    idx = cKDTree(np.asarray(ref_points)).query(np.asarray(points))[1]
    return [GT_CLASS_COLORS.get(ref_classes[i], "white") for i in idx]


def load_arch_mesh(src_root, pid, jaw):
    """
    Load a single arch (upper or lower) IOS mesh in original scan coordinates.

    Parameters
    ----------
    - `src_root (str)`: Folder with the Teeth3DS patient subfolders.
    - `pid (str)`: Patient id.
    - `jaw (str)`: `"upper"` or `"lower"`.

    Returns
    -------
    - `trimesh.Trimesh` or `None`: The arch mesh, or `None` if the file is absent.
    """
    obj = os.path.join(src_root, pid, f"{pid}_{jaw}.obj")
    return load_mesh(obj, process=False) if os.path.exists(obj) else None


def _find_clik_root():
    """Locate the CLIK-Diffusion checkout (env `CLIK_ROOT`, else walk up to a sibling)."""
    env = os.environ.get('CLIK_ROOT')
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    while here != os.path.dirname(here):
        cand = os.path.join(here, 'CLIK-Diffusion')
        if os.path.isdir(os.path.join(cand, 'Code')):
            return cand
        here = os.path.dirname(here)
    raise RuntimeError('CLIK-Diffusion checkout not found; set CLIK_ROOT')


def clik_root():
    """
    Absolute path to the CLIK-Diffusion repository root.

    Returns
    -------
    - `str`: The CLIK-Diffusion checkout (override with the `CLIK_ROOT` env var).

    Example
    -------
    ```python
    ckpt = os.path.join(clik_root(), "Code", "checkpoint")
    ```
    """
    return _find_clik_root()


def load_center(converted_root, pid):
    """
    Load the centering offset the converter subtracted for one patient.

    Parameters
    ----------
    - `converted_root (str)`: Folder produced by the converter (holds `<pid>/center.json`).
    - `pid (str)`: Patient id.

    Returns
    -------
    - `np.ndarray`: Shape `(3,)` offset in millimetres (original - centered).

    Raises
    ------
    - `FileNotFoundError`: If `center.json` is missing (run the converter first).
    """
    path = os.path.join(converted_root, pid, "center.json")
    return np.asarray(json.load(open(path))["center"], dtype=float)


def clik_landmark_color(landmark_id):
    """
    RGB colour for a CLIK landmark id, using CLIK's own `id_color_dict`.

    Parameters
    ----------
    - `landmark_id (str)`: CLIK landmark id (e.g. `"4"`, `"35"`).

    Returns
    -------
    - `tuple`: An `(r, g, b)` triple in `[0, 1]`.
    """
    import sys

    sys.path.insert(0, os.path.join(clik_root(), "Code"))
    from model.core_util import id_color_dict
    import matplotlib.colors as mc

    return mc.to_rgb(id_color_dict.get(landmark_id, "black"))


def load_clik_landmarks(
    output_root,
    pid,
    frame="center",
    converted_root=None,
    include_centroid=False,
    jaw=None,
):
    """
    Load the landmarks CLIK detected for one patient.

    Parameters
    ----------
    - `output_root (str)`: CLIK output folder (holds `<pid>/landmarks/*.json`).
    - `pid (str)`: Patient id.
    - `frame (str, optional)`: `"center"` (as saved) or `"original"` (adds the
      converter offset back, matching the raw IOS scan). Default `"center"`.
    - `converted_root (str, optional)`: Required when `frame="original"`, to read `center.json`.
    - `include_centroid (bool, optional)`: Keep landmark id `"0"` (tooth centroid). Default `False`.
    - `jaw (str, optional)`: Restrict to `"upper"`/`"lower"` teeth (by Universal id). Default `None` (all).

    Returns
    -------
    - `tuple`: `(points, ids)` where `points` is `(N, 3)` float and `ids` is a list of CLIK landmark ids.

    Raises
    ------
    - `ValueError`: If `frame="original"` but `converted_root` is not given.
    """
    keep = set(ARCH_IDS[jaw]) if jaw else None
    pts, ids = [], []
    for jf in sorted(glob.glob(os.path.join(output_root, pid, "landmarks", "*.json"))):
        uid = int(os.path.basename(jf).split(".")[0])
        if keep is not None and uid not in keep:
            continue
        for lid, coord in json.load(open(jf)).items():
            if lid == "0" and not include_centroid:
                continue
            pts.append(np.asarray(coord, float))
            ids.append(lid)
    pts = np.asarray(pts, float)
    if frame == "original":
        if converted_root is None:
            raise ValueError(
                "frame='original' needs converted_root to read center.json"
            )
        pts = pts + load_center(converted_root, pid)
    return pts, ids


def find_gt_files(gt_root, pid):
    """
    Locate the 3DTeethLand ground-truth landmark files for a patient.

    Parameters
    ----------
    - `gt_root (str)`: Root of the unzipped `osfstorage-archive`.
    - `pid (str)`: Patient id.

    Returns
    -------
    - `list`: Paths to the `<pid>_<jaw>__kpt.json` files found (may be empty).
    """
    found = []
    for subset in _GT_SUBSETS:
        for jaw in ("upper", "lower"):
            p = os.path.join(gt_root, subset, jaw, pid, f"{pid}_{jaw}__kpt.json")
            if os.path.exists(p):
                found.append(p)
    return found


def load_gt_landmarks(gt_root, pid, frame="original", center=None, jaw=None):
    """
    Load 3DTeethLand ground-truth landmarks for one patient.

    Parameters
    ----------
    - `gt_root (str)`: Root of the unzipped `osfstorage-archive`.
    - `pid (str)`: Patient id.
    - `frame (str, optional)`: `"original"` (raw scan coords) or `"center"`
      (subtracts `center`, matching converted/CLIK frame). Default `"original"`.
    - `center (np.ndarray, optional)`: Required when `frame="center"`.
    - `jaw (str, optional)`: Restrict to `"upper"`/`"lower"`. Default `None` (both arches).

    Returns
    -------
    - `tuple` or `None`: `(points (M,3), classes list)`, or `None` if no GT exists for this patient.

    Raises
    ------
    - `ValueError`: If `frame="center"` but `center` is not given.
    """
    files = find_gt_files(gt_root, pid)
    if jaw:
        files = [f for f in files if f.endswith(f"{pid}_{jaw}__kpt.json")]
    if not files:
        return None
    pts, cls = [], []
    for f in files:
        for o in json.load(open(f))["objects"]:
            pts.append(np.asarray(o["coord"], float))
            cls.append(o["class"])
    pts = np.asarray(pts, float)
    if frame == "center":
        if center is None:
            raise ValueError("frame='center' needs the center offset")
        pts = pts - center
    return pts, cls


def load_ios_mesh(src_root, pid):
    """
    Load and merge the upper + lower raw IOS meshes for one patient.

    Parameters
    ----------
    - `src_root (str)`: Folder with the Teeth3DS patient subfolders (e.g. `Downloads/tmp`).
    - `pid (str)`: Patient id.

    Returns
    -------
    - `trimesh.Trimesh`: The merged IOS surface in original scan coordinates.

    Notes
    -----
    - Both jaws are concatenated as-is (Teeth3DS scans the arches separately and
      they are not co-registered into occlusion); this is fine for visualisation.
    """
    parts = []
    for jaw in ("upper", "lower"):
        obj = os.path.join(src_root, pid, f"{pid}_{jaw}.obj")
        if os.path.exists(obj):
            parts.append(load_mesh(obj, process=False))
    return trimesh.util.concatenate(parts)
