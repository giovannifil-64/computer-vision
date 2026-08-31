"""
hybrid
======
Produce 3DTeethLand-style landmarks (the 6 classes) from a CLIK run, by combining:
- Part A: relabel CLIK's detected landmarks into the 4 classes it covers (Mesial, Distal, Cusp, Inner), using a map learned from a patient that HAS ground truth (the map is per tooth-type, so it transfers to other patients);
- Part B: derive the 2 classes CLIK does not have (Outer, Facial) directly from the crown geometry.

Functions
---------
- `tooth_type(uid)`: Map a Universal tooth id to incisor/cuspid/premolar/molar.
- `load_clik_per_tooth(output_root, pid)`: CLIK landmarks grouped per tooth.
- `learn_class_map(converted_root, output_root, gt_root, gt_pid)`: Learn the CLIK-id -> class map.
- `geometric_outer_facial(mesh, arch_center, occlusal)`: Geometric Outer + Facial for one crown.
- `build_hybrid(converted_root, output_root, pid, class_map)`: 6-class hybrid landmarks for a patient.

Example
-------
```python
from hybrid import learn_class_map, build_hybrid
cmap = learn_class_map("Data_teeth3ds", "Output_teeth3ds", GT_ROOT, "ZKJEPFDD")
hyb  = build_hybrid("Data_teeth3ds", "Output_teeth3ds", "ZOUIF2W4", cmap)  # [(tid, class, coord_centered), ...]
```

Notes
-----
- All coordinates are in the converter's CENTERED frame; add `load_center(...)` to place them on the original IOS scan for rendering.
- The map is learned from one GT patient, so absolute accuracy on other patients is unverified; this is a visualisation/approximation, not validated ground truth.
"""

import os
import glob
import json
import numpy as np
from meshes import load_mesh
from collections import defaultdict, Counter

from common import load_gt_landmarks

_TYPES = {
    "incisor": [7, 8, 9, 10, 23, 24, 25, 26],
    "cuspid": [6, 11, 22, 27],
    "premolar": [4, 5, 12, 13, 20, 21, 28, 29],
    "molar": [2, 3, 14, 15, 18, 19, 30, 31],
}
_SINGLE = ["Mesial", "Distal", "InnerPoint"]  # one CLIK id each (Cusp -> a set)


def tooth_type(uid):
    """
    Map a Universal tooth id to its type.

    Parameters
    ----------
    - `uid (int)`: Universal tooth id (1..32).

    Returns
    -------
    - `str` or `None`: `"incisor"`, `"cuspid"`, `"premolar"`, `"molar"`, or `None` (3rd molars).
    """
    for t, ids in _TYPES.items():
        if uid in ids:
            return t
    return None


def load_clik_per_tooth(output_root, pid):
    """
    Load CLIK landmarks grouped per tooth (centered frame, centroid id `0` dropped).

    Parameters
    ----------
    - `output_root (str)`: CLIK output folder.
    - `pid (str)`: Patient id.

    Returns
    -------
    - `dict`: `{tooth_id (int): {clik_id (str): coord (np.ndarray)}}`.
    """
    out = {}
    for jf in glob.glob(os.path.join(output_root, pid, "landmarks", "*.json")):
        tid = int(os.path.basename(jf).split(".")[0])
        out[tid] = {
            l: np.asarray(c, float) for l, c in json.load(open(jf)).items() if l != "0"
        }
    return out


def _crowns(converted_root, pid):
    return {
        int(os.path.basename(f).split(".")[0]): load_mesh(f)
        for f in glob.glob(os.path.join(converted_root, pid, "initial", "*.stl"))
    }


def learn_class_map(converted_root, output_root, gt_root, gt_pid):
    """
    Learn the CLIK-id -> 3DTeethLand-class map from one ground-truth patient.

    Parameters
    ----------
    - `converted_root (str)`: Converter output root.
    - `output_root (str)`: CLIK output root.
    - `gt_root (str)`: Ground-truth root (`osfstorage-archive`).
    - `gt_pid (str)`: Patient id that has GT (e.g. `"ZKJEPFDD"`).

    Returns
    -------
    - `dict`: `{tooth_type: {class: set(clik_id)}}` for Mesial/Distal/Cusp/Inner.

    Notes
    -----
    - GT landmarks (no per-tooth label) are assigned to the nearest tooth. For each tooth type, single classes get the CLIK id with the smallest median distance (kept distinct); Cusp gets the set of ids that win >=2 GT cusps.
    """
    from common import load_center

    center = load_center(converted_root, gt_pid)
    clik = load_clik_per_tooth(output_root, gt_pid)
    crowns = _crowns(converted_root, gt_pid)
    ccent = {t: m.vertices.mean(0) for t, m in crowns.items()}
    gt_pts, gt_cls = load_gt_landmarks(gt_root, gt_pid, frame="center", center=center)

    tids = list(ccent)
    cents = np.array([ccent[t] for t in tids])
    gtt = defaultdict(list)
    for p, c in zip(gt_pts, gt_cls):
        gtt[tids[int(np.argmin(np.linalg.norm(cents - p, axis=1)))]].append((c, p))

    def med(typ, cls, cid):
        ds = [
            np.linalg.norm(clik[t][cid] - p)
            for t, gl in gtt.items()
            if tooth_type(t) == typ and cid in clik.get(t, {})
            for c, p in gl
            if c == cls
        ]
        return np.median(ds) if ds else np.inf

    cmap = defaultdict(dict)
    for typ in _TYPES:
        cand = sorted({i for t in clik for i in clik[t] if tooth_type(t) == typ})
        cv = Counter()
        for t, gl in gtt.items():
            if tooth_type(t) != typ or t not in clik:
                continue
            ids = list(clik[t])
            co = np.array([clik[t][i] for i in ids])
            for c, p in gl:
                if c == "Cusp":
                    cv[ids[int(np.argmin(np.linalg.norm(co - p, axis=1)))]] += 1
        cusp = {i for i, k in cv.items() if k >= 2}
        if cusp:
            cmap[typ]["Cusp"] = cusp
        used = set(cusp)
        for cls in _SINGLE:
            best = min(
                cand, key=lambda i: (med(typ, cls, i) if i not in used else np.inf)
            )
            if med(typ, cls, best) < np.inf:
                cmap[typ][cls] = {best}
                used.add(best)
    return cmap


def _occlusal_axis(crowns):
    allv = np.concatenate([m.vertices for m in crowns.values()])
    center = allv.mean(0)
    n = np.linalg.eigh(np.cov((allv - center).T))[1][
        :, 0
    ]  # smallest-variance axis = occlusal normal
    return center, n


def geometric_outer_facial(mesh, arch_center, occlusal):
    """
    Derive the Outer and Facial(-Axis) landmarks of one crown from its geometry.

    Parameters
    ----------
    - `mesh (trimesh.Trimesh)`: One crown mesh (centered frame).
    - `arch_center (np.ndarray)`: Centroid of the whole arch (for the buccal direction).
    - `occlusal (np.ndarray)`: Unit occlusal axis of the arch.

    Returns
    -------
    - `tuple`: `(outer, facial)` coordinates (np.ndarray, centered frame).

    Notes
    -----
    - Outer = buccal-most vertex on the gingival margin (the crown's open boundary loop).
    - Facial = centroid of the buccal-facing vertices, snapped to the surface.
    """
    c = mesh.vertices.mean(0)
    bd = c - arch_center
    bd = bd - bd @ occlusal * occlusal
    bd /= np.linalg.norm(bd)
    e = np.sort(mesh.edges, axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    bv = np.unique(u[cnt == 1])
    outer = mesh.vertices[bv[np.argmax((mesh.vertices[bv] - c) @ bd)]]
    bm = (mesh.vertex_normals @ bd) > 0.35
    fa = mesh.vertices[bm].mean(0) if bm.sum() > 5 else c
    fa = mesh.vertices[np.argmin(np.linalg.norm(mesh.vertices - fa, axis=1))]
    return outer, fa


def build_hybrid(converted_root, output_root, pid, class_map):
    """
    Build the 6-class hybrid landmarks for one patient.

    Parameters
    ----------
    - `converted_root (str)`: Converter output root.
    - `output_root (str)`: CLIK output root.
    - `pid (str)`: Patient id.
    - `class_map (dict)`: Output of `learn_class_map`.

    Returns
    -------
    - `list`: `[(tooth_id, class, coord)]` in the centered frame (4 classes from CLIK
      via the map + Outer/Facial from geometry).
    """
    clik = load_clik_per_tooth(output_root, pid)
    crowns = _crowns(converted_root, pid)
    arch_center, occlusal = _occlusal_axis(crowns)
    hyb = []
    for t, m in crowns.items():
        typ = tooth_type(t)
        if typ is None:
            continue
        for cls, ids in class_map.get(typ, {}).items():
            for i in ids:
                if i in clik.get(t, {}):
                    hyb.append((t, cls, clik[t][i]))
        o, f = geometric_outer_facial(m, arch_center, occlusal)
        hyb.append((t, "OuterPoint", o))
        hyb.append((t, "FacialPoint", f))
    return hyb
