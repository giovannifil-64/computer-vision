"""
prepost_to_clik
===============
Convert the PrePostOrthodontic dataset (Wang et al. 2024) into the per-tooth
folder layout CLIK's `infer_crown.py` expects. Each subject provides a
pre-treatment (`ori`) and a post-treatment (`final`) state; the pre-treatment
teeth are CLIK's input, the post-treatment teeth are the ground truth used to
score the predicted alignment.

Functions
---------
- `extract_teeth(stl_path, seg_path)`: Cut one arch STL into per-tooth crown meshes (Universal ids).
- `load_subject(subject_dir, stage)`: Extract both arches of one subject at one stage.
- `convert_subject(subject_dir, out_root)`: Write the CLIK layout for `ori` and `final`.
- `main()`: CLI entry point converting a list (or all) of subjects.

Example
-------
```python
from prepost_to_clik import convert_subject
convert_subject("PrePostOrthodontic/Orthodontic_dental_dataset/0001", "Data_prepost")
# -> Data_prepost/0001/initial/<id>.stl   (pre-treatment, CLIK input)
#    Data_prepost/0001/final/<id>.stl     (post-treatment, ground truth)
#    Data_prepost/0001/center.json
```

Notes
-----
- The segmentation JSON stores per-tooth vertex *coordinates*, not face indices,
  so vertices are matched back onto the arch STL (they agree to float32
  precision) to recover real meshes with faces, which CLIK needs for normals.
- Both stages are recentred with the SAME offset (computed on `ori`), otherwise
  the pre -> post tooth movement, i.e. exactly what we want to measure, would be
  destroyed. Teeth extracted during treatment simply do not appear in `final`.
"""
import os
import json
import glob
import argparse
import numpy as np

from meshes import load_mesh
from scipy.spatial import KDTree

# FDI (ISO-3950) -> Universal (1..32), the numbering CLIK expects.
FDI2U = {
    18: 1, 17: 2, 16: 3, 15: 4, 14: 5, 13: 6, 12: 7, 11: 8,          # upper right
    21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,   # upper left
    38: 17, 37: 18, 36: 19, 35: 20, 34: 21, 33: 22, 32: 23, 31: 24,  # lower left
    41: 25, 42: 26, 43: 27, 44: 28, 45: 29, 46: 30, 47: 31, 48: 32,  # lower right
}

STAGES = {'ori': 'Ori', 'final': 'Final'}

# CLIK's canonical jaw frame, measured on its own sample data:
#   anterior (incisors - molars) = -x,  patient right = +y,  upper arch = +z
# CLIK's diffusion learned arch shape in this frame, so a differently-oriented
# dentition would be out of distribution. This dataset is NOT consistently
# oriented (subjects differ by whole 180 deg rotations), so the frame is
# estimated per subject from the anatomy instead of using a fixed matrix.
CLIK_ANTERIOR = np.array([-1.0, 0.0, 0.0])
CLIK_RIGHT = np.array([0.0, 1.0, 0.0])
CLIK_UP = np.array([0.0, 0.0, 1.0])

UPPER_IDS = set(range(1, 17))
INCISOR_IDS = {7, 8, 9, 10, 23, 24, 25, 26}
MOLAR_IDS = {2, 3, 14, 15, 18, 19, 30, 31}
RIGHT_IDS = set(range(1, 9)) | set(range(25, 33))    # upper right + lower right


def estimate_frame(teeth):
    """
    Estimate the rotation bringing one dentition into CLIK's canonical frame.

    Parameters
    ----------
    - `teeth (dict)`: `{universal_id: trimesh.Trimesh}` for the whole dentition.

    Returns
    -------
    - `np.ndarray` or `None`: A `(3, 3)` rotation with `det = +1`, or `None` if the
      anatomy needed to define the axes is missing.

    Notes
    -----
    - Axes come from the anatomy itself: anterior from incisors minus molars, up
      from the upper arch minus the lower arch, right from right teeth minus left
      teeth. They are orthonormalised (Gram-Schmidt) and mapped onto CLIK's axes,
      which makes the conversion robust to however the raw scan was oriented.
    """
    c = {t: m.vertices.mean(0) for t, m in teeth.items()}

    def mean_of(ids):
        pts = [c[t] for t in ids if t in c]
        return np.mean(pts, 0) if pts else None

    inc, mol = mean_of(INCISOR_IDS), mean_of(MOLAR_IDS)
    up_a, low_a = mean_of(UPPER_IDS), mean_of(set(c) - UPPER_IDS)
    right, left = mean_of(RIGHT_IDS), mean_of(set(c) - RIGHT_IDS)
    if any(v is None for v in (inc, mol, up_a, low_a, right, left)):
        return None

    norm = lambda v: v / np.linalg.norm(v)
    u = norm(up_a - low_a)                       # occlusal "up"
    a = norm((inc - mol) - ((inc - mol) @ u) * u)  # anterior, orthogonal to up
    r = right - left                             # patient right, Gram-Schmidt vs u and a
    r = norm(r - (r @ u) * u - (r @ a) * a)
    S = np.column_stack([a, r, u])                       # subject axes
    T = np.column_stack([CLIK_ANTERIOR, CLIK_RIGHT, CLIK_UP])  # CLIK axes
    R = T @ S.T
    if np.linalg.det(R) < 0:                             # never mirror the anatomy
        return None
    return R


def extract_teeth(stl_path, seg_path, min_faces=50, tol=0.01):
    """
    Cut one arch STL into per-tooth crown meshes using the segmentation JSON.

    Parameters
    ----------
    - `stl_path (str)`: Arch mesh, e.g. `L_Ori.stl`.
    - `seg_path (str)`: Matching JSON with `segmentation[<FDI>]['vertices']`.
    - `min_faces (int, optional)`: Drop teeth with fewer faces than this. Default `50`.
    - `tol (float, optional)`: Max distance (mm) accepted when matching a segmentation
      vertex to an STL vertex. Default `0.01`.

    Returns
    -------
    - `dict`: `{universal_id (int): trimesh.Trimesh}` for this arch.

    Raises
    ------
    - `FileNotFoundError`: If either file is missing.

    Notes
    -----
    - A face belongs to a tooth when at least 2 of its 3 vertices carry that
      tooth's label, which is robust to the boundaries between adjacent teeth.
    """
    mesh = load_mesh(stl_path, process=True)     # merge duplicated STL vertices
    seg = json.load(open(seg_path))['segmentation']

    tree = KDTree(mesh.vertices)
    labels = np.zeros(len(mesh.vertices), dtype=int)   # 0 = unlabelled (gingiva)
    for fdi_str, payload in seg.items():
        fdi = int(fdi_str)
        if fdi not in FDI2U:
            continue
        pts = np.asarray(payload['vertices'], dtype=float)
        dist, idx = tree.query(pts)
        labels[idx[dist <= tol]] = fdi

    face_lab = labels[mesh.faces]
    teeth = {}
    for fdi in np.unique(labels):
        if fdi == 0:
            continue
        mask = (face_lab == fdi).sum(axis=1) >= 2
        if mask.sum() < min_faces:
            continue
        teeth[FDI2U[int(fdi)]] = mesh.submesh([mask], append=True)
    return teeth


def load_subject(subject_dir, stage):
    """
    Extract the per-tooth meshes of both arches for one subject at one stage.

    Parameters
    ----------
    - `subject_dir (str)`: Subject folder, e.g. `.../Orthodontic_dental_dataset/0001`.
    - `stage (str)`: `"ori"` (pre-treatment) or `"final"` (post-treatment).

    Returns
    -------
    - `dict`: `{universal_id (int): trimesh.Trimesh}` merging lower and upper arch.
    """
    tag = STAGES[stage]
    teeth = {}
    for arch in ('L', 'U'):
        stl = os.path.join(subject_dir, stage, f'{arch}_{tag}.stl')
        js = os.path.join(subject_dir, stage, f'{arch}_{tag}.json')
        if os.path.exists(stl) and os.path.exists(js):
            teeth.update(extract_teeth(stl, js))
    return teeth


def convert_subject(subject_dir, out_root):
    """
    Convert one subject (both stages) into the CLIK folder layout.

    Parameters
    ----------
    - `subject_dir (str)`: Subject folder in the raw dataset.
    - `out_root (str)`: Output root; writes `<out_root>/<id>/{initial,final}/`.

    Returns
    -------
    - `dict` or `None`: `{'id', 'n_ori', 'n_final'}`, or `None` if extraction failed.

    Notes
    -----
    - `initial/` holds the pre-treatment teeth (CLIK input, the name `initial` is
      what `infer_crown.py` looks for) and `final/` the post-treatment ground truth.
      A single centering offset, computed on `ori`, is applied to both stages and
      saved in `center.json`.
    """
    sid = os.path.basename(subject_dir.rstrip('/'))
    ori = load_subject(subject_dir, 'ori')
    if not ori:
        print(f'  [skip] {sid}: no teeth extracted from ori')
        return None
    final = load_subject(subject_dir, 'final')

    # One shared offset and rotation, estimated on `ori`, so the pre -> post
    # movement is preserved while both stages sit in CLIK's canonical frame.
    center = np.concatenate([t.vertices for t in ori.values()], 0).mean(0)
    R = estimate_frame(ori)
    if R is None:
        print(f'  [skip] {sid}: cannot estimate the jaw frame (missing teeth)')
        return None

    for stage, teeth in (('initial', ori), ('final', final)):
        out_dir = os.path.join(out_root, sid, stage)
        os.makedirs(out_dir, exist_ok=True)
        for uid, mesh in sorted(teeth.items()):
            m = mesh.copy()
            # same recentring AND same rotation for both stages, so the pre -> post
            # movement is preserved while the dentition sits in CLIK's frame
            m.vertices = (R @ (m.vertices - center).T).T
            m.export(os.path.join(out_dir, f'{uid}.stl'))

    json.dump({'center': center.tolist(), 'frame_R': R.tolist(),
               'teeth_ori': sorted(ori.keys()),
               'teeth_final': sorted(final.keys()),
               'extracted': sorted(set(ori) - set(final))},
              open(os.path.join(out_root, sid, 'center.json'), 'w'), indent=2)

    print(f'  [ok] {sid}: ori {len(ori)} teeth, final {len(final)} teeth'
          + (f", extracted {sorted(set(ori)-set(final))}" if set(ori) - set(final) else ''))
    return {'id': sid, 'n_ori': len(ori), 'n_final': len(final)}


def main():
    """
    CLI entry point: convert every subject found under `--in_root`.

    Example
    -------
    ```bash
    python prepost_to_clik.py -i ../../PrePostOrthodontic/Orthodontic_dental_dataset -o ../Data_prepost
    ```
    """
    ap = argparse.ArgumentParser(description="Convert PrePostOrthodontic subjects to CLIK layout.")
    ap.add_argument('-i', '--in_root', required=True, help='folder with subject subfolders')
    ap.add_argument('-o', '--out_root', required=True, help='output root (CLIK-style)')
    ap.add_argument('--ids', help='optional text file with subject ids (one per line)')
    ap.add_argument('--limit', type=int, default=0, help='max subjects (0 = all)')
    args = ap.parse_args()

    if args.ids:
        wanted = [l.strip() for l in open(args.ids) if l.strip()]
        dirs = [os.path.join(args.in_root, w) for w in wanted]
        dirs = [d for d in dirs if os.path.isdir(d)]
    else:
        dirs = sorted(d for d in glob.glob(os.path.join(args.in_root, '*')) if os.path.isdir(d))
    if args.limit:
        dirs = dirs[:args.limit]

    print(f'Converting {len(dirs)} subjects from {args.in_root}')
    ok = 0
    for d in dirs:
        if convert_subject(d, args.out_root):
            ok += 1
    print(f'Converted {ok}/{len(dirs)} subjects into {args.out_root}')


if __name__ == '__main__':
    main()
