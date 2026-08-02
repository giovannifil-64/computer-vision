"""
evaluate
========
Quantitative comparison of CLIK-detected landmarks against 3DTeethLand
ground-truth landmarks, with a random-surface-point baseline so the numbers are
interpretable (a dense point cloud scores well by chance; the baseline measures
that chance level).

Functions
---------
- `nearest_distances(query, reference)`: For each query point, distance to the nearest reference point.
- `random_baseline(gt, surface, n_points, trials, seed)`: Expected GT->nearest distance for random points.
- `evaluate_patient(pid, ...)`: Full per-class metric table for one patient.

Example
-------
```python
from evaluate import evaluate_patient
report = evaluate_patient("ZKJEPFDD", converted_root="Data_teeth3ds",
                          output_root="Output_teeth3ds", gt_root=GT_ROOT, src_root=SRC)
print(report['overall'])      # {'clik_median': 0.66, 'random_median': 1.75, ...}
```

Notes
-----
- The core metric is *coverage / recall*: for each ground-truth landmark we find
  the nearest CLIK landmark (of any type) and report the distance. It answers
  "did CLIK place a landmark on this true anatomical point?" It is NOT a
  per-class identity match, because CLIK's landmark scheme differs from
  3DTeethLand's. The random baseline (same number of points, sampled on the same
  crowns) is the chance level for the same metric.
"""

import os
import glob
import numpy as np
import trimesh
from meshes import load_mesh
from scipy.spatial import cKDTree

from common import load_center, load_clik_landmarks, load_gt_landmarks, GT_CLASS_COLORS


def nearest_distances(query, reference):
    """
    For each query point, the Euclidean distance to the nearest reference point.

    Parameters
    ----------
    - `query (np.ndarray)`: `(N, 3)` points to measure from.
    - `reference (np.ndarray)`: `(M, 3)` candidate points.

    Returns
    -------
    - `np.ndarray`: `(N,)` nearest-neighbour distances in millimetres.
    """
    return cKDTree(reference).query(query)[0]


def random_baseline(gt, surface, n_points, trials=20, seed=0):
    """
    Chance-level GT->nearest distance using random points on the crown surface.

    Parameters
    ----------
    - `gt (np.ndarray)`: `(N, 3)` ground-truth points.
    - `surface (np.ndarray)`: `(V, 3)` crown-surface points to sample from.
    - `n_points (int)`: How many random points to draw per trial (match the CLIK count).
    - `trials (int, optional)`: Number of random trials to average. Default `20`.
    - `seed (int, optional)`: RNG seed. Default `0`.

    Returns
    -------
    - `float`: Mean over trials of the median GT->nearest-random distance (mm).
    """
    rng = np.random.default_rng(seed)
    meds = []
    for _ in range(trials):
        idx = rng.choice(len(surface), min(n_points, len(surface)), replace=False)
        meds.append(np.median(nearest_distances(gt, surface[idx])))
    return float(np.mean(meds))


def evaluate_patient(pid, converted_root, output_root, gt_root, src_root, trials=20):
    """
    Compute the full per-class metric table for one patient.

    Parameters
    ----------
    - `pid (str)`: Patient id.
    - `converted_root (str)`: Converter output (for `center.json`).
    - `output_root (str)`: CLIK output (for detected landmarks).
    - `gt_root (str)`: Root of `osfstorage-archive` (ground truth).
    - `src_root (str)`: Teeth3DS source folder (for the crown surface points).
    - `trials (int, optional)`: Random-baseline trials. Default `20`.

    Returns
    -------
    - `dict` or `None`: `{'pid', 'n_clik', 'n_gt', 'overall', 'per_class'}`, or
      `None` if no ground truth exists for this patient.

    Notes
    -----
    - All distances are computed in the *centered* frame (GT is shifted by the
      converter offset) so they are true physical millimetres.
    """
    center = load_center(converted_root, pid)
    clik, _ = load_clik_landmarks(output_root, pid, frame="center")
    gt = load_gt_landmarks(gt_root, pid, frame="center", center=center)
    if gt is None:
        return None
    gt_pts, gt_cls = gt

    # crown surface points (converted crowns are already centered)
    surface = np.concatenate(
        [
            load_mesh(f).vertices
            for f in glob.glob(os.path.join(converted_root, pid, "initial", "*.stl"))
        ],
        0,
    )

    d = nearest_distances(gt_pts, clik)
    per_class = {}
    for cl in GT_CLASS_COLORS:
        m = [i for i, c in enumerate(gt_cls) if c == cl]
        if not m:
            continue
        gt_c = gt_pts[m]
        clik_med = float(np.median(d[m]))
        base_med = random_baseline(gt_c, surface, len(clik), trials)
        per_class[cl] = {
            "n": len(m),
            "clik_median": clik_med,
            "random_median": base_med,
            "skill": skill_score(
                clik_med, base_med
            ),  # baseline-relative, self-calibrating
        }
    return {
        "pid": pid,
        "n_clik": len(clik),
        "n_gt": len(gt_pts),
        "overall": overall(d, gt_pts, surface, len(clik), trials),
        "per_class": per_class,
    }


def overall(d, gt_pts, surface, n_clik, trials):
    """
    Aggregate overall metrics (median, mean, baseline, skill) across all classes.

    Parameters
    ----------
    - `d (np.ndarray)`: GT->nearest-CLIK distances for every ground-truth point.
    - `gt_pts (np.ndarray)`: All ground-truth points `(M, 3)`.
    - `surface (np.ndarray)`: Crown-surface points for the random baseline.
    - `n_clik (int)`: Number of CLIK landmarks (size of the random sample).
    - `trials (int)`: Random-baseline trials.

    Returns
    -------
    - `dict`: `{'clik_median', 'clik_mean', 'random_median', 'skill'}`.
    """
    cm = float(np.median(d))
    bm = random_baseline(gt_pts, surface, n_clik, trials)
    return {
        "clik_median": cm,
        "clik_mean": float(np.mean(d)),
        "random_median": bm,
        "skill": skill_score(cm, bm),
    }


def skill_score(clik_median, random_median):
    """
    Baseline-relative skill score: a self-calibrating "dynamic median".

    Parameters
    ----------
    - `clik_median (float)`: CLIK's GT->nearest median distance (mm).
    - `random_median (float)`: The random-baseline median for the same points (mm).

    Returns
    -------
    - `float`: `1 - clik_median / random_median`. `0` = chance level (the base),
      `→1` = near-perfect, `<0` = worse than chance.

    Notes
    -----
    - This turns an absolute millimetre median into a unitless score whose zero
      point is each class's own random baseline, so classes (and patients) are
      directly comparable regardless of tooth size or point density.
    """
    return float(1.0 - clik_median / max(random_median, 1e-6))


def format_table(report):
    """
    Render an evaluation report as a readable text table.

    Parameters
    ----------
    - `report (dict)`: Output of `evaluate_patient`.

    Returns
    -------
    - `str`: A multi-line table (class, CLIK median, random baseline, verdict).
    """

    def verdict(skill):
        return (
            "good"
            if skill >= 0.33
            else ("~random" if skill >= -0.05 else "worse than random")
        )

    lines = [
        f"{report['pid']}: CLIK={report['n_clik']} landmarks vs GT={report['n_gt']}",
        f"  {'class':12s} {'CLIK(mm)':>9s} {'base(mm)':>9s} {'skill':>6s}  verdict",
    ]
    for cl, v in report["per_class"].items():
        lines.append(
            f"  {cl:12s} {v['clik_median']:9.2f} {v['random_median']:9.2f} "
            f"{v['skill']:6.2f}  {verdict(v['skill'])}"
        )
    o = report["overall"]
    lines.append(
        f"  {'OVERALL':12s} {o['clik_median']:9.2f} {o['random_median']:9.2f} "
        f"{o['skill']:6.2f}  {verdict(o['skill'])}"
    )
    return "\n".join(lines)
