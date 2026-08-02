"""
visualize
=========
Publication-quality renderings of a single IOS arch with landmarks drawn as solid coloured spheres, in the style of the 3DTeethLand figure: an opaque, properly-occluded grey surface seen from an occlusal (top-down) view. Uses properly-occluded grey surface seen from an occlusal (top-down) view. Uses pyvista/VTK offscreen (works headless on macOS); no display required.

Functions
---------
- `occlusal_camera(plotter, vertices, points)`: Aim the camera straight down the occlusal axis.
- `render_arch(mesh, points, colors, out_png, title, sphere_radius)`: Render one arch + landmarks.

Example
-------
```python
from visualize import render_arch
render_arch(arch_mesh, gt_points, gt_colors, "ZKJEPFDD_lower_GT.png", "ZKJEPFDD lower - GT")
```

Notes
-----
- Render ONE arch at a time. Teeth3DS upper/lower are not co-registered, so rendering them together makes the scan look transparent and unreadable.
- Mesh and landmarks must share the same coordinate frame (use the raw scan frame: GT as-is, CLIK with `frame="original"`).
"""

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True


def occlusal_camera(plotter, vertices, orient_points, anterior=None, tilt=28.0):
    """
    Aim the camera to reproduce the 3DTeethLand occlusal view.

    Parameters
    ----------
    - `plotter (pyvista.Plotter)`: The plotter to configure.
    - `vertices (np.ndarray)`: `(V, 3)` mesh vertices (defines the scene bounds).
    - `orient_points (np.ndarray)`: `(N, 3)` reference points on the crowns (e.g. CLIK
      landmarks) used to find the occlusal side; pass the SAME set for GT and CLIK
      so both images share one orientation.
    - `anterior (np.ndarray, optional)`: Unit vector toward the incisors; placed at the
      bottom of the image. If `None`, falls back to the longest in-plane axis.
    - `tilt (float, optional)`: Forward tilt in degrees (shows the labial walls + base). Default `28`.

    Returns
    -------
    - `None`

    Notes
    -----
    - Occlusal axis = axis-snapped vector from the mesh centroid toward the crown
      points (base -> crowns). The arch opens upward, incisors at the bottom.
    """
    ctr = vertices.mean(0)
    op = np.asarray(orient_points)
    d = (op.mean(0) - ctr) if len(op) else np.array([0.0, 0.0, 1.0])
    k = int(np.argmax(np.abs(d)))
    n = np.zeros(3)
    n[k] = np.sign(d[k]) or 1.0  # occlusal axis (up out of crowns)

    if anterior is not None:
        a = np.asarray(anterior, float) - n * (np.asarray(anterior, float) @ n)
        a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else None
    else:
        a = None
    if a is None:
        spread = np.ptp(vertices, 0).astype(float)
        spread[k] = -1.0
        a = np.zeros(3)
        a[int(np.argmax(spread))] = 1.0

    size = np.ptp(vertices, 0).max()
    th = np.radians(tilt)
    cam_dir = n * np.cos(th) + a * np.sin(th)  # above + tilted toward the front
    up = -a - cam_dir * (cam_dir @ (-a))  # posterior(up) -> incisors at the bottom
    up = up / np.linalg.norm(up)

    plotter.camera_position = [tuple(ctr + cam_dir * size * 2.4), tuple(ctr), tuple(up)]
    plotter.reset_camera()
    plotter.camera.zoom(1.25)


def render_arch(
    mesh,
    points,
    colors,
    out_png,
    title="",
    orient_points=None,
    anterior=None,
    tilt=28.0,
    sphere_radius=0.8,
    bg="white",
    size=(1200, 1150),
):
    """
    Render one IOS arch as a solid grey surface with coloured landmark spheres.

    Parameters
    ----------
    - `mesh (trimesh.Trimesh)`: One arch (upper OR lower) in raw scan coordinates.
    - `points (np.ndarray)`: `(N, 3)` landmark coordinates in the same frame as `mesh`.
    - `colors (list)`: Per-landmark colours (names or RGB triples).
    - `out_png (str)`: Output PNG path.
    - `title (str, optional)`: Caption drawn at the top-left.
    - `orient_points (np.ndarray, optional)`: Points used to fix the camera; pass the same set for GT and CLIK so the two images are identically oriented. Defaults to `points`.
    - `anterior (np.ndarray, optional)`: Unit vector toward the incisors (bottom of image).
    - `tilt (float, optional)`: Forward tilt in degrees. Default `28`.
    - `sphere_radius (float, optional)`: Landmark sphere radius in mm. Default `0.8`.
    - `bg (str, optional)`: Background colour. Default `"white"`.
    - `size (tuple, optional)`: Image size in pixels. Default `(1200, 1150)`.

    Returns
    -------
    - `str`: `out_png`.
    """
    faces = (
        np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces])
        .astype(np.int64)
        .ravel()
    )
    surf = pv.PolyData(mesh.vertices, faces)
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background(bg)
    pl.add_mesh(
        surf,
        color=(0.58, 0.58, 0.58),
        smooth_shading=True,
        specular=0.2,
        specular_power=10,
    )
    pts = np.asarray(points)

    for p, c in zip(pts, colors):
        pl.add_mesh(pv.Sphere(radius=sphere_radius, center=p), color=c)

    occlusal_camera(
        pl,
        mesh.vertices,
        pts if orient_points is None else orient_points,
        anterior=anterior,
        tilt=tilt,
    )
    if title:
        pl.add_text(title, font_size=11, color="black")

    pl.screenshot(out_png)
    pl.close()

    return out_png
