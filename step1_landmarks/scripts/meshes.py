"""
meshes
======
A single, typed entry point for reading mesh files.

`trimesh.load` may return a `Scene`, a `PointCloud` or a `Trimesh` depending on
the file, so callers that immediately reach for `.vertices` are making an
assumption the type checker cannot verify, and that occasionally fails at
runtime on files holding more than one object. This wrapper states the
assumption once and enforces it.

Functions
---------
- `load_mesh(path, process)`: Read a file as one `trimesh.Trimesh`.

Example
-------
```python
from meshes import load_mesh
crown = load_mesh('0001/initial/8.stl', process=False)
print(crown.vertices.shape)
```

Notes
-----
- `process=False` preserves the file's vertex order and duplicates, which matters
  whenever two meshes must correspond vertex-to-vertex (as `ori` and `final` do);
  `process=True` merges duplicated vertices, which is what CLIK itself does.
"""
import trimesh


def load_mesh(path, process=True):
    """
    Read a mesh file as a single `Trimesh`.

    Parameters
    ----------
    - `path (str)`: File to read (STL, PLY, OBJ, ...).
    - `process (bool, optional)`: Merge duplicated vertices. Default `True`; pass
      `False` to keep the file's exact vertex order.

    Returns
    -------
    - `trimesh.Trimesh`: The mesh, with scenes concatenated into one body.

    Raises
    ------
    - `TypeError`: If the file does not yield a triangular mesh (e.g. a point cloud).

    Example
    -------
    ```python
    crown = load_mesh('8.stl', process=False)
    ```
    """
    mesh = trimesh.load(path, process=process, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f'{path} non contiene una mesh triangolare ({type(mesh).__name__})')
    return mesh
