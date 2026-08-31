"""
dataset
=======
Loading of the tensors produced by `prepare_training_data`, plus the train /
validation split. Each subject is one sample: the conditioning landmarks, the
dentition descriptor, the target landmarks and the mask saying which of the 256
landmarks actually have a target.

Functions
---------
- `split_subjects(files, n_val, seed)`: Deterministic train / validation split.

Classes
-------
- `LandmarkPairs`: Dataset over the prepared `.npz` files.

Example
-------
```python
from dataset import LandmarkPairs, split_subjects
train_files, val_files = split_subjects(sorted(glob.glob('data/train_tensors/*.npz')), n_val=100)
train = LandmarkPairs(train_files)
cond, desc, target, mask = train[0]
```

Notes
-----
- Tensors come out in the layout the network expects, i.e. `(channels, 256)`,
  which is the transpose of how they are stored.
- The split is drawn with a fixed seed so that training, resuming and evaluating
  always see the same validation subjects.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def split_subjects(files, n_val=100, seed=0):
    """
    Split the prepared files into a training and a validation set.

    Parameters
    ----------
    - `files (list)`: Paths to the `.npz` files.
    - `n_val (int, optional)`: How many subjects to hold out. Default `100`.
    - `seed (int, optional)`: RNG seed, fixed so the split is reproducible. Default `0`.

    Returns
    -------
    - `tuple`: `(train_files, val_files)`.

    Raises
    ------
    - `ValueError`: If `n_val` leaves no training data.
    """
    if n_val >= len(files):
        raise ValueError(f'n_val={n_val} non lascia dati di addestramento ({len(files)} file)')
    order = np.random.default_rng(seed).permutation(len(files))
    val_idx = set(order[:n_val].tolist())
    train = [f for i, f in enumerate(files) if i not in val_idx]
    val = [f for i, f in enumerate(files) if i in val_idx]
    return train, val


class LandmarkPairs(Dataset):
    """
    LandmarkPairs
    =============
    A dataset of pre-treatment / post-treatment landmark pairs, one sample per
    subject, read from the `.npz` files written by `prepare_training_data`.

    Methods
    -------
    - `__len__()`: Number of subjects.
    - `__getitem__(i)`: Conditioning, descriptor, target and mask for one subject.

    Attributes
    ----------
    - `files (list)`: The `.npz` paths backing the dataset.
    - `cache (bool)`: Whether samples are kept in memory after first read.

    Examples
    --------
    ```python
    ds = LandmarkPairs(files, cache=True)
    cond, desc, target, mask = ds[0]   # (5,256) (384,256) (3,256) (256,)
    ```
    """

    def __init__(self, files, cache=True):
        self.files = list(files)
        self.cache = cache
        self._mem = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if i in self._mem:
            return self._mem[i]
        d = np.load(self.files[i])
        sample = (
            torch.from_numpy(d['cond'].T).float(),
            torch.from_numpy(d['desc'].T).float(),
            torch.from_numpy(d['target'].T).float(),
            torch.from_numpy(d['mask']).bool(),
        )
        if self.cache:
            self._mem[i] = sample
        return sample

    @property
    def subject_ids(self):
        """Subject ids, in the same order as the samples."""
        return [os.path.basename(f)[:-4] for f in self.files]
