"""
losses
======
The clinical constraints CLIK trains with, written as separate terms so that each
one's contribution can be turned on independently and measured.

The paper stacks three constraints on top of the plain reconstruction loss, one
per anatomical scale (Eq. 4-11): the dental arch as a whole must keep its shape,
neighbouring teeth must touch, and each tooth must keep its orientation. Their
weights in the paper are very uneven (0.1, 1e-3 and 0.01), so an ablation is the
only honest way to say what each is worth on this dataset.

Measured on 100 test subjects, fine-tuning for 100 epochs, each against the base
loss alone: **contact** is indistinguishable from noise; **individual** helps,
cutting the rotation error by 8% (p = 4e-4); **arch** hurts, badly, and lowering
its weight does not rescue it: at the published 0.1 the rotation error goes from
8.0 to 15.0 degrees with not one subject of a hundred improved, and at 0.001 it
still goes to 9.8 (+21%, p < 0.001). The harm shrinks with the weight but never
turns into a benefit, so this is not only a mis-scaled hyper-parameter.

A likely mechanism, worth testing before writing the term off for good: the loss
compares fourth-order polynomial coefficients, which are badly conditioned (small
changes in tooth position swing the high-order terms a long way), and it is
evaluated at a uniformly random diffusion timestep, so most of the time it is
fitting a polynomial to a dentition the model has barely begun to denoise. Fitting
noise produces meaningless coefficients and enormous gradients. Restricting the
term to low noise levels would test that directly.


None of the index bookkeeping is invented here. CLIK's `core_util` already ships
the landmark layout these constraints need: which landmarks belong to which tooth,
which two are the contact points, which pairs of contact points face each other,
and which landmarks make up the crown. Using those tables is what keeps the terms
faithful to the published formulation.

Functions
---------
- `tooth_validity(mask)`: Which teeth have a target, from the landmark mask.
- `tooth_centers(pts)`: Each tooth's morphological centre.
- `arch_loss(pred, target, valid)`: Dental-arch constraint, Eq. 4-5.
- `contact_loss(pred, valid)`: Inter-tooth contact constraint, Eq. 6-7.
- `individual_loss(pred, target, valid)`: Per-tooth orientation constraint, Eq. 10.
- `clinical_loss(pred, target, mask, weights)`: The enabled terms and their total.

Example
-------
```python
from losses import clinical_loss
extra, parts = clinical_loss(pred, target, mask, {'arch': 0.1, 'individual': 0.01})
loss = base + extra
```

Notes
-----
- Coordinates are in CLIK's normalised units throughout, as during training.
- The collision term of Eq. 8 is deliberately absent. It needs a precomputed
  signed-distance field per tooth, and the paper gives it an effective weight of
  1e-6 (`lambda_2 * lambda_4`), so it cannot plausibly move a fine-tuning run; it
  is still reported as an evaluation metric, where it does matter.
- Teeth missing from a scan, or extracted during treatment, are excluded from
  every term rather than contributing a spurious zero.
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_clik_root():
    """Locate the CLIK-Diffusion checkout (env `CLIK_ROOT`, else walk up)."""
    env = os.environ.get('CLIK_ROOT')
    if env:
        return env
    here = HERE
    while here != os.path.dirname(here):
        cand = os.path.join(here, 'CLIK-Diffusion')
        if os.path.isdir(os.path.join(cand, 'Code')):
            return cand
        here = os.path.dirname(here)
    raise RuntimeError('CLIK-Diffusion checkout not found; set CLIK_ROOT')


ROOT = _find_clik_root()
sys.path.insert(0, os.path.join(ROOT, 'Code'))

# CLIK's own modules only resolve once its Code folder is on the path above
import model.core_util as util

N_TEETH = 28
UPPER = slice(0, 14)
LOWER = slice(14, 28)
DEGREE = 4                              # fourth-order polynomial, as in the paper


def _membership():
    """
    A `(28, 256)` matrix whose rows average the landmarks of one tooth.

    Returns
    -------
    - `torch.Tensor`: Row `i` holds `1/n_i` on tooth `i`'s landmarks and zero elsewhere.
    """
    m = torch.zeros(N_TEETH, 256)
    for i in range(N_TEETH):
        m[i, i] = 1.0                                          # the centroid
        m[i, util.landmark_slices[i]:util.landmark_slices[i + 1]] = 1.0
    return m / m.sum(dim=1, keepdim=True)


def _contact_assignment():
    """
    The two contact landmarks of every tooth, told apart by which way they face.

    Returns
    -------
    - `tuple`: `(right, left)`, each a `(28,)` long tensor of landmark indices.

    Notes
    -----
    - Which of a tooth's two contact landmarks faces the neighbour with the lower
      id swaps at the midline, so the assignment is read off CLIK's own pairing
      tables rather than assumed. The first and last tooth of each arch have one
      unpaired contact, which takes the index left over.
    """
    right = [-1] * N_TEETH
    left = [-1] * N_TEETH
    # pair k joins two adjacent teeth: k and k+1 in the upper arch, shifted by one
    # in the lower arch because the two arches are not neighbours of each other
    for k in range(len(util.contact1_slices)):
        a = k if k < 13 else k + 1
        right[a] = util.contact1_slices[k]
        left[a + 1] = util.contact2_slices[k]
    for i in range(N_TEETH):
        pair = [util.landmark_slices[i + 1] - 2, util.landmark_slices[i + 1] - 1]
        if right[i] < 0:
            right[i] = pair[0] if pair[0] != left[i] else pair[1]
        if left[i] < 0:
            left[i] = pair[0] if pair[0] != right[i] else pair[1]
    return torch.tensor(right, dtype=torch.long), torch.tensor(left, dtype=torch.long)


_MEMBERSHIP = _membership()
_RIGHT, _LEFT = _contact_assignment()
_CROWN = [torch.tensor(v, dtype=torch.long) for v in util.vertical_slices]


def tooth_validity(mask):
    """
    Which teeth have a target, read off the per-landmark mask.

    Parameters
    ----------
    - `mask (torch.Tensor)`: `(B, 256)` boolean, True where a target exists.

    Returns
    -------
    - `torch.Tensor`: `(B, 28)` float, 1 for a tooth that is present.
    """
    return mask[:, :N_TEETH].float()


def tooth_centers(pts):
    """
    The morphological centre of each tooth: the mean of all its landmarks.

    Parameters
    ----------
    - `pts (torch.Tensor)`: `(B, 3, 256)` landmark coordinates.

    Returns
    -------
    - `torch.Tensor`: `(B, 3, 28)` centres.
    """
    return pts @ _MEMBERSHIP.to(pts.device).t()


def _fit_arch(centers, valid, ridge=1e-6):
    """
    Fit the fourth-order polynomial that describes one arch in the horizontal plane.

    Parameters
    ----------
    - `centers (torch.Tensor)`: `(B, 3, n)` tooth centres of one arch.
    - `valid (torch.Tensor)`: `(B, n)` weights, 1 for a tooth that is present.
    - `ridge (float, optional)`: Damping added to the normal equations. Default `1e-6`.

    Returns
    -------
    - `torch.Tensor`: `(B, 5)` coefficients, lowest order first.

    Notes
    -----
    - The arch is single-valued front-to-back as a function of the left-right
      coordinate, so that is the independent variable.
    - A subject can be missing enough teeth to leave the fit underdetermined, which
      makes the plain normal equations singular. The ridge term is the smallest
      change that keeps such a subject in the batch instead of crashing it; at 1e-6
      it is far below the residual of any fit that was already well posed.
    """
    across = centers[:, 1, :]                 # left-right
    along = centers[:, 0, :]                  # front-back
    vander = util.my_vander(across, DEGREE + 1)
    weighted = valid.unsqueeze(-1) * vander
    gram = weighted.transpose(-1, -2) @ weighted
    eye = torch.eye(DEGREE + 1, device=centers.device, dtype=gram.dtype)
    rhs = weighted.transpose(-1, -2) @ (valid * along).unsqueeze(-1)
    return torch.linalg.solve(gram + ridge * eye, rhs).squeeze(-1)


def arch_loss(pred, target, valid):
    """
    Keep the predicted dental arch the shape the real one has (Eq. 4-5).

    Parameters
    ----------
    - `pred (torch.Tensor)`: `(B, 3, 256)` predicted landmarks.
    - `target (torch.Tensor)`: `(B, 3, 256)` ground-truth landmarks.
    - `valid (torch.Tensor)`: `(B, 28)` tooth validity.

    Returns
    -------
    - `torch.Tensor`: Scalar loss, the two arches summed.

    Notes
    -----
    - The comparison is between the five polynomial coefficients, not between the
      curves point by point: that is what makes the term constrain the arch's size
      and shape rather than the position of any individual tooth.
    """
    cp, ct = tooth_centers(pred), tooth_centers(target)
    total = 0.0
    for arch in (UPPER, LOWER):
        a = _fit_arch(cp[:, :, arch], valid[:, arch])
        b = _fit_arch(ct[:, :, arch], valid[:, arch])
        total = total + ((a - b) ** 2).sum(dim=1).mean()
    return total


def contact_loss(pred, valid):
    """
    Pull neighbouring teeth into contact (Eq. 6-7).

    Parameters
    ----------
    - `pred (torch.Tensor)`: `(B, 3, 256)` predicted landmarks.
    - `valid (torch.Tensor)`: `(B, 28)` tooth validity.

    Returns
    -------
    - `torch.Tensor`: Scalar loss, the two arches summed.

    Notes
    -----
    - A pair counts only when both of its teeth are present, so a gap left by an
      extraction is not read as a contact that failed to close.
    """
    total = 0.0
    for base, arch in ((0, 0), (13, 14)):
        num, den = 0.0, 0.0
        for k in range(13):
            a = arch + k
            i1 = util.contact1_slices[base + k]
            i2 = util.contact2_slices[base + k]
            w = valid[:, a] * valid[:, a + 1]
            gap = ((pred[:, :, i1] - pred[:, :, i2]) ** 2).sum(dim=1)
            num = num + w * gap
            den = den + w
        total = total + (num / den.clamp(min=1.0)).mean()
    return total


def individual_loss(pred, target, valid):
    """
    Keep every tooth pointing the way it should (Eq. 10).

    Parameters
    ----------
    - `pred (torch.Tensor)`: `(B, 3, 256)` predicted landmarks.
    - `target (torch.Tensor)`: `(B, 3, 256)` ground-truth landmarks.
    - `valid (torch.Tensor)`: `(B, 28)` tooth validity.

    Returns
    -------
    - `torch.Tensor`: Scalar loss.

    Notes
    -----
    - Two directions per tooth: root-to-crown, as the mean offset of the crown
      landmarks from the centroid, and distal-mesial, as the line joining the two
      contact points. Both are compared by cosine similarity, so the term acts on
      orientation alone and leaves position to the reconstruction loss.
    - A direction of zero length has no orientation to compare, which happens when
      the detector puts a tooth's two contact landmarks on the same vertex. Such a
      tooth is dropped from that term; scoring it as maximally wrong would put a
      constant penalty in the loss that no amount of training could remove.
    """
    right = _RIGHT.to(pred.device)
    left = _LEFT.to(pred.device)

    def directions(pts):
        crown = torch.stack([(pts[:, :, _CROWN[i].to(pts.device)]
                              - pts[:, :, i:i + 1]).mean(dim=2)
                             for i in range(N_TEETH)], dim=2)
        mesial = pts[:, :, right] - pts[:, :, left]
        return crown, mesial

    cp, mp = directions(pred)
    ct, mt = directions(target)
    total = 0.0
    for a, b in ((cp, ct), (mp, mt)):
        usable = valid * (a.norm(dim=1) > 1e-8) * (b.norm(dim=1) > 1e-8)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1, eps=1e-8)
        total = total + ((1.0 - cos) * usable).sum() / usable.sum().clamp(min=1.0)
    return total


def clinical_loss(pred, target, mask, weights):
    """
    The enabled clinical terms, weighted and summed.

    Parameters
    ----------
    - `pred (torch.Tensor)`: `(B, 3, 256)` predicted landmarks.
    - `target (torch.Tensor)`: `(B, 3, 256)` ground-truth landmarks.
    - `mask (torch.Tensor)`: `(B, 256)` boolean landmark mask.
    - `weights (dict)`: Any of `arch`, `contact`, `individual`; a term with weight
      zero or absent is not computed at all.

    Returns
    -------
    - `tuple`: `(total, parts)` where `parts` maps each enabled term to its own
      unweighted value, for logging the ablation.
    """
    valid = tooth_validity(mask)
    terms = {'arch': lambda: arch_loss(pred, target, valid),
             'contact': lambda: contact_loss(pred, valid),
             'individual': lambda: individual_loss(pred, target, valid)}

    total = torch.zeros((), device=pred.device)
    parts = {}
    for name, fn in terms.items():
        w = weights.get(name, 0.0)
        if w:
            value = fn()
            parts[name] = float(value.detach())
            total = total + w * value
    return total, parts
