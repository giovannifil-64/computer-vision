"""
stage_b_train
=============
Stage B of the fine-tuning pipeline: fine-tune CLIK's diffusion model on the
PrePostOrthodontic training split, starting from the released weights.

The model predicts the clean target landmarks directly rather than the noise, as
the paper does, so the base loss is a masked mean squared error between the
predicted and the true post-treatment landmarks. The mask matters: teeth missing
from the scan, and teeth extracted during treatment, have an input but no target,
and must not contribute to the gradient.

On top of that base loss sit the paper's clinical constraints, each behind its own
weight so that a run can enable any subset. Running them one at a time was the
point, and it paid: only `--individual` helps here, `--contact` does nothing
measurable, and `--arch` at the published weight wrecks the model. `--clinical
paper` restores the published combination in one go and is kept for reproduction
only: it is not the configuration to build on. The numbers, and why the published
weights do not transfer to fine-tuning, are in `losses.py`.

Functions
---------
- `masked_mse(pred, target, mask)`: Mean squared error over the valid landmarks only.
- `training_step(net, batch, device, weights)`: One noised forward pass and its loss.
- `validation_loss(net, loader, device, weights)`: Mean loss over the held-out subjects.
- `main()`: Fine-tune, checkpointing the best model by validation loss.

Example
-------
```bash
python stage_b_train.py --tensors ../data/train_tensors --out ../output/base_loss
python stage_b_train.py --tensors ../data/train_tensors --out ../output/arch --arch 0.1
python stage_b_train.py --tensors ../data/train_tensors --out ../output/paper --clinical paper
```

Notes
-----
- Coordinates are in CLIK's normalised units, so the loss is not in millimetres;
  the millimetre figures come from the evaluation on the test split.
- The run is resumable: the last checkpoint is written every epoch and reloaded
  automatically if training is interrupted.
- The log records each clinical term separately, so a run shows what every
  constraint was worth even when several are enabled together.
"""
import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


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
from s2_LandmarkDiffusion import load_diffusion
from dataset import LandmarkPairs, split_subjects
from losses import clinical_loss

CKPT = os.path.join(ROOT, 'Code', 'checkpoint', 'diffusion-e20000.pth')
TERMS = ('arch', 'contact', 'individual')
PAPER_WEIGHTS = {'arch': 0.1, 'contact': 1e-3, 'individual': 0.01}   # lambda 1..3


def masked_mse(pred, target, mask):
    """
    Mean squared error restricted to the landmarks that have a target.

    Parameters
    ----------
    - `pred (torch.Tensor)`: `(B, 3, 256)` predicted landmarks.
    - `target (torch.Tensor)`: `(B, 3, 256)` ground-truth landmarks.
    - `mask (torch.Tensor)`: `(B, 256)` boolean, True where a target exists.

    Returns
    -------
    - `torch.Tensor`: Scalar loss; falls back to zero if a batch has no valid landmark.
    """
    m = mask.unsqueeze(1).expand_as(pred).float()
    denom = m.sum().clamp(min=1.0)
    return (((pred - target) ** 2) * m).sum() / denom


def noise_level(net, b, device, continuous):
    """
    Draw the noise level a training batch is conditioned on.

    Parameters
    ----------
    - `net`: The diffusion network, for its `gammas` schedule.
    - `b (int)`: Batch size.
    - `device (torch.device)`: Where to build the tensor.
    - `continuous (bool)`: Draw anywhere inside a step rather than exactly on it.

    Returns
    -------
    - `torch.Tensor`: `(b, 1)` noise levels.

    Notes
    -----
    - The released code has no training loop, so which of these two the authors
      used cannot be read off. The discrete one picks a step and takes its gamma;
      the continuous one picks a step and then a level anywhere between it and the
      previous, which is what Palette does, and this architecture is a Palette
      derivative down to the names of its methods. The difference matters because
      the discrete version only ever shows the network 2000 distinct noise levels,
      while sampling visits every one of them in turn.
    """
    if not continuous:
        t = torch.randint(0, net.num_timesteps, (b,), device=device)
        return net.gammas[t].view(b, 1)
    t = torch.randint(1, net.num_timesteps, (b,), device=device)
    low, high = net.gammas[t - 1], net.gammas[t]
    return (low + (high - low) * torch.rand(b, device=device)).view(b, 1)


def training_step(net, batch, device, weights=None, continuous=False):
    """
    Noise the target, ask the network to recover it, and score the attempt.

    Parameters
    ----------
    - `net`: The diffusion network.
    - `batch (tuple)`: `(cond, desc, target, mask)` as produced by the dataset.
    - `device (torch.device)`: Where to run.
    - `weights (dict, optional)`: Clinical term weights; omit for the base loss alone.
    - `continuous (bool, optional)`: Sample the noise level continuously. Default `False`.

    Returns
    -------
    - `tuple`: `(loss, parts)` where `parts` holds each enabled clinical term's own
      unweighted value, empty when none are enabled.

    Notes
    -----
    - The network is conditioned on the noise level exactly as at inference time,
      and predicts the clean landmarks directly. That is what makes the clinical
      constraints applicable during training at all: they need a dentition to look
      at, and here every step produces one.
    """
    cond, desc, target, mask = (x.to(device) for x in batch)
    b = cond.shape[0]
    gamma = noise_level(net, b, device, continuous)
    noise = torch.randn_like(target)
    g = gamma.view(b, 1, 1)
    noisy = g.sqrt() * target + (1.0 - g).sqrt() * noise
    pred = net.denoise_fn(torch.cat([cond, noisy], dim=1), gamma, desc)

    loss = masked_mse(pred, target, mask)
    parts = {}
    if weights:
        extra, parts = clinical_loss(pred, target, mask, weights)
        loss = loss + extra
    return loss, parts


@torch.no_grad()
def validation_loss(net, loader, device, weights=None, repeats=4, continuous=False):
    """
    Mean loss over the held-out subjects, both as trained and on the base term alone.

    Parameters
    ----------
    - `net`: The diffusion network.
    - `loader (DataLoader)`: Validation loader.
    - `device (torch.device)`: Where to run.
    - `weights (dict, optional)`: Clinical term weights, matching training.
    - `repeats (int, optional)`: Passes over the set, averaging different noise
      levels so the figure is less jumpy. Default `4`.
    - `continuous (bool, optional)`: Match the training noise sampling. Default `False`.

    Returns
    -------
    - `tuple`: `(total, base)` averaged losses.

    Notes
    -----
    - The base figure is reported separately because the clinical terms are orders
      of magnitude larger than the reconstruction error, which would otherwise make
      two runs with different constraints impossible to compare: their totals would
      not be the same quantity.
    """
    net.train(False)
    total, base, n = 0.0, 0.0, 0
    for _ in range(repeats):
        for batch in loader:
            total += training_step(net, batch, device, weights, continuous)[0].item()
            base += training_step(net, batch, device, None, continuous)[0].item()
            n += 1
    net.train(True)
    return total / max(n, 1), base / max(n, 1)


def main():
    """Fine-tune the diffusion model, keeping the best checkpoint by validation loss."""
    ap = argparse.ArgumentParser(description="Fine-tune CLIK's diffusion model (base loss).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--tensors', required=True, help='folder with the prepared .npz files')
    ap.add_argument('--out', required=True, help='folder for checkpoints and the log')
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--batch-size', type=int, default=48)      # as in the paper
    ap.add_argument('--lr', type=float, default=5e-5)          # as in the paper
    ap.add_argument('--n-val', type=int, default=100)
    ap.add_argument('--val-every', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gamma', choices=['discrete', 'continuous'], default='discrete',
                    help='how the training noise level is drawn; see noise_level()')
    for term in TERMS:
        ap.add_argument(f'--{term}', type=float, default=0.0,
                        help=f'weight of the {term} constraint (0 disables it)')
    ap.add_argument('--clinical', choices=['paper'],
                    help="shorthand for the paper's weights: " +
                         ', '.join(f'{k} {v:g}' for k, v in PAPER_WEIGHTS.items()))
    args = ap.parse_args()

    weights = dict(PAPER_WEIGHTS) if args.clinical == 'paper' else \
        {t: getattr(args, t) for t in TERMS if getattr(args, t)}

    device = util.DEVICE
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    files = sorted(glob.glob(os.path.join(args.tensors, '*.npz')))
    train_files, val_files = split_subjects(files, args.n_val, seed=args.seed)
    train_ds, val_ds = LandmarkPairs(train_files), LandmarkPairs(val_files)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size)
    json.dump({'train': train_ds.subject_ids, 'val': val_ds.subject_ids},
              open(os.path.join(args.out, 'split.json'), 'w'), indent=2)

    net = load_diffusion(CKPT)
    net.to(device)
    net.train(True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    last = os.path.join(args.out, 'last.pt')
    start_epoch, best = 0, float('inf')
    if os.path.exists(last):
        ck = torch.load(last, map_location=device)
        net.load_state_dict(ck['model']); opt.load_state_dict(ck['optim'])
        start_epoch, best = ck['epoch'] + 1, ck['best']
        print(f"Resuming from epoch {start_epoch} (best so far {best:.6f})")

    print(f'device {device} | {len(train_ds)} training, {len(val_ds)} validation | '
          f'batch {args.batch_size}, lr {args.lr}')
    print('clinical constraints: ' + (', '.join(f'{k} {v:g}' for k, v in weights.items())
                                 if weights else 'none (base loss only)'))
    print(f'noise level: {args.gamma}')
    log_path = os.path.join(args.out, 'log.csv')
    if not os.path.exists(log_path):
        open(log_path, 'w').write('epoch,train_loss,val_loss,val_base,seconds,'
                                  + ','.join(TERMS) + '\n')

    for epoch in range(start_epoch, args.epochs):
        t0, total, n = time.time(), 0.0, 0
        running = {t: 0.0 for t in TERMS}
        for batch in train_ld:
            loss, parts = training_step(net, batch, device, weights, args.gamma == 'continuous')
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
            for k, v in parts.items():
                running[k] += v
        train_loss = total / max(n, 1)
        means = {t: running[t] / max(n, 1) for t in TERMS}

        val_loss = val_base = float('nan')
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_loss, val_base = validation_loss(net, val_ld, device, weights,
                                                 continuous=args.gamma == 'continuous')
            # selection is on the base term so that checkpoints from runs with
            # different constraints are chosen by the same yardstick
            if val_base < best:
                best = val_base
                torch.save({'model': net.state_dict(), 'epoch': epoch,
                            'val_loss': val_loss, 'val_base': val_base,
                            'weights': weights}, os.path.join(args.out, 'best.pt'))

        dt = time.time() - t0
        open(log_path, 'a').write(f'{epoch},{train_loss:.6f},{val_loss:.6f},{val_base:.6f},'
                                  f'{dt:.1f},'
                                  + ','.join(f'{means[t]:.6f}' for t in TERMS) + '\n')
        torch.save({'model': net.state_dict(), 'optim': opt.state_dict(),
                    'epoch': epoch, 'best': best, 'weights': weights}, last)
        msg = f'  epoch {epoch + 1}/{args.epochs}  train {train_loss:.6f}'
        if not np.isnan(val_base):
            msg += f'  val {val_base:.6f}' + ('  <- best' if val_base == best else '')
        if weights:
            msg += '  [' + ' '.join(f'{k} {means[k]:.4f}' for k in weights) + ']'
        print(msg + f'  ({dt:.0f}s)', flush=True)

    print(f'\nDone. Best validation (base term): {best:.6f}')
    print(f'Checkpoint at {os.path.join(args.out, "best.pt")}')


if __name__ == '__main__':
    main()
