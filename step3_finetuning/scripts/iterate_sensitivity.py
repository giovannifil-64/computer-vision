"""
iterate_sensitivity
===================
Measure how much the diffusion network's output actually depends on the noisy
landmarks it is asked to denoise.

A reverse diffusion step is a function of two things, the conditioning and the
current noisy iterate, and the whole point of running the chain is that each step
refines the previous one. That only works if the network reads the iterate. This
script tests whether it does, by holding the conditioning and the noise level
fixed, feeding several different iterates, and comparing how far the predictions
spread against how far the inputs that produced them spread.

The ratio between those two spreads is the number that matters. At high noise a
small ratio is expected even from a network that works as intended, because the
iterate is then almost pure noise and the best prediction of the clean landmarks
rests on the conditioning. At low noise the iterate is almost the answer, and a
network that reads it has to show a ratio close to one there; a ratio near zero at
the lowest noise levels means the network answered from the conditioning alone and
ignored the iterate, in which case the chain has nothing to refine and one step is
worth as many as you like. That consequence is measurable end to end with
`stage_c_predict.py --steps 1`, and this script is the direct measurement of the
cause behind it.

Functions
---------
- `sensitivity(net, cond, desc, target, level, draws, seed)`: Output spread over input spread at one noise level.
- `sweep(net, tensors, sids, levels, draws)`: The same measurement across noise levels.
- `main()`: Run the sweep for one checkpoint and print the table.

Example
-------
```bash
python iterate_sensitivity.py --ckpt ../output/exp_long/best.pt \\
    --tensors ../data/test_tensors --limit 8
```

Notes
-----
- The iterates are built the way training builds them, by noising the true target
  to the requested level, so the network is asked the question it was trained on
  rather than an artificial one.
- The levels are given as diffusion steps, 1999 being almost pure noise and 1 a
  dentition already close to its target, and the ratio is reported at each, so a
  network that consulted its input only near the end of the chain would still
  show up.
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stage_c_predict import load_network, load_inputs, ROOT      # noqa: E402

sys.path.insert(0, os.path.join(ROOT, 'Code'))
import model.core_util as util                                    # noqa: E402
from model.diffusion_network import extract                       # noqa: E402


@torch.no_grad()
def sensitivity(net, cond, desc, target, level, draws=8, seed=0):
    """
    Spread of the predictions over spread of the iterates, at one noise level.

    Parameters
    ----------
    - `net`: The diffusion network, already on the device and in inference mode.
    - `cond (np.ndarray)`: `(N, 5, 256)` conditioning landmarks and identifiers.
    - `desc (np.ndarray)`: `(N, 384, 256)` shape descriptors.
    - `target (np.ndarray)`: `(N, 3, 256)` true target landmarks, used to build
      iterates from the same distribution training used.
    - `level (int)`: Diffusion step, from `0` to `net.num_timesteps - 1`.
    - `draws (int, optional)`: How many different iterates to try. Default `8`.
    - `seed (int, optional)`: RNG seed, so the measurement repeats. Default `0`.

    Returns
    -------
    - `tuple`: `(ratio, in_spread, out_spread)` in normalised units, each spread
      being the standard deviation across the draws, averaged over coordinates.

    Notes
    -----
    - Everything except the iterate is held fixed, so any variation in the output
      is attributable to the iterate and to nothing else.
    """
    device = util.DEVICE
    c = torch.as_tensor(cond, dtype=torch.float32, device=device)
    d = torch.as_tensor(desc, dtype=torch.float32, device=device)
    y0 = torch.as_tensor(target, dtype=torch.float32, device=device)

    t = torch.full((c.shape[0],), int(level), device=device, dtype=torch.long)
    gamma = extract(net.gammas, t, x_shape=(1, 1)).to(device)
    g = gamma.view(-1, 1, 1)

    torch.manual_seed(seed)
    ins, outs = [], []
    for _ in range(draws):
        noise = torch.randn_like(y0)
        y_t = g.sqrt() * y0 + (1.0 - g).sqrt() * noise
        pred = net.denoise_fn(torch.cat([c, y_t], dim=1), gamma, d).clamp(-1.0, 1.0)
        ins.append(y_t)
        outs.append(pred)

    in_spread = torch.stack(ins).std(dim=0).mean().item()
    out_spread = torch.stack(outs).std(dim=0).mean().item()
    return out_spread / max(in_spread, 1e-12), in_spread, out_spread


def sweep(net, tensors, sids, levels, draws=8):
    """
    Run `sensitivity` at several noise levels on the same subjects.

    Parameters
    ----------
    - `net`: The diffusion network.
    - `tensors (str)`: Folder of `.npz` files written by stage A.
    - `sids (list)`: Subjects to use; they are processed as one batch.
    - `levels (list)`: Diffusion steps to measure at.
    - `draws (int, optional)`: Iterates per level. Default `8`.

    Returns
    -------
    - `list`: One `(level, ratio, in_spread, out_spread)` tuple per level.
    """
    ids, cond, desc = load_inputs(tensors, sids)
    target = np.stack([np.load(os.path.join(tensors, f'{s}.npz'))['target'].T
                       for s in ids]).astype(np.float32)
    return [(lv, *sensitivity(net, cond, desc, target, lv, draws)) for lv in levels]


def main():
    """Measure the sensitivity of one checkpoint to its own iterate and print it."""
    ap = argparse.ArgumentParser(description='Does the network read the iterate it is denoising?')
    ap.add_argument('--ckpt', required=True, help='a stage B checkpoint, or a released one')
    ap.add_argument('--tensors', required=True, help='folder of .npz files written by stage A')
    ap.add_argument('--subjects', nargs='*', help='subjects to use; default the first --limit')
    ap.add_argument('--limit', type=int, default=8, help='how many subjects if none are named')
    ap.add_argument('--draws', type=int, default=8, help='different iterates per noise level')
    ap.add_argument('--levels', type=int, nargs='*',
                    default=[1999, 1500, 1000, 500, 200, 50, 10, 1],
                    help='diffusion steps to measure at, high noise first')
    args = ap.parse_args()

    sids = args.subjects
    if not sids:
        files = sorted(glob.glob(os.path.join(args.tensors, '*.npz')))[:args.limit]
        sids = [os.path.basename(f)[:-4] for f in files]

    net = load_network(args.ckpt)
    rows = sweep(net, args.tensors, sids, args.levels, args.draws)

    print(f'\n{len(sids)} subjects, {args.draws} iterates per level, {args.ckpt}\n')
    print(f'  {"step":>5s} {"spread in":>11s} {"spread out":>11s} {"out / in":>10s}')
    for level, ratio, si, so in rows:
        print(f'  {level:5d} {si:11.5f} {so:11.5f} {ratio:10.5f}')
    worst = max(r for _, r, _, _ in rows)
    print(f'\n  largest ratio over all levels: {worst:.5f}')
    print('  A network that used its iterate would sit near 1 at the lowest noise levels,\n'
          '  where the iterate is almost the answer. A ratio this far below 1 there means\n'
          '  the prediction comes from the conditioning alone, which is why one denoising\n'
          '  step reproduces the full schedule.')


if __name__ == '__main__':
    main()
