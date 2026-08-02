"""
setup_input
===========
Reorganise the full Teeth3DS download (which is split into `data_part_*/upper`
and `data_part_*/lower` subfolders) into the per-patient layout the pipeline
expects: one folder per patient holding both arches
(`<pid>/<pid>_{upper,lower}.{obj,json}`).

Functions
---------
- `index_parts(parts_root)`: Map every available `(pid, jaw, ext)` to its file path.
- `main()`: Gather the requested patients into `--out`, copying or symlinking.

Example
-------
```bash
python setup_input.py --parts ~/Downloads --out ../Teeth3DS_input \
    --ids gt_patient_ids.txt --copy --limit 8
```

Notes
-----
- Only patients that have all four files (upper/lower obj + json) are kept.
- `--copy` makes the folder independent of the download; without it, symlinks are
  created (faster and lighter, but they break if the download is deleted).
"""
import os
import glob
import shutil
import argparse


def index_parts(parts_root):
    """
    Index every patient arch file found under the `data_part_*` folders.

    Parameters
    ----------
    - `parts_root (str)`: Folder that contains the `data_part_*` directories.

    Returns
    -------
    - `dict`: `{(pid, jaw, ext): path}` for every file present.
    """
    files = {}
    for part in glob.glob(os.path.join(parts_root, "data_part_*")):
        for jaw in ("upper", "lower"):
            for pdir in glob.glob(os.path.join(part, jaw, "*")):
                pid = os.path.basename(pdir)
                for ext in ("obj", "json"):
                    f = os.path.join(pdir, f"{pid}_{jaw}.{ext}")
                    if os.path.exists(f):
                        files[(pid, jaw, ext)] = f
    return files


def main():
    """Gather the requested patients into the per-patient layout under `--out`."""
    ap = argparse.ArgumentParser(description="Reorganise Teeth3DS data_part_* into per-patient folders.")
    ap.add_argument("--parts", required=True, help="folder containing the data_part_* directories")
    ap.add_argument("--out", required=True, help="target folder (one subfolder per patient)")
    ap.add_argument("--ids", help="text file with patient ids (one per line); default = all found")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    ap.add_argument("--limit", type=int, default=0, help="max number of patients (0 = no limit)")
    args = ap.parse_args()

    files = index_parts(args.parts)
    all_pids = sorted({k[0] for k in files})
    wanted = [l.strip() for l in open(args.ids)] if args.ids else all_pids
    wanted = [p for p in wanted if p]

    done, missing = 0, 0
    for pid in wanted:
        need = [(pid, j, e) for j in ("upper", "lower") for e in ("obj", "json")]
        if not all(k in files for k in need):
            missing += 1
            continue
        dst_dir = os.path.join(args.out, pid)
        os.makedirs(dst_dir, exist_ok=True)
        for k in need:
            dst = os.path.join(dst_dir, os.path.basename(files[k]))
            if os.path.exists(dst) or os.path.islink(dst):
                continue
            if args.copy:
                shutil.copy2(files[k], dst)
            else:
                os.symlink(os.path.abspath(files[k]), dst)
        done += 1
        if args.limit and done >= args.limit:
            break

    print(f"indexed {len(all_pids)} patients across the parts")
    print(f"prepared {done} complete patients into {args.out}"
          + (f"  ({missing} requested ids had incomplete/absent data)" if args.ids else ""))


if __name__ == "__main__":
    main()
