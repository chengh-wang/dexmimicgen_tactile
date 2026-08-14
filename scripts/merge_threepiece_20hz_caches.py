#!/usr/bin/env python3
"""Merge ThreePiece 20 Hz latent-cache HDF5 files.

Each input cache is a flat HDF5 file with one group per demo. Groups contain
`tactile_mu`, `joint`, and `action`. This script copies groups losslessly into a
new cache and fails on duplicate keys.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def is_success_key(key: str) -> bool:
    return key.split("__")[-1].startswith("success_")


def is_failure_key(key: str) -> bool:
    return key.split("__")[-1].startswith("failure_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    total = 0
    success = 0
    failure = 0
    with h5py.File(tmp, "w") as out:
        out.attrs["format"] = "threepiece_20hz_latent_cache_merged"
        out.attrs["inputs"] = "\n".join(args.inputs)
        for input_path in args.inputs:
            with h5py.File(input_path, "r") as src:
                for key in src.keys():
                    if key in out:
                        raise KeyError(f"duplicate cache key: {key}")
                    src.copy(src[key], out, name=key)
                    total += 1
                    success += int(is_success_key(key))
                    failure += int(is_failure_key(key))
    tmp.rename(output)
    print(f"wrote={output}")
    print(f"total={total} success={success} failure={failure}")


if __name__ == "__main__":
    main()
