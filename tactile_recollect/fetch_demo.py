"""
Stream a few demos out of a (multi-GB) HuggingFace DexMimicGen dataset via
byte-range reads and cache them into a tiny local hdf5 with the IDENTICAL layout
(data/<demo>/{states,actions,obs/*,...}, per-demo attrs model_file+num_samples,
data.attrs env_args). Lets the rest of the pipeline run locally and fast without
downloading the whole 4-6 GB file.

Usage:
    SSL_CERT_FILE=/tmp/ca_combined.pem \
    python -m tactile_recollect.fetch_demo \
        --task two_arm_can_sort_random --demos demo_0 demo_1 \
        --out /tmp/mini_can_sort.hdf5
"""
import argparse
import json

import h5py
import numpy as np
import fsspec

HF_DIR = "datasets/MimicGen/dexmimicgen_datasets/generated"


def _copy_group(src, dst):
    for k in src.keys():
        if isinstance(src[k], h5py.Group):
            _copy_group(src[k], dst.create_group(k))
        else:
            dst.create_dataset(k, data=src[k][()])
    for ak, av in src.attrs.items():
        dst.attrs[ak] = av


def fetch(task, demos, out):
    path = f"{HF_DIR}/{task}.hdf5"
    fs = fsspec.filesystem("hf")
    with fs.open(path, "rb") as fobj, h5py.File(fobj, "r") as hf:
        data = hf["data"]
        env_args = data.attrs["env_args"]
        print("env_args:", json.dumps(json.loads(env_args), indent=2)[:800])
        if not demos:
            all_demos = list(data.keys())
            demos = sorted(all_demos, key=lambda d: int(d.split("_")[1]))[:2]
        with h5py.File(out, "w") as of:
            od = of.create_group("data")
            od.attrs["env_args"] = env_args
            total = 0
            for d in demos:
                g = data[d]
                print(f"  {d}: keys={list(g.keys())} "
                      f"states={g['states'].shape} n={g.attrs['num_samples']}")
                if "obs" in g:
                    print("    obs keys:", list(g["obs"].keys()))
                og = od.create_group(d)
                _copy_group(g, og)
                total += int(g.attrs["num_samples"])
            od.attrs["total"] = total
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="two_arm_can_sort_random")
    ap.add_argument("--demos", nargs="*", default=["demo_0", "demo_1"])
    ap.add_argument("--out", default="/tmp/mini_can_sort.hdf5")
    a = ap.parse_args()
    fetch(a.task, a.demos, a.out)
