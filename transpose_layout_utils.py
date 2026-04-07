import sys
import time
import torch
import sbvr
import copy
import os
import re
import shutil

src_dir = "/home/wonjun/workspace/sbvr/data"
dst_dir = "/home/wonjun/workspace/sbvr/data_tp_layout"

os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    if not filename.endswith(".pt"):
        continue

    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    # Rule 1: Just copy tensors whose name starts with "matrix"
    if filename.startswith("matrix"):
        shutil.copy2(src_path, dst_path)
        print(f"[Rule 1] Copied: {filename}")

    # Rules 2 & 3: Handle tensors starting with "sbvr"
    elif filename.startswith("sbvr"):
        # Extract the shape substring after "v1_", e.g. [1_1024] or [1024_1024]
        match = re.search(r'v1_\[(\d+)_(\d+)\]', filename)
        if match:
            first_dim = int(match.group(1))
            second_dim = int(match.group(2))

            # Rule 2: If the shape is [1_xxx], just copy the tensor
            if first_dim == 1:
                shutil.copy2(src_path, dst_path)
                print(f"[Rule 2] Copied: {filename}")

            # Rule 3: If first_dim > 1, do some operations
            else:
                # TODO: Implement transpose layout operations here
                sbvr_orig = sbvr.load(src_path, device="cpu", verbose_level=0, cpu_kernel=True)
                sbvr_orig.coeff_idx.data = sbvr_orig.coeff_idx.transpose(0, 1).contiguous()  # Transpose the coeff_idx tensor
                sbvr_orig.save(dst_path)
                print(f"[Rule 3] Transposed: {filename} (shape [{first_dim}_{second_dim}])")
                pass
        else:
            print(f"[Skip] Could not parse shape from: {filename}")
