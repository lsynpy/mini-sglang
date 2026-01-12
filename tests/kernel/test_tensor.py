from __future__ import annotations

import torch
from minisgl.kernel import test_tensor
from minisgl.utils import call_if_main


@call_if_main()
def main():
    # This test specifically requires cuda:1 as the kernel is hardcoded to only accept cuda:1
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = "cuda:1"
    else:
        print("Skipping test: requires at least 2 CUDA devices (cuda:0 and cuda:1)")
        return

    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y = torch.empty((12, 1024), dtype=torch.int64, device=device)
    test_tensor(x, y)
