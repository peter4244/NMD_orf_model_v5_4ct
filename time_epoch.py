#!/usr/bin/env python
"""What does one epoch actually cost? Measured before committing to 40 runs."""
import sys, time, numpy as np, torch
sys.path.insert(0, ".")
from train_v6 import TensorSource, make_batches, INTERPRETABLE
from model_v6 import ScanningNMDModel, count_parameters

dev = "cuda" if torch.cuda.is_available() else "cpu"
src = TensorSource("results_tensor_v6/nmd_tensor.h5", INTERPRETABLE)
tr = src.indices("train")
print(f"device {dev}   train {len(tr):,} tx, {src.count[tr].sum():,} candidates")
if dev == "cuda":
    print(f"gpu {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB\n")
print(f"{'max_pad':>8} {'C':>4} {'batches':>8} {'load s':>8} {'step s':>8} "
      f"{'load%':>6} {'epoch min':>10} {'peak GB':>8}")
print(f"{'-'*8} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*10} {'-'*8}")
for mp in (512, 1024, 2048):
    b = make_batches(src.count[tr], mp)
    for C in (32, 128):
        try:
            if dev == "cuda": torch.cuda.reset_peak_memory_stats()
            m = ScanningNMDModel(conv_channels=C, n_bins=8, n_structural=1).to(dev)
            opt = torch.optim.Adam(m.parameters())
            n = 12
            t0 = time.time()
            for bb in b[:n]: src.batch(tr[bb], dev)
            if dev == "cuda": torch.cuda.synchronize()
            tl = (time.time() - t0) / n
            t0 = time.time()
            for bb in b[:n]:
                A,S,U,M,y = src.batch(tr[bb], dev)
                opt.zero_grad(set_to_none=True)
                torch.nn.functional.binary_cross_entropy_with_logits(m(A,S,U,M), y).backward()
                opt.step()
            if dev == "cuda": torch.cuda.synchronize()
            tf = (time.time() - t0) / n
            pk = torch.cuda.max_memory_allocated()/1e9 if dev=="cuda" else 0
            print(f"{mp:>8} {C:>4} {len(b):>8,} {tl:>8.3f} {tf:>8.3f} "
                  f"{tl/tf*100:>5.0f}% {len(b)*tf/60:>10.1f} {pk:>8.1f}")
            del m, opt
            if dev == "cuda": torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            print(f"{mp:>8} {C:>4} {len(b):>8,} {'OOM':>8}")
            if dev == "cuda": torch.cuda.empty_cache()
