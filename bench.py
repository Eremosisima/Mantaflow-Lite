"""Benchmark: python bench.py [res ...]"""
import platform
import sys
import time

from mantaflow_lite import SimConfig, create_backend

STEPS = 8
WARMUP = 3


def bench(name, res, solver=None):
    cfg = SimConfig(res=res, steps=STEPS, **({"solver": solver} if solver else {}))
    b = create_backend(name, cfg)
    for _ in range(WARMUP):          # JIT + init GPU
        b.step()
    if name == "gpu":
        import torch
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        b.step()
    if name == "gpu":
        import torch
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / STEPS * 1000.0
    return ms, b.pre_div, b.max_divergence(), b.info()


def main():
    res_list = [int(a) for a in sys.argv[1:]] or [128, 256, 512]
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if gpu_ok else "N/A"
    except Exception:
        gpu_ok, dev = False, "N/A"
    print(f"CPU      : {platform.processor() or platform.machine()} / {platform.platform()}")
    print(f"GPU      : {dev}")
    print(f"Python   : {platform.python_version()}")
    print()
    print("| res | backend | ms/paso | divergencia pre -> post |")
    print("|-----|---------|---------|-------------------------|")
    combos = [("cpu", "cg", "cpu"), ("tricks", None, "tricks"),
              ("gpu", "jacobi", "gpu jacobi"), ("gpu", "cg", "gpu cg")]
    for res in res_list:
        for name, solver, label in combos:
            if name == "gpu" and not gpu_ok:
                continue
            ms, pre, post, info = bench(name, res, solver)
            print(f"| {res} | {label} | {ms:.2f} | {pre:.4f} -> {post:.5f} |")


if __name__ == "__main__":
    main()
