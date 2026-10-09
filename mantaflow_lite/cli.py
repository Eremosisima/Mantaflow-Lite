"""CLI: python -m mantaflow_lite --backend cpu|tricks|gpu ..."""
import argparse
import os
import time

from .render import save_frame
from .sim import SimConfig, create_backend


def build_parser():
    p = argparse.ArgumentParser(
        prog="mantaflow_lite",
        description="Solver de fluidos 2D con backend seleccionable (cpu|tricks|gpu)",
    )
    p.add_argument("--backend", choices=["cpu", "tricks", "gpu"], default="cpu")
    p.add_argument("--res", type=int, default=256, help="rejilla res x res (par)")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--out", default="out", help="directorio de frames PNG")
    p.add_argument("--save-every", type=int, default=5, help="guardar 1 frame cada N pasos (0=solo final)")
    p.add_argument("--solver", choices=["cg", "jacobi"], default="jacobi",
                   help="solver de presion para backend gpu")
    p.add_argument("--jacobi-iters", type=int, default=8, help="iteraciones fijas (tricks/gpu)")
    p.add_argument("--polish-iters", type=int, default=4, help="pulido a res completa (tricks/gpu)")
    p.add_argument("--no-half-res", action="store_true", help="presion a res completa")
    p.add_argument("--cg-tol", type=float, default=1e-3, help="tolerancia CG (cpu/gpu-cg)")
    p.add_argument("--gravity", type=float, default=8.0)
    p.add_argument("--pour-x", type=float, default=0.5, help="posicion horizontal del vertido [0,1]")
    p.add_argument("--pour-width", type=int, default=3, help="ancho del chorro/gota en celdas")
    p.add_argument("--pour-speed", type=float, default=8.0, help="velocidad inicial del vertido")
    p.add_argument("--scene", choices=["default", "rain"], default="default",
                   help="default: vertido unico | rain: gotas de lluvia sobre esfera invisible")
    p.add_argument("--obstacle-x", type=float, default=0.5, help="centro horizontal de la esfera")
    p.add_argument("--obstacle-y", type=float, default=0.45, help="centro vertical de la esfera (0=piso)")
    p.add_argument("--obstacle-radius", type=float, default=0.18, help="radio de la esfera (fraccion de res)")
    p.add_argument("--rain-every", type=int, default=4, help="intervalo medio entre gotas")
    p.add_argument("--rain-drops", type=int, default=3, help="ranuras de gotas (se escalonan entre si)")
    p.add_argument("--rain-seed", type=int, default=7, help="semilla de la lluvia")
    p.add_argument("--rain-spread", type=float, default=None,
                   help="semi-ancho de la banda de gotas alrededor de la esfera "
                        "(por defecto 2x radio; 0 = ancho completo)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = SimConfig(
        res=args.res,
        steps=args.steps,
        dt=args.dt,
        solver=args.solver,
        jacobi_iters=args.jacobi_iters,
        polish_iters=args.polish_iters,
        half_res=not args.no_half_res,
        cg_tol=args.cg_tol,
        gravity=args.gravity,
        pour_x=args.pour_x,
        pour_width=args.pour_width,
        pour_speed=args.pour_speed,
        obstacle="sphere" if args.scene == "rain" else "none",
        obstacle_x=args.obstacle_x,
        obstacle_y=args.obstacle_y,
        obstacle_radius=args.obstacle_radius,
        rain=args.scene == "rain",
        rain_every=args.rain_every,
        rain_drops=args.rain_drops,
        rain_seed=args.rain_seed,
        rain_spread=(args.rain_spread if args.rain_spread is not None
                     else (2.0 * args.obstacle_radius if args.scene == "rain" else 0.0)),
    )
    backend = create_backend(args.backend, cfg)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    for s in range(cfg.steps):
        backend.step()
        if args.save_every and (s % args.save_every == 0):
            save_frame(backend.density(), os.path.join(args.out, f"{args.backend}_{s:04d}.png"))
    # evita la sincronizacion implicita del timing en gpu
    if args.backend == "gpu":
        import torch
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    save_frame(backend.density(), os.path.join(args.out, f"{args.backend}_final.png"))
    info = backend.info()
    print(f"backend   : {args.backend}")
    print(f"device    : {info.get('device', '?')}")
    print(f"res/steps : {cfg.res}x{cfg.res} / {cfg.steps}")
    print(f"tiempo    : {elapsed:.3f} s total | {1000 * elapsed / cfg.steps:.2f} ms/paso")
    print(f"divergencia  antes -> despues de proyectar: "
          f"{backend.pre_div:.5f} -> {backend.max_divergence():.5f}")
    extra = {k: v for k, v in info.items() if k != "device"}
    if extra:
        print(f"solver    : {extra}")
    print(f"frames    : {args.out}")


if __name__ == "__main__":
    main()
