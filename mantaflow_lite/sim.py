"""Configuracion, fabrica de backends y escenas compartidas."""
import random
from dataclasses import dataclass

import numpy as np


@dataclass
class SimConfig:
    res: int = 256                 # rejilla res x res (par para media resolucion)
    steps: int = 60                # pasos a simular
    dt: float = 0.1
    pour_x: float = 0.5            # posicion horizontal del vertido [0,1]
    pour_width: int = 3            # ancho del chorro en celdas
    pour_speed: float = 8.0        # velocidad inicial del vertido
    gravity: float = 8.0           # gravedad sobre la densidad
    # --- escena: esfera invisible + lluvia ---
    obstacle: str = "none"         # "none" | "sphere"
    obstacle_x: float = 0.5        # centro horizontal [0,1]
    obstacle_y: float = 0.45       # centro vertical desde el piso [0,1]
    obstacle_radius: float = 0.18  # radio como fraccion de res
    rain: bool = False             # gotas de lluvia (reemplaza al vertido)
    rain_every: int = 4            # pasos entre lluvias
    rain_drops: int = 3            # gotas por lluvia
    rain_seed: int = 7             # semilla -> misma lluvia en todos los backends
    # --- knobs de solver ---
    cg_tol: float = 1e-3           # tolerancia relativa (modo cpu)
    cg_max_iter: int = 400
    jacobi_iters: int = 8          # iteraciones fijas en media res (tricks)
    polish_iters: int = 4          # iteraciones de pulido a res completa (tricks)
    half_res: bool = True          # resolver presion a media resolucion (tricks)
    solver: str = "cg"             # "cg" | "jacobi" (solo gpu decide aqui)

    def __post_init__(self):
        if self.res % 2 != 0:
            raise ValueError(f"res debe ser par (media resolucion), recibido {self.res}")


def build_fluid(cfg: SimConfig) -> np.ndarray:
    """Mascara de celdas de fluido (1.0) vs solido (0.0), forma (res, res).

    La esfera usa centros de celda (i+0.5, j+0.5); la fila 0 es el piso.
    Es la MISCA referencia para los tres backends: paridad exacta.
    """
    r = cfg.res
    if cfg.obstacle != "sphere":
        return np.ones((r, r), dtype=np.float64)
    ii, jj = np.meshgrid(np.arange(r, dtype=np.float64) + 0.5,
                         np.arange(r, dtype=np.float64) + 0.5)
    cx = cfg.obstacle_x * r
    cy = cfg.obstacle_y * r
    rad = cfg.obstacle_radius * r
    inside = (ii - cx) ** 2 + (jj - cy) ** 2 < rad * rad
    fluid = np.ones((r, r), dtype=np.float64)
    fluid[inside] = 0.0
    return fluid


def rain_spots(cfg: SimConfig, step: int):
    """Centros horizontales [0,1] de las gotas en este paso (determinista).

    Funcion pura: los tres backends ven exactamente la misma lluvia.
    """
    if not cfg.rain or step % cfg.rain_every != 0:
        return ()
    rng = random.Random(cfg.rain_seed * 1_000_003 + step)
    return tuple(rng.random() for _ in range(cfg.rain_drops))


def create_backend(name: str, cfg: SimConfig):
    """Instancia el backend indicado: 'cpu' | 'tricks' | 'gpu'."""
    name = name.lower()
    if name == "cpu":
        from .backends.cpu import CpuBackend
        return CpuBackend(cfg)
    if name == "tricks":
        from .backends.tricks import TricksBackend
        return TricksBackend(cfg)
    if name == "gpu":
        from .backends.gpu import GpuBackend
        return GpuBackend(cfg)
    raise ValueError(f"backend desconocido: {name!r} (use cpu|tricks|gpu)")
