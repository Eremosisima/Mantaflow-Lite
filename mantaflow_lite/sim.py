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
    rain_every: int = 4            # intervalo medio entre gotas (cada ranura)
    rain_drops: int = 3            # ranuras de gotas (se escalonan entre si)
    rain_seed: int = 7             # semilla -> misma lluvia en todos los backends
    rain_spread: float = 0.0       # semi-ancho de la banda de gotas [0,1] (0=ancho completo)
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
    """Centros (x, y) de las gotas en este paso, como fracciones de rejilla.

    - Cada ranura de gota tiene fase propia: gotas escalonadas, no ráfagas.
    - Si `rain_spread` > 0, x cae en una banda alrededor de la esfera.
    - Con `obstacle == "sphere"`, ~40% de las gotas nace BAJO la esfera (las
      que "pasaron por detrás"): al caer cierran su arco inferior y la esfera
      se lee como volumen y no como columna.

    y=1.0 es el techo (fila r-1), y baja hacia 0.0 (piso).
    Funcion pura: los tres backends ven exactamente la misma lluvia.
    """
    if not cfg.rain or cfg.rain_drops <= 0:
        return ()
    every = max(1, cfg.rain_every)
    rng = random.Random(cfg.rain_seed * 1_000_003 + step)
    spots = []
    for k in range(cfg.rain_drops):
        phase = random.Random(cfg.rain_seed * 65_599 + k).randrange(every)
        if (step - phase) % every != 0:
            continue
        x = rng.random()
        if cfg.obstacle == "sphere" and rng.random() < 0.4:
            x = cfg.obstacle_x + (2.0 * rng.random() - 1.0) * cfg.obstacle_radius * 0.9
            # altura con variacion: la linea de aparicion no se ve como guiones
            y = (cfg.obstacle_y - cfg.obstacle_radius) - (2.5 + 2.0 * rng.random()) / cfg.res
            y = max(y, 3.0 / cfg.res)
        else:
            if cfg.rain_spread > 0.0:
                x = (cfg.obstacle_x - cfg.rain_spread) + 2.0 * cfg.rain_spread * x
            y = 1.0
        spots.append((min(max(x, 0.0), 1.0), y))
    return tuple(spots)


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
