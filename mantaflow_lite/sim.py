"""Configuracion y fabrica de backends."""
from dataclasses import dataclass


@dataclass
class SimConfig:
    res: int = 256                 # rejilla res x res (par para media resolucion)
    steps: int = 60                # pasos a simular
    dt: float = 0.1
    pour_x: float = 0.5            # posicion horizontal del vertido [0,1]
    pour_width: int = 3            # ancho del chorro en celdas
    pour_speed: float = 8.0        # velocidad inicial del vertido
    gravity: float = 8.0           # gravedad sobre la densidad
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
