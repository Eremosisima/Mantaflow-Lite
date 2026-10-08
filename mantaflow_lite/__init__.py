"""Mantaflow-Lite: selector de backends para un solver de fluidos 2D.

Backends:
  cpu    - numba (paralelo CPU) + Gradiente Conjugado (solucion exacto)
  tricks - aproximaciones de videojuego (Jacobi fijo + presion a media resolucion)
  gpu    - PyTorch/CUDA (Jacobi fijo por defecto, o CG con --solver cg)
"""

__version__ = "0.1.0"

from .sim import SimConfig, create_backend

__all__ = ["SimConfig", "create_backend", "__version__"]
