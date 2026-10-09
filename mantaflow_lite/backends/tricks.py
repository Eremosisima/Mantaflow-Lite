"""Backend tricks: las "trampas" de los videojuegos, en CPU.

Trucos aplicados (trabajo FIJO por frame, estilo motor de juego):
  1. Jacobi de iteraciones fijas en vez de CG hasta converger.
  2. Presion resuelta a MEDIA RESOLUCION y subida con bilineal (barato).
  3. Pulido con polish_iters iteraciones a resolucion completa (fijo).
  4. Float32 (suficiente para graficos).

La adveccion usa los MISMOS kernels numba que el backend cpu
(numba recompila por tipo: float32 aqui, float64 alla).
Lo unico que cambia respecto a cpu es el solver de presion.
"""
import numpy as np

from ..sim import build_fluid, rain_spots
from .cpu import _advect_d, _advect_u, _advect_v, _diag_from_pairs, _open_pairs


def _jacobi(b, diag, nl, nu, iters, p0=None):
    """Jacobi vectorizado: A·p = b, A = diag - vecinos abiertos. p0 = estimacion inicial."""
    p = np.zeros_like(b) if p0 is None else p0.copy()
    for _ in range(iters):
        ps = np.zeros_like(p)
        ps[:, 1:] += nl * p[:, :-1]
        ps[:, :-1] += nl * p[:, 1:]
        ps[1:, :] += nu * p[:-1, :]
        ps[:-1, :] += nu * p[1:, :]
        p = np.where(diag > 0.0, (b + ps) / np.maximum(diag, 1.0), 0.0)
    return p


def _upsample_pos(r):
    """Coords de malla full -> malla half para subir la presion (bilineal)."""
    pos = (np.arange(r, dtype=np.float64) - 0.5) * 0.5
    rc = max(r // 2, 1)
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, max(rc - 2, 0))
    t = np.clip(pos - i0, 0.0, 1.0).astype(np.float32)
    i1 = np.minimum(i0 + 1, rc - 1)
    return i0.astype(np.int64), i1, t


def _upsample_bilinear(p2, idx0, idx1, t):
    a = p2[idx0, :] * (1.0 - t[:, None]) + p2[idx1, :] * t[:, None]
    return a[:, idx0] * (1.0 - t[None, :]) + a[:, idx1] * t[None, :]


class TricksBackend:
    name = "tricks"

    def __init__(self, cfg):
        self.cfg = cfg
        r = cfg.res
        self.u = np.zeros((r, r + 1), dtype=np.float32)
        self.v = np.zeros((r + 1, r), dtype=np.float32)
        self.d = np.zeros((r, r), dtype=np.float32)
        self._ut = np.zeros_like(self.u)
        self._vt = np.zeros_like(self.v)
        self._dt = np.zeros_like(self.d)
        self._f = build_fluid(cfg).astype(np.float32)
        self._nl, self._nu = _open_pairs(self._f)
        self._diag_full = _diag_from_pairs(self._nl, self._nu)
        if cfg.half_res:
            self._f2 = self._f.reshape(r // 2, 2, r // 2, 2).mean(axis=(1, 3))
            self._nl2, self._nu2 = _open_pairs(self._f2)
            self._diag_half = _diag_from_pairs(self._nl2, self._nu2)
            self._up0, self._up1, self._upt = _upsample_pos(r)
        else:
            self._f2 = self._nl2 = self._nu2 = self._diag_half = None
            self._up0 = self._up1 = self._upt = None
        self.iters_used = 0
        self.pre_div = 0.0
        self.t = 0

    def _inject(self):
        cfg, r = self.cfg, self.cfg.res
        half = cfg.pour_width // 2
        if cfg.rain:
            for x, y in rain_spots(cfg, self.t):
                cx = int(x * r)
                lo, hi = max(0, cx - half), min(r, cx + half + 1)
                j = min(r - 1, max(1, int(y * (r - 1))))
                self.d[j, lo:hi] = 1.0
                self.v[j, lo:hi] = -cfg.pour_speed
        else:
            cx = int(cfg.pour_x * r)
            lo, hi = max(0, cx - half), min(r, cx + half + 1)
            self.d[r - 1, lo:hi] = 1.0
            self.v[r - 1, lo:hi] = -cfg.pour_speed

    def _apply_gravity(self, dt):
        g = self.cfg.gravity * dt
        if g > 0.0:
            self.v[1:-1, :] -= np.float32(g) * 0.5 * (self.d[:-1, :] + self.d[1:, :])

    def _enforce_solids(self):
        self.u[:, 1:-1] *= self._nl
        self.v[1:-1, :] *= self._nu
        self.d *= self._f

    def _advect(self, dt):
        self._ut[:] = 0.0
        _advect_u(self.u, self.v, dt, self._ut)
        self._vt[:] = 0.0
        _advect_v(self.u, self.v, dt, self._vt)
        self.u, self._ut = self._ut, self.u
        self.v, self._vt = self._vt, self.v
        self._dt[:] = 0.0
        _advect_d(self.u, self.v, self.d, dt, self._dt)
        np.clip(self._dt, 0.0, 1.0, out=self._dt)
        self.d, self._dt = self._dt, self.d

    def divergence(self):
        return (self.u[:, 1:] - self.u[:, :-1]) + (self.v[1:, :] - self.v[:-1, :])

    def _project(self):
        cfg = self.cfg
        b = -self.divergence()
        b *= self._f
        self.pre_div = float(np.abs(b).max())
        iters = cfg.jacobi_iters
        if cfg.half_res:
            b2 = b.reshape(cfg.res // 2, 2, cfg.res // 2, 2).mean(axis=(1, 3))
            p2 = _jacobi(b2.astype(np.float32), self._diag_half,
                         self._nl2, self._nu2, cfg.jacobi_iters)
            p = _upsample_bilinear(p2, self._up0, self._up1, self._upt)
            iters += cfg.polish_iters
            if cfg.polish_iters > 0:
                p = _jacobi(b, self._diag_full, self._nl, self._nu,
                            cfg.polish_iters, p0=p)
        else:
            p = _jacobi(b, self._diag_full, self._nl, self._nu, cfg.jacobi_iters)
        self.iters_used = iters
        self.u[:, 1:-1] -= self._nl * (p[:, 1:] - p[:, :-1])
        self.v[1:-1, :] -= self._nu * (p[1:, :] - p[:-1, :])

    def step(self, dt=None):
        dt = self.cfg.dt if dt is None else dt
        self._inject()
        self._apply_gravity(dt)
        self._enforce_solids()
        self._advect(dt)
        self._enforce_solids()
        self._project()
        self.t += 1

    def max_divergence(self):
        return float(np.abs(self.divergence()).max())

    def density(self):
        return self.d

    def info(self):
        return {"jacobi_iters": self.iters_used, "device": "CPU (numpy fijo/trucos)"}
