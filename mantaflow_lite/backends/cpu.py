"""Backend CPU: numba (paralelo) para adveccion + Gradiente Conjugado para presion.

Es la referencia de calidad: CG converge, la divergencia residual es minima.
"""
import math

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------- muestreo
@njit(cache=True, inline="always")
def _bilinear(f, fx, fy):
    """_interpola f[j,i] cuyo punto de malla es (i, j) en la posicion (fx, fy)."""
    ny, nx = f.shape
    x0 = int(math.floor(fx))
    y0 = int(math.floor(fy))
    tx = fx - x0
    ty = fy - y0
    if x0 < 0:
        x0 = 0
        tx = 0.0
    elif x0 > nx - 2:
        x0 = nx - 2
        tx = 1.0
    if y0 < 0:
        y0 = 0
        ty = 0.0
    elif y0 > ny - 2:
        y0 = ny - 2
        ty = 1.0
    v00 = f[y0, x0]
    v01 = f[y0, x0 + 1]
    v10 = f[y0 + 1, x0]
    v11 = f[y0 + 1, x0 + 1]
    return (v00 * (1.0 - tx) + v01 * tx) * (1.0 - ty) + (v10 * (1.0 - tx) + v11 * tx) * ty


@njit(cache=True, inline="always")
def _sample_u(u, x, y):
    return _bilinear(u, x, y - 0.5)


@njit(cache=True, inline="always")
def _sample_v(v, x, y):
    return _bilinear(v, x - 0.5, y)


@njit(cache=True, inline="always")
def _sample_d(d, x, y):
    return _bilinear(d, x - 0.5, y - 0.5)


# ---------------------------------------------------------------- adveccion
@njit(parallel=True, cache=True)
def _advect_u(u, v, dt, out):
    ny, nxf = u.shape
    for j in prange(ny):
        for i in range(1, nxf - 1):
            x = float(i)
            y = j + 0.5
            vel_u = _sample_u(u, x, y)
            vel_v = _sample_v(v, x, y)
            out[j, i] = _sample_u(u, x - dt * vel_u, y - dt * vel_v)


@njit(parallel=True, cache=True)
def _advect_v(u, v, dt, out):
    nyf, nx = v.shape
    for j in prange(1, nyf - 1):
        for i in range(nx):
            x = i + 0.5
            y = float(j)
            vel_u = _sample_u(u, x, y)
            vel_v = _sample_v(v, x, y)
            out[j, i] = _sample_v(v, x - dt * vel_u, y - dt * vel_v)


@njit(parallel=True, cache=True)
def _advect_d(u, v, d, dt, out):
    ny, nx = d.shape
    for j in prange(ny):
        for i in range(nx):
            x = i + 0.5
            y = j + 0.5
            vel_u = 0.5 * (u[j, i] + u[j, i + 1])
            vel_v = 0.5 * (v[j, i] + v[j + 1, i])
            out[j, i] = _sample_d(d, x - dt * vel_u, y - dt * vel_v)


# ---------------------------------------------------------------- laplaciano
def _neighbor_diag(r):
    diag = np.full((r, r), 4.0)
    diag[:, 0] -= 1.0
    diag[:, -1] -= 1.0
    diag[0, :] -= 1.0
    diag[-1, :] -= 1.0
    return diag


def _spmv(p, diag):
    """A p = diag*p - suma de vecinos (condiciones de Neumann en paredes)."""
    ap = diag * p
    ap[:, 1:] -= p[:, :-1]
    ap[:, :-1] -= p[:, 1:]
    ap[1:, :] -= p[:-1, :]
    ap[:-1, :] -= p[1:, :]
    return ap


class CpuBackend:
    name = "cpu"

    def __init__(self, cfg):
        self.cfg = cfg
        r = cfg.res
        self.u = np.zeros((r, r + 1), dtype=np.float64)
        self.v = np.zeros((r + 1, r), dtype=np.float64)
        self.d = np.zeros((r, r), dtype=np.float64)
        self._ut = np.zeros_like(self.u)
        self._vt = np.zeros_like(self.v)
        self._dt = np.zeros_like(self.d)
        self._diag = _neighbor_diag(r)
        self.last_cg_iters = 0
        self.pre_div = 0.0

    # -------------------------------------------------------------- pasos
    def _inject_pour(self):
        r = self.cfg.res
        cx = int(self.cfg.pour_x * r)
        half = self.cfg.pour_width // 2
        lo, hi = max(0, cx - half), min(r, cx + half + 1)
        self.d[r - 1, lo:hi] = 1.0
        self.v[r - 1, lo:hi] = -self.cfg.pour_speed

    def _apply_gravity(self, dt):
        g = self.cfg.gravity * dt
        if g > 0.0:
            self.v[1:-1, :] -= g * 0.5 * (self.d[:-1, :] + self.d[1:, :])

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
        dvm = (self.u[:, 1:] - self.u[:, :-1]) + (self.v[1:, :] - self.v[:-1, :])
        return dvm

    def _cg_solve(self, b):
        cfg = self.cfg
        p = np.zeros_like(b)
        r = b.copy()
        x = r.copy()
        rs = float(r.ravel() @ r.ravel())
        bnorm = math.sqrt(rs)
        if bnorm < 1e-14:
            return p, 0
        target = cfg.cg_tol * bnorm
        it = 0
        for it in range(1, cfg.cg_max_iter + 1):
            ap = _spmv(x, self._diag)
            denom = float(x.ravel() @ ap.ravel())
            if abs(denom) < 1e-30:
                break
            alpha = rs / denom
            p += alpha * x
            r -= alpha * ap
            rs_new = float(r.ravel() @ r.ravel())
            if math.sqrt(rs_new) < target:
                rs = rs_new
                break
            x = r + (rs_new / rs) * x
            rs = rs_new
        p -= p.mean()
        return p, it

    def _project(self):
        b = -self.divergence()
        self.pre_div = float(np.abs(b).max())
        p, iters = self._cg_solve(b)
        self.last_cg_iters = iters
        self.u[:, 1:-1] -= p[:, 1:] - p[:, :-1]
        self.v[1:-1, :] -= p[1:, :] - p[:-1, :]

    def step(self, dt=None):
        dt = self.cfg.dt if dt is None else dt
        self._inject_pour()
        self._apply_gravity(dt)
        self._advect(dt)
        self._project()

    def max_divergence(self):
        return float(np.abs(self.divergence()).max())

    def density(self):
        return self.d

    def info(self):
        return {"cg_iters_last": self.last_cg_iters, "device": "CPU (numba parallel)"}
