"""Backend GPU: PyTorch/CUDA.

Dos solvers:
  jacobi (default) - misma receta "tricks" que el backend cpu (media res +
                      pulido, iteraciones fijas). Permite comparar 1:1 el
                      speedup puro de hardware con el mismo algoritmo.
  cg               - Gradiente Conjugado completo en la GPU (calidad de "cpu").

Todo el estado vive en VRAM: no hay transfers por paso, solo al renderizar.
"""
import math

import torch


def _bilinear(f, fx, fy):
    """f[j,i] con punto de malla (i,j); interpola en (fx, fy) en unidades de malla."""
    ny, nx = f.shape
    x0 = torch.floor(fx).long().clamp(0, nx - 2)
    y0 = torch.floor(fy).long().clamp(0, ny - 2)
    tx = (fx - x0.float()).clamp(0.0, 1.0)
    ty = (fy - y0.float()).clamp(0.0, 1.0)
    v00 = f[y0, x0]
    v01 = f[y0, x0 + 1]
    v10 = f[y0 + 1, x0]
    v11 = f[y0 + 1, x0 + 1]
    top = v00 * (1.0 - tx) + v01 * tx
    bot = v10 * (1.0 - tx) + v11 * tx
    return top * (1.0 - ty) + bot * ty


def _neighbor_diag(r, device):
    diag = torch.full((r, r), 4.0, device=device)
    diag[:, 0] -= 1.0
    diag[:, -1] -= 1.0
    diag[0, :] -= 1.0
    diag[-1, :] -= 1.0
    return diag


def _spmv(p, diag):
    ap = diag * p
    ap[:, 1:] -= p[:, :-1]
    ap[:, :-1] -= p[:, 1:]
    ap[1:, :] -= p[:-1, :]
    ap[:-1, :] -= p[1:, :]
    return ap


def _jacobi(b, diag, iters, p0=None):
    p = torch.zeros_like(b) if p0 is None else p0.clone()
    for _ in range(iters):
        ps = torch.zeros_like(p)
        ps[:, 1:] += p[:, :-1]
        ps[:, :-1] += p[:, 1:]
        ps[1:, :] += p[:-1, :]
        ps[:-1, :] += p[1:, :]
        p = torch.where(diag > 0.0, (b + ps) / diag.clamp(min=1.0), torch.zeros_like(p))
    return p


def _upsample_bilinear(p2, idx0, idx1, t):
    """Sube (r/2)^2 -> r^2 con bilineal (misma formula que el backend tricks)."""
    a = p2[idx0, :] * (1.0 - t[:, None]) + p2[idx1, :] * t[:, None]
    return a[:, idx0] * (1.0 - t[None, :]) + a[:, idx1] * t[None, :]


class GpuBackend:
    name = "gpu"

    def __init__(self, cfg):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no disponible: no se puede usar backend=gpu")
        self.cfg = cfg
        self.device = torch.device("cuda:0")
        r = cfg.res
        dev, f32 = self.device, torch.float32
        self.u = torch.zeros((r, r + 1), dtype=f32, device=dev)
        self.v = torch.zeros((r + 1, r), dtype=f32, device=dev)
        self.d = torch.zeros((r, r), dtype=f32, device=dev)
        self._diag_full = _neighbor_diag(r, dev)
        self._diag_half = _neighbor_diag(r // 2, dev) if cfg.half_res else None
        if cfg.half_res:
            pos = (torch.arange(r, dtype=f32, device=dev) - 0.5) * 0.5
            rc = max(r // 2, 1)
            i0 = torch.floor(pos).long().clamp(0, max(rc - 2, 0))
            self._upt = (pos - i0.float()).clamp(0.0, 1.0)
            self._up0, self._up1 = i0, (i0 + 1).clamp(max=rc - 1)
        self.iters_used = 0
        self.pre_div = 0.0

        # coordenadas de caras para adveccion (se reutilizan cada paso)
        self._ux_x = torch.arange(1, r, dtype=f32, device=dev).unsqueeze(0).expand(r, -1)
        self._ux_y = (torch.arange(r, dtype=f32, device=dev) + 0.5).unsqueeze(1).expand(-1, r - 1)
        self._vx_x = (torch.arange(r, dtype=f32, device=dev) + 0.5).unsqueeze(0).expand(r - 1, -1)
        self._vx_y = torch.arange(1, r, dtype=f32, device=dev).unsqueeze(1).expand(-1, r)
        self._dx_x = (torch.arange(r, dtype=f32, device=dev) + 0.5).unsqueeze(0).expand(r, -1)
        self._dx_y = (torch.arange(r, dtype=f32, device=dev) + 0.5).unsqueeze(1).expand(-1, r)

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
        u, v, d = self.u, self.v, self.d
        # --- u (caras verticales interiores) ---
        x, y = self._ux_x, self._ux_y
        pu = _bilinear(u, x, y - 0.5)
        pv = _bilinear(v, x - 0.5, y)
        u_new = torch.zeros_like(u)
        u_new[:, 1:-1] = _bilinear(u, x - dt * pu, y - dt * pv - 0.5)
        # --- v (caras horizontales interiores) ---
        x, y = self._vx_x, self._vx_y
        pu = _bilinear(u, x, y - 0.5)
        pv = _bilinear(v, x - 0.5, y)
        v_new = torch.zeros_like(v)
        v_new[1:-1, :] = _bilinear(v, x - dt * pu - 0.5, y - dt * pv)
        self.u, self.v = u_new, v_new
        # --- densidad (con la velocidad ya advectada, igual que los backends cpu) ---
        x, y = self._dx_x, self._dx_y
        cu = 0.5 * (u_new[:, :-1] + u_new[:, 1:])
        cv = 0.5 * (v_new[:-1, :] + v_new[1:, :])
        d_new = _bilinear(d, x - dt * cu - 0.5, y - dt * cv - 0.5)
        self.d = d_new.clamp(0.0, 1.0)

    def divergence(self):
        return (self.u[:, 1:] - self.u[:, :-1]) + (self.v[1:, :] - self.v[:-1, :])

    def _cg_solve(self, b):
        cfg = self.cfg
        p = torch.zeros_like(b)
        r = b.clone()
        x = r.clone()
        rs = torch.sum(r * r)
        bnorm = torch.sqrt(rs).item()
        if bnorm < 1e-14:
            return p, 0
        target = cfg.cg_tol * bnorm
        it = 0
        for it in range(1, cfg.cg_max_iter + 1):
            ap = _spmv(x, self._diag_full)
            denom = torch.sum(x * ap).item()
            if abs(denom) < 1e-30:
                break
            alpha = rs / denom
            p += alpha * x
            r -= alpha * ap
            rs_new = torch.sum(r * r)
            if math.sqrt(rs_new.item()) < target:
                rs = rs_new
                break
            x = r + (rs_new / rs) * x
            rs = rs_new
        p -= p.mean()
        return p, it

    def _project(self):
        cfg = self.cfg
        b = -self.divergence()
        self.pre_div = float(torch.abs(b).max().item())
        if cfg.solver == "cg":
            p, iters = self._cg_solve(b)
        else:
            iters = cfg.jacobi_iters
            if cfg.half_res:
                b2 = torch.nn.functional.avg_pool2d(
                    b.unsqueeze(0).unsqueeze(0), 2
                ).squeeze(0).squeeze(0)
                p2 = _jacobi(b2, self._diag_half, cfg.jacobi_iters)
                p = _upsample_bilinear(p2, self._up0, self._up1, self._upt)
                iters += cfg.polish_iters
                if cfg.polish_iters > 0:
                    p = _jacobi(b, self._diag_full, cfg.polish_iters, p0=p)
            else:
                p = _jacobi(b, self._diag_full, cfg.jacobi_iters)
        self.iters_used = iters
        self.u[:, 1:-1] -= p[:, 1:] - p[:, :-1]
        self.v[1:-1, :] -= p[1:, :] - p[:-1, :]

    @torch.no_grad()
    def step(self, dt=None):
        dt = self.cfg.dt if dt is None else dt
        self._inject_pour()
        self._apply_gravity(dt)
        self._advect(dt)
        self._project()

    def max_divergence(self):
        return float(torch.abs(self.divergence()).max().item())

    def density(self):
        return self.d.detach().cpu().numpy()

    def info(self):
        return {
            "iters": self.iters_used,
            "solver": self.cfg.solver,
            "device": torch.cuda.get_device_name(0),
        }
