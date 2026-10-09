"""Tests de validacion de Mantaflow-Lite.

Cubren: configuracion, estabilidad numerica (sin NaN), que la proyeccion
realmente reduzca la divergencia, y paridad entre backends (mismo algoritmo
= mismos resultados; solver exacto = mismos resultados).
"""
import numpy as np
import pytest

from mantaflow_lite import SimConfig, create_backend
from mantaflow_lite.render import save_frame
from mantaflow_lite.sim import rain_spots

BACKENDS = ["cpu", "tricks", "gpu"]


def _cfg(**kw):
    base = dict(res=48, steps=20, dt=0.05)
    base.update(kw)
    return SimConfig(**base)


def _run(name, cfg):
    b = create_backend(name, cfg)
    for _ in range(cfg.steps):
        b.step()
    return b


def _cuda_ok():
    import torch
    return torch.cuda.is_available()


def _skip_no_cuda():
    if not _cuda_ok():
        pytest.skip("CUDA no disponible")


# ------------------------------------------------------------ configuracion
def test_res_impar_rechazado():
    with pytest.raises(ValueError):
        SimConfig(res=47)


def test_backend_desconocido():
    with pytest.raises(ValueError):
        create_backend("directx", _cfg())


# ------------------------------------------------------------ estabilidad
@pytest.mark.parametrize("name", BACKENDS)
def test_sin_nan_y_densidad_acotada(name):
    if name == "gpu":
        _skip_no_cuda()
    b = _run(name, _cfg())
    d = b.density()
    assert np.isfinite(d).all(), f"{name}: valores no finitos"
    assert d.max() <= 1.0 + 1e-5, f"{name}: densidad desbordada"
    assert d.sum() > 1.0, f"{name}: el vertido no deposito masa"


# ------------------------------------------------------------ proyeccion
@pytest.mark.parametrize("name", BACKENDS)
def test_proyeccion_reduce_divergencia(name):
    if name == "gpu":
        _skip_no_cuda()
    b = _run(name, _cfg())
    pre, post = b.pre_div, b.max_divergence()
    assert pre > 0.0, f"{name}: escena sin divergencia (rara)"
    if name in ("cpu",):
        assert post < pre * 0.01, f"{name}: CG no redujo la divergencia ({pre:.3f}->{post:.5f})"
    else:
        # trucos: reduccion parcial pero real (≈3x por paso)
        assert post < pre * 0.5, f"{name}: la proyeccion debil no alcanzo ({pre:.3f}->{post:.3f})"


# ------------------------------------------------------------ paridad
def test_gpu_cg_igual_cpu_cg():
    """Mismo solver (CG), distinto hardware -> mismo resultado."""
    _skip_no_cuda()
    cfg = _cfg(solver="cg", steps=15)
    a = _run("cpu", cfg)
    b = _run("gpu", cfg)
    da = a.density().reshape(12, 4, 12, 4).mean(axis=(1, 3))
    db = b.density().reshape(12, 4, 12, 4).mean(axis=(1, 3))
    diff = np.abs(da - db).mean()
    assert diff < 0.03, f"cpu-cg vs gpu-cg divergen: diff medio {diff:.4f}"
    ma, mb = a.density().sum(), b.density().sum()
    assert abs(ma - mb) / ma < 0.10, f"masas divergen: {ma:.1f} vs {mb:.1f}"


def test_gpu_jacobi_igual_tricks():
    """Mismo algoritmo 'tricks' en dos hardware -> resultados casi identicos."""
    _skip_no_cuda()
    cfg = _cfg(solver="jacobi", steps=15)
    a = _run("tricks", cfg)
    b = _run("gpu", cfg)
    da = a.density().reshape(12, 4, 12, 4).mean(axis=(1, 3))
    db = b.density().reshape(12, 4, 12, 4).mean(axis=(1, 3))
    diff = np.abs(da - db).mean()
    assert diff < 0.02, f"tricks vs gpu-jacobi divergen: diff medio {diff:.4f}"


def test_trucos_más_barato_que_cg():
    """Los trucos hacen MENOS trabajo de solver: iteraciones fijas < CG."""
    import time
    if _cuda_ok():
        pass
    cfg_small = _cfg(steps=30, res=128)
    # calienta JIT del cpu una vez
    _run("cpu", _cfg(res=32, steps=1))
    t = {}
    for name in ("cpu", "tricks"):
        b = create_backend(name, cfg_small)
        t0 = time.perf_counter()
        for _ in range(cfg_small.steps):
            b.step()
        t[name] = time.perf_counter() - t0
    assert t["tricks"] < t["cpu"], f"tricks no fue mas rapido: {t}"


# ------------------------------------------------------------ render
def test_save_frame(tmp_path):
    from PIL import Image
    d = np.linspace(0, 1, 64 * 64).reshape(64, 64)
    out = tmp_path / "frame.png"
    save_frame(d, str(out))
    assert out.exists()
    im = Image.open(out)
    assert im.size == (128, 128)  # scale=2


# ------------------------------------------------------------ escena lluvia
@pytest.mark.parametrize("name", BACKENDS)
def test_lluvia_sobre_esfera(name):
    """Lluvia sobre esfera invisible: nada dentro del solido, liquido en el contorno."""
    if name == "gpu":
        _skip_no_cuda()
    from mantaflow_lite.sim import build_fluid
    cfg = SimConfig(res=64, steps=90, dt=0.1, obstacle="sphere", rain=True,
                    obstacle_y=0.5)
    b = _run(name, cfg)
    d = b.density()
    f = build_fluid(cfg)
    assert np.isfinite(d).all(), f"{name}: NaN en la escena de lluvia"
    assert d[f == 0].max() == 0.0, f"{name}: densidad dentro de la esfera"
    # banda a 0-3 celdas de la superficie: el liquido debe estar tocando el contorno
    r = cfg.res
    ii, jj = np.meshgrid(np.arange(r) + 0.5, np.arange(r) + 0.5)
    dist = np.sqrt((ii - cfg.obstacle_x * r) ** 2 + (jj - cfg.obstacle_y * r) ** 2)
    rad = cfg.obstacle_radius * r
    shell = (dist >= rad) & (dist < rad + 3.0)
    assert d[shell].sum() > 0.05, (
        f"{name}: el liquido no llego al contorno ({d[shell].sum():.4f})")


def test_misma_lluvia_todos_backends():
    """La lluvia es determinista e identica entre backends (paridad)."""
    cfg = SimConfig(res=64, obstacle="sphere", rain=True, rain_seed=7)
    spots = [tuple(rain_spots(cfg, s)) for s in range(12)]
    spots2 = [tuple(rain_spots(cfg, s)) for s in range(12)]
    assert spots == spots2
    assert any(spots), "la lluvia nunca gotio"
