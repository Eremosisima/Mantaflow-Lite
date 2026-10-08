"""Render de frames: densidad -> magenta fluorescente sobre negro.

Homenaje al efecto que origino este proyecto: vertido fluorescente
bajo luz negra. El liquido de la simulacion se colorea como el video.

Convencion: en la simulacion la fila 0 es el piso (y crece hacia arriba),
asi que el frame se voltea verticalmente al renderizar; se aplica una
curva raiz como tone-map para que el penacho diluido sea visible.
"""
import numpy as np
from PIL import Image


def save_frame(density, path, scale=2):
    a = np.clip(np.asarray(density, dtype=np.float32), 0.0, 1.0)
    a = np.sqrt(a)
    a = np.flipud(a)
    h, w = a.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = (a * 255).astype(np.uint8)          # rojo
    img[..., 1] = (a * 30).astype(np.uint8)           # verde tenue
    img[..., 2] = (a * 200).astype(np.uint8)          # azul -> magenta
    im = Image.fromarray(img, mode="RGB")
    if scale != 1:
        im = im.resize((w * scale, h * scale), Image.NEAREST)
    im.save(path)
