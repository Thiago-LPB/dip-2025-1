import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    linhas, colunas = img.shape

    # 1) Translação (deslocar p/ baixo e p/ direita)
    dy, dx = 15, 15
    trans = np.zeros_like(img)
    trans[dy:, dx:] = img[:linhas-dy, :colunas-dx]

    # 2) Rotação 90° sentido horário
    rot = np.zeros((colunas, linhas), dtype=img.dtype)
    for i in range(linhas):
        for j in range(colunas):
            rot[j, linhas-1-i] = img[i, j]

    # 3) Esticar na horizontal (1.5x)
    nova_col = int(colunas * 1.5)
    stretch = np.zeros((linhas, nova_col), dtype=img.dtype)
    for j in range(nova_col):
        orig_j = int(j / 1.5)
        stretch[:, j] = img[:, orig_j]

    # 4) Espelhamento horizontal
    mirror = img[:, ::-1]

    # 5) Distorção
    dist = np.zeros_like(img)
    yy, xx = np.indices((linhas, colunas))
    xn = (2 * xx / colunas) - 1
    yn = (2 * yy / linhas) - 1
    raio = np.sqrt(xn**2 + yn**2)

    k = 0.25
    raio2 = raio * (1 + k * raio**2)

    ang = np.arctan2(yn, xn)
    xn2 = raio2 * np.cos(ang)
    yn2 = raio2 * np.sin(ang)

    x_map = ((xn2 + 1) * colunas / 2).astype(int)
    y_map = ((yn2 + 1) * linhas / 2).astype(int)

    dentro = (x_map >= 0) & (x_map < colunas) & (y_map >= 0) & (y_map < linhas)
    dist[yy[dentro], xx[dentro]] = img[y_map[dentro], x_map[dentro]]

    return {
        "translated": trans,
        "rotated": rot,
        "stretched": stretch,
        "mirrored": mirror,
        "distorted": dist
    }