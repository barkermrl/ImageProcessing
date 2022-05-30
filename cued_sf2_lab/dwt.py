# TODO complete this!
import numpy as np
from typing import Tuple
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.laplacian_pyramid import quantise

from .laplacian_pyramid import rowdec, rowdec2, rowint, rowint2

__all__ = ["h1", "h2", "g1", "g2", "dwt", "idwt"]

h1 = np.array([-1, 2, 6, 2, -1]) / 8
h2 = np.array([-1, 2, -1]) / 4

g1 = np.array([1, 2, 1]) / 2
g2 = np.array([-1, -2, 6, -2, -1]) / 4


def dwt(X: np.ndarray, h1: np.ndarray = h1, h2: np.ndarray = h2) -> np.ndarray:
    """
    Return a 1-level 2-D discrete wavelet transform of X.

    Default h1 and h2 are the LeGall filter pair.

    Parameters:
        X: Image matrix (Usually 256x256)
        h1, h2: Filter coefficients
    Returns:
        Y: 1-level 2D DWT of X
    """
    m, n = X.shape
    if m % 2 or n % 2:
        raise ValueError("Image dimensions must be even")
    Y = np.concatenate([rowdec(X, h1), rowdec2(X, h2)], axis=1)
    Y = np.concatenate([rowdec(Y.T, h1).T, rowdec2(Y.T, h2).T], axis=0)
    return Y


def idwt(X: np.ndarray, g1: np.ndarray = g1, g2: np.ndarray = g2) -> np.ndarray:
    """
    Return a 1-level 2-D inverse discrete wavelet transform on X.

    If filters G1 and G2 are given, then they are used, otherwise the LeGall
    filter pair are used.
    """
    m, n = X.shape
    if m % 2 or n % 2:
        raise ValueError("Image dimensions must be even")
    m2 = m // 2
    n2 = n // 2
    Y = rowint(X[:m2, :].T, g1).T + rowint2(X[m2:, :].T, g2).T
    Y = rowint(Y[:, :n2], g1) + rowint2(Y[:, n2:], g2)
    return Y


def nlevdwt(X, n):
    m = 256
    Y = dwt(X)
    for i in range(1, n):
        m = m // 2
        Y[:m, :m] = dwt(Y[:m, :m])
    return Y


def nlevidwt(Y, n):
    Xr = Y.copy()
    m = 256 // 2**n
    for i in range(n):
        m *= 2
        Xr[:m, :m] = idwt(Xr[:m, :m])
    return Xr


def quantdwt(Y: np.ndarray, dwtstep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    Yq = Y.copy()
    n = dwtstep.shape[1] - 1

    m = 256
    dwtenc = np.zeros(dwtstep.shape)

    for i in range(n):
        m = int(m / 2)
        size = m**2
        # top right
        Yq[:m, m : 2 * m] = quantise(Yq[:m, m : 2 * m], dwtstep[0, i])
        dwtenc[0, i] = bpp(Yq[:m, m : 2 * m]) * size

        # bottom left
        Yq[m : 2 * m, :m] = quantise(Yq[m : 2 * m, :m], dwtstep[1, i])
        dwtenc[1, i] = bpp(Yq[m : 2 * m, :m]) * size

        # bottom right
        Yq[m : 2 * m, m : 2 * m] = quantise(Yq[m : 2 * m, m : 2 * m], dwtstep[2, i])
        dwtenc[2, i] = bpp(Yq[m : 2 * m, m : 2 * m]) * size

    Yq[:m, :m] = quantise(Yq[:m, :m], dwtstep[0, n])
    dwtenc[0, n] = bpp(Yq[:m, :m]) * size

    return Yq, dwtenc


def energy_Z(n, layer, i):
    """
    Returns energy of reconstructed image given an impulse
    at layer with image index i.
    """
    Y = np.zeros((256, 256))

    if layer == n:
        if i == 0:
            m = 256 // 2**n
            Y[m // 2, m // 2] = 100
            Z = nlevidwt(Y, n)
            return np.sum(Z**2)
        else:
            return np.inf

    m = 256 // 2 ** (layer + 1)

    if i == 0:
        Y[m // 2, m + m // 2] = 100

    elif i == 1:
        Y[m + m // 2, m // 2] = 100

    elif i == 2:
        Y[m + m // 2, m + m // 2] = 100

    Z = nlevidwt(Y, n)
    return np.sum(Z**2)
