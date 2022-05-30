from cued_sf2_lab.dwt import nlevdwt, nlevidwt, quantdwt
from cued_sf2_lab.familiarisation import load_mat_img, plot_grid
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("graphs.mplstyle")

X, _ = load_mat_img(img="lighthouse.mat", img_info="X", cmap_info={"map", "map2"})
X = X - 128.0

n = 4

Y = nlevdwt(X, n)

dwtstep = np.ones((3,n+1))*10
Yq, dwtenc = quantdwt(Y, dwtstep)

Z = nlevidwt(Yq, n)

plot_grid(
    [
        (X, "Original image X"),
        (Y, "DWT Y"),
        (Yq, f"Quantised Yq"),
        (Z, f"DCT compressed Z"),
    ],
    nrows=2,
    ncols=2,
    sharex=True,
    sharey=True,
)