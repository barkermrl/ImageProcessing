from cued_sf2_lab.dct import dct, idct, dctbpp, regroup
from cued_sf2_lab.familiarisation import (
    load_mat_img,
    plot_image,
    optimise_stepsize,
    rms_error,
    plot_grid,
)
from cued_sf2_lab.laplacian_pyramid import quantise
import matplotlib.pyplot as plt

plt.style.use("graphs.mplstyle")

X_pre_zero_mean, cmaps_dict = load_mat_img(
    img="lighthouse.mat", img_info="X", cmap_info={"map", "map2"}
)
X = X_pre_zero_mean - 128.0

N = 8

Y = dct(X, N)
Yr = regroup(Y, N)

Xq = quantise(X, 17)
target_rms = rms_error(X, Xq)

res = optimise_stepsize(lambda stepsize: idct(quantise(Y, stepsize), N), X, target_rms)

Yq = quantise(Y, res.x)
nbits = dctbpp(regroup(Yq, N), N)

Z = idct(Yq, N)

plot_grid(
    [
        (X, "Original image X"),
        (Yr, "Regrouped DCT Yr"),
        (Xq, f"Quantised Xq\n(Stepsize: {17}, RMS error: {target_rms:.2f})"),
        (Z, f"DCT compressed Z\n(Stepsize: {res.x:.2f}, RMS error: {target_rms:.2f})"),
    ],
    nrows=2,
    ncols=2,
    sharex=True,
    sharey=True,
)
