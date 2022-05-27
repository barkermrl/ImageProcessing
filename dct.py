from cued_sf2_lab.dct import dct, idct, dctbpp, regroup
from cued_sf2_lab.familiarisation import (
    load_mat_img,
    plot_image,
    optimise_stepsize,
    rms_error,
)
from cued_sf2_lab.laplacian_pyramid import bpp, quantise
import matplotlib.pyplot as plt
import numpy as np

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

Yq = quantise(Y,res.x)
nbits = dctbpp(regroup(Yq,N), N)

Z = idct(Yq, N)

fig, axs = plt.subplots(2,2,sharex=True, sharey=True)

plot_image(X, ax=axs[0,0])
axs[0,0].set_title('Original image X')

plot_image(Yr, ax=axs[0,1])
axs[0,1].set_title('Regrouped DCT Yr')

plot_image(Xq, ax=axs[1,0])
axs[1,0].set_title(f'Quantised Xq\n(Stepsize: {17}, RMS error: {target_rms:.2f})')

plot_image(Z, ax=axs[1,1])
axs[1,1].set_title(f'DCT compressed Z\n(Stepsize: {res.x:.2f}, RMS error: {target_rms:.2f})')

plt.show()

