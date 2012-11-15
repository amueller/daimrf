import Image
import matplotlib.pyplot as plt
import numpy as np
from time import time

from gco_python import cut_simple
from daicrf import mrf
from pyqpbo import alpha_expansion_grid

from IPython.core.debugger import Tracer
tracer = Tracer()


def stereo_unaries(img1, img2):
    differences = []
    max_disp = 8
    for disp in np.arange(max_disp):
        if disp == 0:
            diff = np.sum((img1 - img2) ** 2, axis=2)
        else:
            diff = np.sum((img1[:, 2 * disp:, :] - img2[:, :-2 * disp, :]) **
                    2, axis=2)
        if disp != max_disp - 1:
            diff = diff[:, max_disp - disp - 1:disp - max_disp + 1]
        differences.append(diff)
    return np.dstack(differences).copy("C")


def example():
    img1 = np.asarray(Image.open("scene1.row3.col1.ppm")) / 255.
    img2 = np.asarray(Image.open("scene1.row3.col2.ppm")) / 255.
    img1 = img1[120:180, 80:180]
    img2 = img2[120:180, 80:180]
    unaries = (stereo_unaries(img1, img2) * 100).astype(np.int32)
    n_disps = unaries.shape[2]

    newshape = unaries.shape[:2]
    start = time()
    potts_cut = cut_simple(unaries, -5 * np.eye(n_disps, dtype=np.int32))
    time_gc = time() - start

    start = time()
    qpbo = alpha_expansion_grid(unaries, -5 * np.eye(n_disps, dtype=np.int32))
    time_qpbo = time() - start

    x, y = np.ogrid[:n_disps, :n_disps]
    one_d_topology = np.abs(x - y).astype(np.int32).copy("C")

    start = time()
    one_d_cut = cut_simple(unaries, 5 * one_d_topology)
    time_gc1d = time() - start

    # build edges for max product inference:
    inds = np.arange(np.prod(newshape)).reshape(newshape).astype(np.int64)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.vstack([horz, vert]).copy()
    pairwise = np.exp(5 * np.eye(n_disps))
    #asdf = np.random.permutation(len(edges))
    #edges = edges[asdf]
    start = time()
    max_product = mrf(np.exp(-unaries.reshape(-1, n_disps)), edges, pairwise, alg='maxprod')
    time_maxprod = time() - start
    start = time()
    trw = mrf(np.exp(-unaries.reshape(-1, n_disps)), edges, pairwise, alg='trw')
    time_trw = time() - start

    #start = time()
    #treeep = mrf(np.exp(-unaries.reshape(-1, n_disps)), edges, pairwise, alg='treeep')
    #time_treeep = time() - start

    start = time()
    gibbs = mrf(np.exp(-unaries.reshape(-1, n_disps)), edges, pairwise, alg='gibbs')
    time_gibbs = time() - start

    fix, axes = plt.subplots(3, 3, figsize=(16, 8))


    axes[0, 0].imshow(img1)
    axes[0, 1].imshow(img2)
    axes[0, 2].matshow(np.argmin(unaries, axis=2), vmin=0, vmax=8)
    axes[1, 0].set_title("gc %.2f" % time_gc)
    axes[1, 0].matshow(potts_cut.reshape(newshape), vmin=0, vmax=8)
    axes[1, 1].set_title("gc 1d %.2f" % time_gc1d)
    axes[1, 1].matshow(one_d_cut.reshape(newshape), vmin=0, vmax=8)
    axes[1, 2].set_title("mp %.2f" % time_maxprod)
    axes[1, 2].matshow(max_product.reshape(newshape), vmin=0, vmax=8)
    axes[2, 0].set_title("trw %.2f" % time_trw)
    axes[2, 0].matshow(trw.reshape(newshape), vmin=0, vmax=8)
    axes[2, 1].set_title("qpbo %f" % time_qpbo)
    axes[2, 1].matshow(qpbo.reshape(newshape), vmin=0, vmax=8)
    axes[2, 2].set_title("gibbs %.2f" % time_gibbs)
    axes[2, 2].matshow(gibbs.reshape(newshape), vmin=0, vmax=8)
    for ax in axes.ravel():
        ax.set_xticks(())
        ax.set_yticks(())
    plt.tight_layout()
    plt.show()

example()
