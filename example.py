from daicrf import potts_mrf, mrf
import matplotlib.pyplot as plt
import numpy as np


def compare_algorithms():
    x = np.ones((10, 10))
    x[:, 5:] = -1
    x_noisy = x + np.random.normal(0, 1.8, size=x.shape)

    inds = np.arange(x.size).reshape(x.shape).astype(np.int64)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.vstack([horz, vert])

    #unaries = 2 * x_thresh.astype(np.float).ravel() - 1
    unaries = x_noisy.ravel()
    unaries = np.c_[np.exp(-unaries), np.exp(unaries)]
    pairwise = np.exp(np.eye(2) * 4.1)
    # repeat pairwise for each edge
    pairwise = np.repeat(pairwise[np.newaxis, :, :], len(edges), axis=0)
    algorithms = ["maxprod", "gibbs", "jt", "trw", "treeep"]
    fix, axes = plt.subplots(1, len(algorithms))
    for ax, alg in zip(axes, algorithms):
        result_mrf = mrf(unaries, edges, pairwise, verbose=1, alg=alg)
        ax.matshow(result_mrf.reshape(x.shape))
        ax.set_title(alg)
        ax.set_xticks(())
        ax.set_yticks(())
    plt.show()


def example_binary():
    x = np.ones((10, 10))
    x[:, 5:] = -1
    x_noisy = x + np.random.normal(0, .8, size=x.shape)
    x_thresh = x_noisy > .0

    fig, axes = plt.subplots(1, 5)
    axes[0].set_title("Original")
    axes[0].matshow(x)
    axes[1].set_title("binary")
    axes[1].matshow(x_noisy)

    axes[2].set_title("thresholded")
    axes[2].matshow(x_thresh)

    inds = np.arange(x.size).reshape(x.shape).astype(np.int64)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.vstack([horz, vert])

    #unaries = 2 * x_thresh.astype(np.float).ravel() - 1
    unaries = x_noisy.ravel()
    unaries = np.c_[np.exp(-unaries), np.exp(unaries)]

    axes[3].set_title("Potts MRF")
    result = potts_mrf(unaries, edges, 1.1, verbose=1)
    axes[3].matshow(result.reshape(x.shape))

    # repeat pairwise for each edge
    pairwise = np.exp(np.eye(2) * 1.1)
    pairwise = np.repeat(pairwise[np.newaxis, :, :], len(edges), axis=0)
    result_mrf = mrf(unaries, edges, pairwise, verbose=1, alg="trw")
    axes[4].set_title("MRF")
    axes[4].matshow(result_mrf.reshape(x.shape))
    for ax in axes:
        ax.set_xticks(())
        ax.set_yticks(())
    plt.show()


def example_multinomial():
    np.random.seed(45)
    unaries = np.zeros((10, 12, 3))
    unaries[:, :4, 0] = 1
    unaries[:, 4:8, 1] = 1
    unaries[:, 8:, 2] = 1
    x = np.argmax(unaries, axis=2)
    unaries_noisy = unaries + np.random.normal(size=unaries.shape)
    x_thresh = np.argmax(unaries_noisy, axis=2)
    unaries_noisy = np.exp(unaries_noisy).reshape(-1, 3)

    inds = np.arange(x.size).reshape(x.shape).astype(np.int64)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.vstack([horz, vert])
    result = potts_mrf(unaries_noisy, edges, 1.1)
    pairwise = np.eye(3) + np.ones((1, 1))
    pairwise[-1, 0] = 0
    pairwise[0, -1] = 0
    print(pairwise)
    # repeat pairwise for each edge
    pairwise = np.repeat(pairwise[np.newaxis, :, :], len(edges), axis=0)
    result_mrf = mrf(unaries_noisy, edges, np.exp(pairwise), alg="jt")
    plot, axes = plt.subplots(1, 4)
    axes[0].set_title("original")
    axes[0].matshow(x)
    axes[1].set_title("thresholded")
    axes[1].matshow(x_thresh)
    axes[1].set_title("Potts MRF")
    axes[2].matshow(result.reshape(x.shape))
    axes[1].set_title("MRF")
    axes[3].matshow(result_mrf.reshape(x.shape))
    for ax in axes:
        ax.set_xticks(())
        ax.set_yticks(())
    plt.show()

if __name__ == "__main__":
    example_binary()
    #example_multinomial()
    #compare_algorithms()
