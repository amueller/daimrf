from daicrf import crf_map
import matplotlib.pyplot as plt
import numpy as np

x = np.ones((10, 10))
x[:, 5:] = -1
x_noisy = x + np.random.normal(0, 0.8, size=x.shape)
x_thresh = x_noisy > .0
plt.matshow(x)
plt.matshow(x_noisy)
plt.matshow(x_thresh)

inds = np.arange(x.size).reshape(x.shape)
horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]

edges = np.vstack([horz, vert])

#unaries = 2 * x_thresh.astype(np.float).ravel() - 1
unaries = x_noisy.ravel()
unaries = np.c_[np.exp(-unaries), np.exp(unaries)]

result = crf_map(unaries, edges, 1.1)
plt.matshow(result.reshape(x.shape))
plt.show()
