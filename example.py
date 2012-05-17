from daicrf import potts_crf, mrf
import matplotlib.pyplot as plt
import numpy as np

x = np.ones((10, 10))
x[:, 5:] = -1
x_noisy = x + np.random.normal(0, 0.8, size=x.shape)
x_thresh = x_noisy > .0
plt.subplot(141)
plt.imshow(x, interpolation='nearest')
plt.subplot(142)
plt.imshow(x_noisy, interpolation='nearest')
plt.subplot(143)
plt.imshow(x_thresh, interpolation='nearest')

inds = np.arange(x.size).reshape(x.shape)
horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]

edges = np.vstack([horz, vert])

#unaries = 2 * x_thresh.astype(np.float).ravel() - 1
unaries = x_noisy.ravel()
unaries = np.c_[np.exp(-unaries), np.exp(unaries)]

result = potts_crf(unaries, edges, 1.1)
plt.subplot(144)
plt.imshow(result.reshape(x.shape), interpolation='nearest')

result_mrf = mrf(unaries, edges, np.exp(np.eye(2) * 1.1))
plt.matshow(result_mrf.reshape(x.shape), interpolation='nearest')
plt.show()
