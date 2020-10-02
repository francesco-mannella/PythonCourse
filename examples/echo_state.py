import numpy as np
import matplotlib.pyplot as plt

# parameters
n = 100
alpha = 0.2
lbd = 0.00001
stime = 20


# generate innwer weights
W = np.random.randn(n, n)
W = alpha * W/np.abs(np.linalg.eigvals(W)).max()
# generate input weights
input_w = np.ones(n)*(np.random.rand(n) < 0.03)

# the input series is an exponential decay 
input_series = np.exp(-0.2*np.linspace(0, 1, stime))
# the target trajectory is an composition of two gaussians 
target_series = np.exp(-((np.linspace(0, 1, stime) - 0.3)**2)/0.01 ) +  \
        np.exp(-((np.linspace(0, 1, stime) - 0.7)**2)/0.01)
# build the inputs (series)*(weights)        
inputs = input_w.reshape(n, 1) * input_series.reshape(1, stime)

# iterate to generate the activation over time
X = np.zeros([n, stime])
for t in range(1, stime):
    X[:, t] = \
            np.dot(W, X[:, t - 1]) + \
            inputs[:, t]

# Ridge regression to find the readout weights
Y = target_series
w_readout = np.dot(np.linalg.inv(np.dot(X, X.T) - lbd*np.eye(n)), \
        np.dot(X, Y.reshape(stime, 1)))

# compute the activation of the readout unit over time
readout = np.dot(X.T, w_readout)

plt.subplot(311)
plt.plot(X.T)
plt.subplot(312)
plt.imshow(X, aspect="auto")
plt.subplot(313)
plt.plot(target_series, lw=4)
plt.plot(readout)
plt.show()

evals = np.linalg.eigvals(W)
plt.scatter(evals.real, evals.imag)
plt.show()

