from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

# parameters
n_neurons = 100
dt = 1.0  # ms
tau = 10.0  # ms
v_r = 0.0   # mV
v_th = 15.0 # mV

sim_time = 500

# arrays
v = np.zeros([n_neurons, sim_time])
s = np.zeros([n_neurons, sim_time])
inps = np.zeros([n_neurons, sim_time])

# fills input array with spikes
for n in range(n_neurons):
    spiketimes = np.random.poisson(range(40, 440, 20))
    inps[n, spiketimes] = 80.0

# compute neurons' activations
for t in range(1, sim_time):
    v[:, t] = v[:, t - 1] + (dt / tau) * (- v[:, t - 1] + inps[:, t -1])
    ths = np.where(v[:, t] >= v_th)
    s[ths, t] = 1
    v[ths, t] = v_r

plt.subplot(311)
plt.imshow(inps, cmap=plt.cm.binary)
plt.subplot(312)
plt.imshow(v)
plt.subplot(313)
plt.imshow(s, cmap=plt.cm.binary)
plt.show()
