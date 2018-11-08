from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

"""
Simulating an array of 100 leaky-integrate-and-fire (LIF) neurons.
Each neuron receives a train of poissonian inpulses.  
"""

# Parameters
n_neurons = 100
dt = 1.0  # ms
tau = 20.0  # ms
v_r = -70.0   # mV
v_th = 5.0 # mV

sim_time = 500

# Arrays
v = np.zeros([n_neurons, sim_time])
s = np.zeros([n_neurons, sim_time])
inps = np.zeros([n_neurons, sim_time])

# Fills input array with spikes
for n in range(n_neurons):
    # Create the train of inpulses for each neuron.
    # We give to the np.random.poisson a uniform 
    # sequence of timesteps
    spiketimes = np.random.poisson(range(40, 440, 15))
    inps[n, spiketimes] = 60.0

# Compute neurons' activations
for t in range(1, sim_time):
    # the vecctor of all neurons at current itmesteps is updated based 
    # on the vector of previous potentials and the vector of inputs
    v[:, t] = v[:, t - 1] + (dt / tau) * (
            - v[:, t - 1]   # decays based on the previous potentials 
            + inps[:, t] )   # current inpulses

    ths = np.where(v[:, t] >= v_th)    # select neurons whose potentials are over threshold
    s[ths, t] = 1    # Record spikes for neurons with potential over threshold
    v[ths, t] = v_r    # Reset potential to minimum level
    v[ths, t - 1] = 40.0    # Previous potential is set to spiking level (for graphics) 

plt.subplot(311)
# Plot poisson inputs
plt.title("Inmpulses")
plt.imshow(inps/np.max(inps), cmap=plt.cm.binary)
plt.colorbar()
plt.ylabel("Neurons")
plt.xlabel("Time (ms)")
plt.subplot(312)
# Plot potentials
plt.title("Potentials")
plt.imshow(v)
plt.colorbar()
plt.ylabel("Neurons")
plt.xlabel("Time (ms)")
plt.subplot(313)
# Plot spikes
plt.title("Spikes")
plt.imshow(s, cmap=plt.cm.binary)
plt.colorbar()
plt.ylabel("Neurons")
plt.xlabel("Time (ms)")
plt.tight_layout()
plt.show()
# Plot neuron 50 as an example 
plt.title("neuron 50 - potential")
plt.plot(range(sim_time), v[50,:])
plt.show()
