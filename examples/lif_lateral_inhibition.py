from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

"""
Simulating an array of 100 leaky-integrate-and-fire (LIF) neurons.
Each neuron receives a train of poissonian inpulses.
Neurons send lateral excitations to their neighborhoods and lateral inhibitions 
to all the other neurons.
"""

# Parameters
n_neurons =50
dt = 1.0  # ms:
tau = 20.0  # ms
v_r = -70.0   # mV
v_th = 5.0 # mV

w_ex = 20.0
w_in = -400.0

sim_time = 500


# all lower and upper diagonals from 0 to 50 are set to 1
Ex = np.eye(n_neurons, n_neurons) 
for i in range(1,5):
    Ex +=  np.eye(n_neurons, n_neurons, i) 
    Ex +=  np.eye(n_neurons, n_neurons, -i) 

# Mirrow of Ex: all other diagonals are set to 1 
In = 1 - Ex

# Fill weights matrix with lateral excitations and 
# lateral inhibitions
W =  w_ex*Ex  + w_in*In 

# Arrays
v = np.zeros([n_neurons, sim_time])
s = np.zeros([n_neurons, sim_time])
inps = np.zeros([n_neurons, sim_time])

# Fills input array with spikes
for n in range(n_neurons):
    # Create the train of inpulses for each neuron.
    # We give to the np.random.poisson a uniform 
    # sequence of timesteps
    spiketimes = np.random.poisson(range(40, 140, 5) + range(300, 420, 5)  )
    inps[n, spiketimes] = 60.0

# Compute neurons' activations
for t in range(1, sim_time):
    # the vector of all neurons at current itmesteps is updated based 
    # on the vector of previous potentials and the vector of inputs
    v[:, t] = v[:, t - 1] + (dt / tau) * (
            - v[:, t - 1]   # decays based on the previous potentials 
            + inps[:, t]    # current inpulses
            + np.dot(W, s[:, t -1]))   # matrix multiplication between 
                                       # the weights matrix and the
                                       # vector of previous spikes
    
    ths = np.where(v[:, t] >= v_th)   # select neurons whose potentials are over threshold
    s[ths, t] = 1    # Record spikes for neurons with potential over threshold
    v[ths, t] = v_r    # Reset potential to minimum level
    v[ths, t - 1] = 40.0    # Previous potential is set to spiking level (for graphics) 

plt.subplot(311)
# Plot poisson inputs
plt.title("Inmpulses")
plt.imshow(inps/np.max(inps), aspect="auto", cmap=plt.cm.binary)
plt.colorbar()
plt.ylabel("Neurons")
plt.xlabel("Time (ms)")
plt.subplot(312)
# Plot pote
plt.title("Potentials")
plt.imshow(v, aspect="auto")
plt.colorbar()
plt.ylabel("Neurons")
plt.xlabel("Time (ms)")
plt.subplot(313)
# Plot spikes
plt.title("Spikes")
plt.imshow(s,  aspect="auto", cmap=plt.cm.binary)
plt.colorbar()
plt.ylabel("Neurons")
plt.xlabel("Time (ms)")
plt.tight_layout()
plt.show()

# Plot weights 
plt.title("weights")
plt.imshow(W,  aspect="auto")
plt.show()
# Plot neuron 5 as an example 
plt.title("neuron 5 - potential")
spiking = s.sum(1)>2    # select neurons that spike at least two times
plt.plot(range(sim_time), v[spiking][0])   # plot the first of the selected group 
plt.show()
