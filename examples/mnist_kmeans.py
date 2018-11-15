# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import mnist

plt.ion()

# ------------------------------------------------------------------------------
# Dati

num_clusters = 20
epoche = 30

data = mnist.train_images()/255.0
num_patterns, pattern_side, _ = data.shape
pattern_len = pattern_side*pattern_side
data = data.reshape(num_patterns, pattern_len)

centroids = np.zeros([num_clusters, pattern_len]) 

# ------------------------------------------------------------------------------
# Plotting: inizializza il plot dei pesi


plot_centroids = []

fig = plt.figure(figsize=(10, 8))
for i in range(num_clusters):

    ax = fig.add_subplot(4, num_clusters/4, i + 1, aspect="equal")
    im = ax.imshow(centroids[i].reshape(pattern_side, pattern_side), 
            cmap=plt.cm.gray, vmin=0, vmax=1)
    ax.set_axis_off()
    plot_centroids.append(im)
plt.tight_layout()

# ------------------------------------------------------------------------------
# Itera per il numero di epoche 

for epoca in range(epoche):
    
    
    # calcola le distanze dei centroidi dai dati
    norms = np.linalg.norm(
                 data.reshape(num_patterns,            1,  pattern_len) - 
            centroids.reshape(           1, num_clusters,  pattern_len),  
            axis=2)
    
    # calcola la distanza minima per ogni pattern dei dati 
    # (ritorna l'indice del centroide con distanza minima)
    nk = np.argmin(norms, 1)
    
    # aggiorna i plot dei centroidi
    for i in range(num_clusters):
        plot_centroids[i].set_data(centroids[i].reshape(
            pattern_side, pattern_side))
   
    # ricalcola i centroidi facendo la media dei dati per ogni gruppo
    for i in range(num_clusters):
        if (nk==i).sum() > 0:
            centroids[i] = data[nk==i].mean(0) 
    
    plt.pause(0.05) 


raw_input()

