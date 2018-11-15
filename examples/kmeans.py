# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

# ------------------------------------------------------------------------------
# Dati

# genera i dati a partire da 'num_clusters' centroidi

num_clusters = 5
num_patterns = 200*num_clusters
epoche = 30
num_cluster_patterns = num_patterns//num_clusters

# genera i centroidi iniziali
centroids = np.random.randn(num_clusters, 2) 
# genera i centroidi target
centroids_target = np.random.uniform(-7, 7, [num_clusters, 2])
# deviazioni standard dai centroidi
std = 3.0
# genera i dati a partire dai centroidi target 
# e le deviazioni standard
data = np.vstack([
    std*np.random.randn(num_cluster_patterns,  2) + 
centroids_target[i].reshape(                   1,  2) 
    for i in range(num_clusters)])

# ------------------------------------------------------------------------------
# Plotting: inizializza lo scatterplot dei dati

plt.figure(figsize=(10, 10))
plt.subplot(111, aspect="equal")
color_indices = np.linspace(0, 1, num_clusters)
colors = plt.cm.gist_rainbow(color_indices)

plot_clusters = []
for i in range(num_clusters):
    curr_data = data[i*num_cluster_patterns:(i+1)*num_cluster_patterns]
    plot = plt.scatter(*curr_data.T, s=100, color = colors[i], alpha=0.2)
    plot_clusters.append(plot)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.draw()
plot_centroids = plt.scatter(*centroids.T, s=200, c=colors)
plot_epoche = plt.text(8, 8, "", fontsize=30)

# ------------------------------------------------------------------------------
# Itera per il numero di epoche 

for epoca in range(epoche):
    
    
    # calcola le distanze dei centroidi dai dati
    norms = np.linalg.norm(
                 data.reshape(num_patterns,            1,  2) - 
            centroids.reshape(           1, num_clusters,  2),  axis=2)
    
    # calcola la distanza minima per ogni pattern dei dati 
    # (ritorna l'indice del centroide con distanza minima)
    nk = np.argmin(norms, 1)
    
    # aggiorna i plot dei dati
    for i in range(num_clusters):
        plot_clusters[i].set_offsets(data[nk==i])
    # aggiorna il plot dei centroidi 
    plot_centroids.set_offsets(centroids)
    # aggiorna il plot del contatore delle epoche
    plot_epoche.set_text("%d" % epoca)    
   
    # ricalcola i centroidi facendo la media dei dati per ogni gruppo
    for i in range(num_clusters):
        if (nk==i).sum() > 0:
            centroids[i] = data[nk==i].mean(0) 
    
    plt.pause(0.005) 


raw_input()

