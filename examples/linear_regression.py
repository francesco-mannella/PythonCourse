import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# Dati

# Il modello dei dati 
coeff, intercept = np.random.uniform(-5, 5, 2)
target_weights = np.hstack([coeff, intercept])

# Genera i dati a partire dal modello 
n = 200
noise = 3.0
x = np.random.uniform(-10, 10, n)
y_target = coeff*x + intercept + \
        noise * np.random.randn(n)

# ------------------------------------------------
# Esegue la regressione per trovare i pesi del
# modello se abbiamo solo x e y_target

# Parameteri
eta = 0.00015
epochs = 200

# Inizializzazioni
weights = np.zeros(2)
weight_store = np.zeros([epochs + 1, 2])
weight_store[0] = weights

# Stochastic gradient descent
for epoch in range(epochs):
    
    # inizializza i gradienti      
    delta_weights = np.zeros_like(weights)

    # Aggiorna i pesi in base agli input
    for t in range(n):
        
        # Predizione del modello corrente in 
        # base all'input
        x_current = np.hstack([x[t], 1])
        y = np.dot(weights, x_current) 
        
        # Calcola il gradiente
        delta_weights += eta * x_current \
                * (y_target[t] - y)
        
    # Aggiorna i pesi
    weights += delta_weights 
    weight_store[epoch + 1, :] = weights  


# grafici

# Primo grafico: dati e modello
plt.subplot(121, aspect="equal")
# Plotta il modello (linea rossa)
model_x = np.linspace(-10, 10, n)
plt.plot(model_x, coeff*model_x + intercept,
        color="red")
# Plotta i dati (punti)
plt.scatter(x, y_target, 
        color="black", s=5)
# Plotta l'approssimazione corrente 
# del modello (linea blu)
plt.plot(model_x, weights[0]*model_x +
        weights[1], color="blue")

plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xlabel("x")
plt.ylabel("y")

# Secondo grafico: aggiornamento dei pesi
plt.subplot(122, aspect="equal")
# Stato iniziale dei pesi
plt.scatter(*weights, color="green", s=20)
# Stato corrente dei pesi
plt.plot(*weights.T, color="black")
# Pesi target
plt.scatter(*target_weights, 
        color="red", s=20)

plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.xlabel("weight 0")
ax2.ylabel("weight 1")

plt.tight_layout()

raw_input("Simulazione terminata!")

