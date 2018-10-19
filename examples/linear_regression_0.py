import numpy as np
import matplotlib.pyplot as plt

# ----
# Data

# Data model
coeff, intercept = np.random.uniform(-2, 2, 2)
target_weights = np.hstack([coeff, intercept])

# Sampling from the model 
n = 80
noise = 3.0
x = np.random.uniform(-10, 10, n)
y_target = coeff*x + intercept + \
        noise * np.random.randn(n)


# ----
# Linear regression to find the model parameters
# (we pretend we do not have the model)

# Parameters
eta = 0.0003
epochs = 5000

# Initializations
weights = np.zeros(2)
weight_store = np.zeros([epochs + 1, 2])
weight_store[0] = weights

# Stochastic gradient descent
for epoch in range(epochs):
    
    # initialize gradients      
    delta_weights = np.zeros_like(weights)

    # Update gradients based on inputs
    for t in range(n):
        
        # Current prediction of the input
        x_current = np.hstack([x[t], 1])
        y = np.dot(weights, x_current) 
        
        # gradient
        delta_weights += eta * x_current * (y_target[t] - y)
        
    # Batch update of  weights 
    weights += delta_weights 
    weight_store[epoch + 1, :] = weights  

# Plots

plt.figure(figsize=(4.5, 2))

# First plot: moodel and data
plt.subplot(121, aspect="equal")
# Plot the model (red line)
model_x = np.linspace(-10, 10, n)
plt.plot(model_x, coeff*model_x + intercept, color="red", lw=5, alpha=0.8)
# Plot data (black dots)
plt.scatter(x, y_target, color="black", s=1)
# Plot the current model
# approximation (blue line)
plt.plot(model_x, weights[0]*model_x + weights[1], color="blue")

plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.xlabel("x")
plt.ylabel("y")

# Second plot: weight update
plt.subplot(122, aspect="equal")
# Initial weights
plt.scatter(*np.zeros(2), color="green", s=20)
# Current weights
plt.plot(*weight_store.T, color="black")
# Target weights
plt.scatter(*target_weights, color="red", s=20)

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel("weight 0")
plt.ylabel("weight 1")

plt.tight_layout()

plt.show()
