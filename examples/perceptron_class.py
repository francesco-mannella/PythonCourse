import numpy as np
import matplotlib.pyplot as plt

data = [[[0.44, 0.83], 0],
        [[0.83, 0.66], 1],
        [[0.52, 0.83], 0],
        [[0.84, 0.55], 1],
        [[0.71, 0.92], 0],
        [[0.51, 0.15], 1],
        [[0.24, 0.35], 0],
        [[0.34, 0.43], 0],
        [[0.29, 0.81], 0],
        [[0.66, 0.3 ], 1]]

epochs = 2

def step_fun(x):
    if x > 0:
        return 1
    else:
        return 0

def wsum(vec, weights):
    res = 0
    for i, x in enumerate(vec):
        res += x*weights[i]
    return res

class Perceptron:
    
    def __init__(self, eta, out_fun):

        self.eta = eta
        self.out_fun = out_fun
        self.weights = [0, 0]

    def activation(self, inp):

        pot = wsum(inp, self.weights)
        act = step_fun(pot)

        return act, pot
        
    def learn(self, inp, out, teach):
        for i, x in  enumerate(inp):
            self.weights[i] += self.eta * x * (teach - out)

perc = Perceptron(eta=0.1, out_fun=step_fun)

for epoch in range(epochs):
    for item in data:
        inp, lab = item

        act,_ = perc.activation(inp)
        perc.learn(inp, act, lab)

        print(lab, act, perc.weights)
        print("")
    
