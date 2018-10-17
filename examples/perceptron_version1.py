from __future__ import print_function
from __future__ import division


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
eta = 1.0

weights = [0, 0] 

def step_fun(x):

    if x > 0:
        return 1
    else:
        return 0

for epoch in range(epochs):
    for item in data:
        inp, lab = item

        # potential
        pot = 0
        for i, x in enumerate(inp):
            pot += x * weights[i]
        
        # activation
        act =  step_fun(pot)

        # learn
        for i, x in  enumerate(inp):
            weights[i] += eta * x * (lab - act)
        

        print(lab, act, weights)
        print("")
    
