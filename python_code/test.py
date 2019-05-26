#pylint: skip-file
import os
import numpy as np 
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a
def ReLU(x):
    a = []
    for item in x:
        a.append(np.max(np.array([0,item])))
    return a
def Tanh(x):
    a = []
    for item in x:
        a.append((math.exp(item) - math.exp(-item))/(math.exp(item)+math.exp(-item)))
    return a
x = np.arange(-10., 10., 0.2)
y = Tanh(x)

plt.plot(x,y)

plt.show()
