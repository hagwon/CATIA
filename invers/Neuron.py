import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    u = 1/(1+np.exp(-x))
    return u

def neuron(x, w, b):
    y = sigmoid(w*x-b)
    return y

x = np.arange(-5,5,0.1)

# Red Line: w=1, b=1
y = neuron(x,1,1)
plt.plot(x,y, color='r', label='w:1, b:1')

# Blue Line: w=0.5, b=1
y = neuron(x,0.5,1)
plt.plot(x,y, color='b', label='w:0.5, b:1')

# Green Line: w=2, b=-1
y = neuron(x,2,-1)
plt.plot(x,y, color='g', label='w:2, b:-1')

# Orange Line: three Input & bias
y = sigmoid(1*x + 0.5*x + 2*x -(1+1-1))
plt.plot(x,y, color='orange', label='three Input')

plt.ylim(-0.5,1.5)
plt.legend(loc='upper left')
plt.show()