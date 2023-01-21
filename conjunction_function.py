import numpy as np
import matplotlib.pyplot as plt

def l1(x):
    return 1 - x

def l2(x):
    return x + 1


def E(x1,x2, r=-100):
    x1 = np.tanh(x1)
    x2 = np.tanh(x2)

    return 1/r*np.log(0.5*np.exp(r*x1) + 0.5*np.exp(r*x2) )

def h(x):
    return E(l1(x), l2(x))

X = np.arange(-5,5,0.01)
plt.figure()
plt.plot(X, h(X))
plt.show()