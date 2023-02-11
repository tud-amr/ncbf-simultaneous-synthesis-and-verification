import numpy as np
import control

A = np.array([[0, 1.0],[0, 0.0]])
B = np.array([0, 0.5]).reshape((2,1))

Q = np.eye(2)

R = np.eye(1)

K, S, E = control.lqr(A, B, Q, R)
print(K)

s = np.array([-0.2, 1]).reshape((2,1))

u = -K @ s 
print(u)