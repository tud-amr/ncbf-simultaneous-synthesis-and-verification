
from safe_rl_cbf.Models.NeuralCBF import *
from Dynamics.Car import Car
from Dataset.DataModule import DataModule
from Dynamics.dynamic_system_instances import car1
import torch

import matplotlib.pyplot as plt 


NN = torch.load("NN.pt")

car1.set_barrier_function(NN)
car1.dt = 0.01

s0 = torch.tensor([0.5, -0.3], dtype=torch.float).reshape((1,2))

t = 0
dt = car1.dt
t_record = []
s_record = []
u_record = []
u_ref_record = []
h_record = []

N_steps = 1000

s = s0
hs_0 =NN(s)
print(hs_0)
# wait = input("press something to continue")
try:
    for i in range(N_steps):
        
        hs = NN(s)
        assert hs >= 0.0, f"not safe state"
        h_record.append(hs.item())

        # u_ref = torch.rand(1,1, dtype=torch.float)*4 - torch.tensor([2], dtype=torch.float).reshape((1,1))
        u_ref = torch.tensor([2], dtype=torch.float).reshape((1,1))
        u_result, r_result = car1.h.solve_CLF_QP(s, u_ref, epsilon=1)
        assert not(r_result > 0.0), f"not feasible!!!!!!"
        # if u_result == 1.0:
        #     car1.solve_CLF_QP(s, u_ref)
        s_next = car1.step(s, u_result)

        t = t + dt
        s = s_next

        s_record.append(s)
        u_ref_record.append(u_ref)
        u_record.append(u_result)
        t_record.append(t)
except:
    print("error!!!")


t_record = torch.tensor(t_record)
s_record = torch.vstack(s_record)
u_record = torch.tensor(u_record)
u_ref_record = torch.tensor(u_ref_record)
h_record = torch.tensor(h_record)

plt.figure()

plt.subplot(4,1,1)
plt.plot(t_record, s_record[:, 0])
plt.title("t-x")

plt.subplot(4,1,2)
plt.plot(t_record, s_record[:, 1])
plt.title("t-v")

plt.subplot(4,1,3)
plt.plot(t_record, u_record, linestyle='solid',color='red', label="actual")
plt.plot(t_record, u_ref_record, linestyle=(0, (5,10)) ,color='blue', label="ref")
plt.legend()


plt.subplot(4,1,4)
plt.plot(t_record, h_record, linestyle='solid',color='red', label="actual")

plt.figure()
plt.plot(s_record[:, 0], s_record[:, 1])

plt.show()    



