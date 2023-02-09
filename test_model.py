import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from MyNeuralNetwork import *
from CARs import car1
from DataModule import DataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

data_module = DataModule(system=car1, training_sample_num=50000)

NN = torch.load("NN.pt")
NN.to(device)

################ test satisfaction_rate ########################
# loss_record = []
# sample_count = 0
# correct_point_count = 0
# for i in np.arange(-2, 2, 0.1):
#     for j in np.arange(-1,1,0.1):
#         sample_count += 1
#         X_test = torch.tensor([[i, j]], dtype=torch.float, device=device)
        
#         hs = h(X_test)

#         if (hs >= 0 and X_test.norm(dim=-1) <= 1) or (hs < 0 and X_test.norm(dim=-1) >= 1.2):
#             correct_point_count += 1
        


# success_rate = correct_point_count/sample_count
# print(f"average success rate is : {success_rate}")

################## single point test #######################
# X_test = torch.rand(5, 2, dtype=torch.float, device=device)
# X_test = torch.tensor([1.099, 0.0], dtype=torch.float).reshape((1,2)).to(device)
# print(h(X_test))

################### draw CBF ################################

x = np.arange(-2, 2, 0.1)
u = np.arange(-2, 2, 0.1)

X, U = np.meshgrid(x, u)

print(X.shape)
print(U.shape)

H = []
V = []
with torch.no_grad():
    for col in range(X.shape[1]):
        x_c = X[:, col].reshape((-1, 1))
        u_c = U[:, col].reshape((-1, 1))
        
        s_c = np.hstack((x_c, u_c))
        s_c_tensor = torch.from_numpy(s_c).float().to(device)
        h_s_c_gpu, v_s_c_gpu = NN(s_c_tensor)
        h_s_c = h_s_c_gpu.cpu().numpy()
        v_s_c = v_s_c_gpu.cpu().numpy()
        H.append(h_s_c)
        V.append(v_s_c)



H = np.hstack(H)
V = np.hstack(V)

H = (H >= 0)

fig1,ax1=plt.subplots(1,1)
cp = ax1.contourf(X, U, H)
fig1.colorbar(cp) # Add a colorbar to a plot
ax1.set_title('Filled Contours Plot')
ax1.set_xlabel('x')
ax1.set_ylabel('u')
plt.show()

# fig = plt.figure()
# ax2 = plt.axes(projection='3d')
# ax2.contour3D(X, U, V, 50, cmap='binary')
# ax2.set_xlabel('x')
# ax2.set_ylabel('u')
# ax2.set_zlabel('v')
# plt.show()




