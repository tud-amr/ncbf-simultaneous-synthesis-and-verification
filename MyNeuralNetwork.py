import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from matplotlib import cm
from matplotlib.ticker import LinearLocator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

############### define dynamics #######################
A = torch.tensor([[0, 1],[0, 0]], dtype=torch.float).to(device)
def f(s):
    result = A @ s.T 
    return result.T

def g(s, m=0.5):
    result =  torch.tensor([0, m], dtype=torch.float).reshape((1,2)).to(device) * torch.ones((s.shape[0], 2), dtype=torch.float).to(device)
    return result




############### exam dynamic model ##################
# dt = 0.01
# t = 0
# s0 = torch.tensor([0,0], dtype=torch.float).reshape((2,1))
# s = s0

# s_record = []
# u_record = []
# t_record = []

# for k in range(1000):
#     u = torch.tensor([0.3], dtype=torch.float).reshape((1,1))
#     ds = f(s) + g(s) @ u
#     s = s+ ds * dt
#     t = t+dt

#     s_record.append(s)
#     u_record.append(u)
#     t_record.append(t)

# t_record = torch.tensor(t_record)
# s_record = torch.hstack(s_record)
# u_record = torch.tensor(u_record)

# plt.figure()
# plt.plot(t_record, s_record[0, :])
# plt.show()    


################### define constraints functions#######################


u_v1 = -1
u_v2 = 1
A1 = torch.tensor([1,0], dtype=torch.float).reshape((1,2)).to(device)
A2 = torch.tensor([0,1], dtype=torch.float).reshape((1,2)).to(device)

def l1(s):
    result = A1 @ s.T + 1
    return result.T


def l2(s):
    result = -A1 @ s.T + 1
    return result.T

def l3(s):
    result = A2 @ s.T + 1
    return result.T

def l4(s):
    result = - A2 @ s.T + 1
    return result.T

def h_bar(s, nc=4, r=-10):
    w_i = 1/nc
    sum = w_i*( torch.exp(r*l1(s)) + torch.exp(r*l2(s)) + torch.exp(r*l3(s)) + torch.exp(r*l4(s)) ) + torch.finfo(torch.float).tiny
    return 1/r * torch.log(sum)


####################### show h_bar #########################
# X_test = []
# for i in np.arange(-0.2, 0.2, 0.01):
#     for j in np.arange(-0.2,0.2,0.01):
#         X_test.append(np.array([i, j],dtype=np.float32).reshape((2,1)))
# X_test = np.hstack(X_test).T
# X_test = torch.from_numpy(X_test).float().to(device)
# print(X_test.shape)

# y_test = torch.ones((1, X_test.shape[0]), dtype=torch.float).to(device)

# Z = h_bar(X_test)
# print(Z.shape)
# x = X_test[:,0].cpu().numpy()
# y = X_test[:,1].cpu().numpy()
# z = Z.cpu().numpy().T
# print(x.shape)
# print(y.shape)
# print(z.shape)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# sca = ax.scatter(x,y,z)

# X = np.arange(-0.2, 0.2, 0.1)
# Y = np.arange(-0.2, 0.2, 0.1)
# X,Y = np.meshgrid(X,Y)
# Z = 0 * np.sin(X + Y)
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, facecolors= cm.jet(np.ones((Z.shape[0], Z.shape[1]))) )
# plt.show()


######################### define neural network #########################


class NeuralNetwork(nn.Module):
    def __init__(self, h_bar):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.V = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        self.beta = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.h_bar = h_bar

    def forward(self, s):
        Vs = self.V(s)
        h_bar_s = self.h_bar(s)
        beta_s = self.beta(s)
        # print(Vs.shape)
        # print(h_bar_s.shape)
        # print(beta_s.shape)
        
        # print(torch.norm(Vs, dim=1).reshape(-1,1))
        # print(torch.norm(beta_s, dim=1).reshape(-1,1))

        return torch.norm(Vs, dim=1).reshape(-1,1) * h_bar_s  - torch.norm(beta_s, dim=1).reshape(-1,1)


# h = NeuralNetwork(h_bar=h_bar).to(device)
# s0 = torch.tensor([-1.8545, 0.2772], dtype=torch.float).reshape((1,2)).to(device)
# s0.requires_grad_(True)
# output = h(s0)
# print(output)

#################### define CBF conditions #####################
def get_dhdx(s, h):
    output = h(s)
    output.backward(retain_graph=True)
    dhdx = s.grad

    return dhdx

def c(s, u_vi, dhdx, h, alpha=0.5):
    output = h(s)
    
    return dhdx @ f(s).T + dhdx @ g(s).T * u_vi + alpha * output


def C_bar(s, h, u_v1, u_v2, dhdx, nv=2, r=10):
    w_i = 1/nv
    sum = w_i*( torch.exp(r*c(s, u_v1, dhdx, h)) + torch.exp(r*c(s, u_v2, dhdx, h)) ) + torch.finfo(torch.float).tiny
    return 1/r * torch.log(sum)

def myCustomLoss(h_bar_s, C_bar_s, epsilon = 0.01):
    
    return (torch.sgn(h_bar_s) + 1)/2 * torch.relu(-C_bar_s+epsilon)



# dhdx = get_dhdx(s0, h)

# C_bar_s = C_bar(s0, h, u_v1, u_v2, dhdx)
# h_bar_s = h_bar(s0)


# loss = myCustomLoss(h_bar_s, C_bar_s)
# loss.backward()

# print(s0.grad)


