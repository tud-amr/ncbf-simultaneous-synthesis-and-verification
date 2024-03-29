from typing import Tuple, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

import numpy as np

from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem

class Car(ControlAffineSystem):
    # Number of states and controls
    N_DIMS = 2
    N_CONTROLS = 1
    N_DISTURBANCE = 0

    # State indices
    X = 0
    X_DOT = 1
    # Control indices
    U = 0

    def __init__(self, ns=2, nu=1, nd=0, dt=0.01):
        super().__init__(ns, nu, nd, dt)
        # self.K_lqr = torch.tensor([[1.0, 2.23606]]).float()

        self.c = 0.2
    def f(self, s):
        batch_size = s.shape[0]
        f = torch.zeros((batch_size, self.ns, 1))
        f = f.type_as(s)
        
        f[:, Car.X, 0] = s[:, Car.X_DOT]
        f[:, Car.X_DOT, 0] = - self.c * s[:, Car.X_DOT]

        
        return f.squeeze(dim=-1)

    def g(self, s, m=0.5):
        
        batch_size = s.shape[0]
        g = torch.zeros((batch_size, self.ns, self.nu))
        g = g.type_as(s)


        # Effect on theta dot
        g[:, Car.X_DOT, Car.U] = 1.0 
        
        return g
    
    def d(self, s):
        return torch.zeros((s.shape[0], self.ns, self.nd), dtype=torch.float).to(s.device)

    def range_dxdt(self, x_l: torch.Tensor, x_u:torch.Tensor,  u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the range of dsdt(x,u) for all s in the batch.

        args:
            x_l, x_u: a tensor of (batch_size, self.n_dims) points in the state space
            u: a tensor of (batch_size, self.n_controls) points in the control space
        returns:
            a tuple (lower, upper) of tensors of (batch_size, self.n_dims) points
            giving the lower and upper bounds on dxdt(x,u) for all x in the batch.
        """

        print(f"x_l shape is {x_l.shape}")
        print(f"x_u is {x_u.shape}")
        pass
    

    
if __name__ == "__main__":
    car = Car(ns=2, nu=1)

    domain_lower_bd = torch.Tensor([-2, -2]).float()
    domain_upper_bd = -domain_lower_bd

    control_lower_bd =torch.Tensor([-1]).float()
    control_upper_bd = -control_lower_bd
        
    def rou(s: torch.Tensor) -> torch.Tensor:
        rou_1 = torch.unsqueeze(s[:, 0] + 1, dim=1)
        rou_2 = torch.unsqueeze( - s[:, 0] + 1, dim=1)
        rou_3 = torch.unsqueeze(s[:, 1] + 1, dim=1)
        rou_4 = torch.unsqueeze( -s[:, 1] + 1, dim=1)
        return torch.hstack( (rou_1, rou_2, rou_3, rou_4) ) 

    def rou_n(s: torch.Tensor) -> torch.Tensor:
        s_norm = torch.norm(s, dim=1, keepdim=True)

        return - s_norm + 0.6

    car.set_domain_limits(domain_lower_bd, domain_upper_bd)
    car.set_control_limits(control_lower_bd, control_upper_bd)
    car.set_state_constraints(rou)
    car.set_nominal_state_constraints(rou_n)

    

    x = torch.rand(3,2, dtype=torch.float)
    u_ref = torch.rand(3, 1, dtype=torch.float)
    
    f = car.f(x)
    print(f"the shape of f is {f.shape} \n f is {f} \n ")
    g = car.g(x)
    print(f"the shape of g is {g.shape} \n g is {g} \n ")

    dsdt =car.dsdt(x, u_ref)
    print(f"the shape of dsdt is {dsdt.shape} \n dsdt is {dsdt} \n ")

    x_next = car.step(x, u_ref)
    print(f"x is {x} \n")
    print(f"x_nest is {x_next}")

    print(f"the K is {car.K}, the shape is {car.K.shape}")
    print(f"u= - K*s is { -car.K @ x[0,:]} ")