from typing import Tuple, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

import numpy as np

from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem

class Car(ControlAffineSystem):
    def __init__(self, ns=2, nu=1, dt=0.01):
        super().__init__(ns, nu, dt)
        self.K_lqr = torch.tensor([[1.0, 2.23606]]).float()

        
    def f(self, s):
        batch_size = s.shape[0]

        a = torch.tensor([[0, 1],[0, 0]], dtype=torch.float).to(s.device)
        A = [ a.unsqueeze(dim=0) for i in range(batch_size) ]
        A = torch.vstack(A)

        result = torch.bmm(A, s.unsqueeze(dim=-1))
        
        return result.squeeze(dim=-1)

    def g(self, s, m=0.5):
        old_result =  torch.tensor([0, m], dtype=torch.float).reshape((1,2)).to(s.device) * torch.ones((s.shape[0], 2 ), dtype=torch.float).to(s.device)
        result =  torch.tensor([0, m], dtype=torch.float).reshape((self.ns, self.nu)).unsqueeze(dim=0).to(s.device) * torch.ones((s.shape[0], self.ns, self.nu), dtype=torch.float).to(s.device)
        return result

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
        s_norm = torch.norm(s, dim=1)

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