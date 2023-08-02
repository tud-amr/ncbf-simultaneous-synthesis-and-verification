from typing import Tuple, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

import numpy as np
import control

from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem

class DubinsCar(ControlAffineSystem):
    """
    Represents a damped inverted pendulum.

    The system has state

        s = [x, y, theta]

    representing the angle and velocity of the pendulum, and it
    has control inputs

        u = [v, w]

    representing the torque applied.

    """
    
    # Number of states and controls
    N_DIMS = 3
    N_CONTROLS = 2
    N_DISTURBANCES = 0

    # State indices
    X = 0
    Y = 1
    THETA = 2
    
    # Control indices
    V = 0
    W = 1

    def __init__(self, ns=N_DIMS, nu=N_CONTROLS, nd=N_DISTURBANCES , dt=0.01):
        super().__init__(ns, nu, nd, dt)
        self.period_state_index = [DubinsCar.THETA]


    def f(self, s: torch.Tensor) -> torch.Tensor:
        batch_size = s.shape[0]
        f = torch.zeros((batch_size, self.ns, 1))
        f = f.type_as(s)
        return f.squeeze(dim=-1)
    
    
    def g(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.ns, self.nu))
        g = g.type_as(x)


        # Effect on theta dot
        g[:, DubinsCar.X, DubinsCar.V] = torch.cos(x[:, DubinsCar.THETA])
        g[:, DubinsCar.Y, DubinsCar.V] = torch.sin(x[:, DubinsCar.THETA])
        g[:, DubinsCar.THETA, DubinsCar.W] = 1.0

        return g

    def d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the disturbance-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state

        returns:
            d: bs x self.n_dims x self.n_disturbances tensor
        """
        return torch.zeros((x.shape[0], self.ns, self.nd), dtype=torch.float).to(x.device)


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

        # print(f"x_l is {x_l}")
        # print(f"x_u is {x_u}")

        number_of_samples = 1000
        dxdt_list = []
        for i in range(number_of_samples):
            sample = (x_u - x_l) * torch.rand_like(x_l) + x_l
            # print(f"sample  is {sample}")
            dxdt = self.dsdt(sample, u)
            dxdt_list.append(dxdt)

        dxdt = torch.stack(dxdt_list, dim=0)
        # print(f"dxdt is {dxdt}")
        # print(f"dxdt shape is {dxdt.shape}")

        dxdt_min, _ = torch.min(dxdt, dim=0)
        dxdt_max, _ = torch.max(dxdt, dim=0)
        # print(f"dxdt_min is {dxdt_min}")
        # print(f"dxdt_max is {dxdt_max}")
        return dxdt_min, dxdt_max




if __name__ == "__main__":

    dubins_car = DubinsCar()

    domain_lower_bd = torch.Tensor([-1, -1, -torch.pi]).float()
    domain_upper_bd = torch.Tensor([9, 9, torch.pi]).float()

    control_lower_bd = torch.Tensor([-1, -1]).float()
    control_upper_bd = -control_lower_bd
        
    def rou(s: torch.Tensor) -> torch.Tensor:
        rou_1 = torch.unsqueeze(s[:, 0] + 8, dim=1)
        rou_2 = torch.unsqueeze( - s[:, 0] + 8, dim=1)
        rou_3 = torch.unsqueeze(s[:, 1] + 8, dim=1)
        rou_4 = torch.unsqueeze( -s[:, 1] + 8, dim=1)
        rou_5 = torch.norm(s[:, 0:2] - torch.tensor([5,5]).to(s.device).reshape(1, 2), dim=1, keepdim=True) - 2.8
        print(f"rou_5 is {rou_5}")
        return torch.hstack( (rou_1, rou_2, rou_3, rou_4, rou_5) ) 

    def rou_n(s: torch.Tensor) -> torch.Tensor:
        s_norm = torch.norm(s, dim=1, keepdim=True)

        return - s_norm + 0.6

    dubins_car.set_domain_limits(domain_lower_bd, domain_upper_bd)
    dubins_car.set_control_limits(control_lower_bd, control_upper_bd)
    dubins_car.set_state_constraints(rou)
    dubins_car.set_nominal_state_constraints(rou_n)

    
    x = torch.tensor([5, 5, 0], dtype=torch.float).reshape(1, 3)
    # x = torch.rand(3,3, dtype=torch.float)
    u_ref = torch.rand(1, 2, dtype=torch.float)
    
    f = dubins_car.f(x)
    print(f"the shape of f is {f.shape} \n f is {f} \n ")
    g = dubins_car.g(x)
    print(f"the shape of g is {g.shape} \n g is {g} \n ")

    dsdt =dubins_car.dsdt(x, u_ref)
    print(f"the shape of dsdt is {dsdt.shape} \n dsdt is {dsdt} \n ")

    x_next = dubins_car.step(x, u_ref)
    print(f"x is {x} \n")
    print(f"x_nest is {x_next}")

    rou = dubins_car.state_constraints(x)