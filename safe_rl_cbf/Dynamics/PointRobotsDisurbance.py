from typing import Tuple, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

import numpy as np
import control

from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem

class PointRobotsDisturbance(ControlAffineSystem):
    """
    
    The system has state

        s = [x_e, y_e, v_e_x, v_e_y, x_c, y_c, v_c_x, v_c_y]

    representing the angle and velocity of the pendulum, and it
    has control inputs

        u = [a_x, a_y]

    representing the acceleration applied on ego agent. and the 
    disturbance is
        d = [d_x, d_y]

    """
    
    # Number of states and controls
    N_DIMS = 8
    N_CONTROLS = 2
    N_DISTURBANCES = 2

    # State indices
    XE = 0
    YE = 1
    VEX = 2
    VEY = 3
    XC = 4
    YC = 5
    VCX = 6
    VCY = 7

    # Control indices
    AX = 0
    AY = 1

    # Disturbance indices
    DX = 0
    DY = 1

    def __init__(self, ns=N_DIMS, nu=N_CONTROLS, nd=N_DISTURBANCES , dt=0.01):
        super().__init__(ns, nu, nd, dt)


    def f(self, s: torch.Tensor) -> torch.Tensor:
        batch_size = s.shape[0]
        f = torch.zeros((batch_size, self.ns, 1))
        f = f.type_as(s)


        f[:, PointRobotsDisturbance.XE, 0] = s[:, PointRobotsDisturbance.VEX]
        f[:, PointRobotsDisturbance.YE, 0] = s[:, PointRobotsDisturbance.VEY]
        f[:, PointRobotsDisturbance.XC, 0] = s[:, PointRobotsDisturbance.VCX]
        f[:, PointRobotsDisturbance.YC, 0] = s[:, PointRobotsDisturbance.VCY]

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
        g[:, PointRobotsDisturbance.VEX, PointRobotsDisturbance.AX] = 1
        g[:, PointRobotsDisturbance.VEY, PointRobotsDisturbance.AY] = 1

        return g

    def d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the disturbance-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state

        returns:
            d: bs x self.n_dims x self.n_disturbances tensor
        """
        batch_size = x.shape[0]
        d = torch.zeros((batch_size, self.ns, self.nd))
        d = d.type_as(x)

        d[:, PointRobotsDisturbance.VCX, PointRobotsDisturbance.DX] = 1
        d[:, PointRobotsDisturbance.VCY, PointRobotsDisturbance.DY] = 1

        return d

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

    point_robots_dis = PointRobotsDisturbance()

    domain_lower_bd = torch.Tensor([-1, -1, -1.2, -1.2, -1, -1, -1.2, -1.2]).float()
    domain_upper_bd = torch.Tensor([9, 9, 1.2, 1.2, 9, 9, 1.2, 1.2]).float()

    control_lower_bd = torch.Tensor([-1, -1]).float()
    control_upper_bd = -control_lower_bd

    disturbance_lower_bd = torch.Tensor([-0.5, -0.5]).float()
    disturbance_upper_bd = -disturbance_lower_bd
        
    def rou(s: torch.Tensor) -> torch.Tensor:
        rou_1 = torch.unsqueeze(s[:, 0] - 0.3, dim=1)
        rou_2 = torch.unsqueeze( - s[:, 0] + 7.7, dim=1)
        rou_3 = torch.unsqueeze(s[:, 1] - 0.3 , dim=1)
        rou_4 = torch.unsqueeze( -s[:, 1] + 7.7, dim=1)
        rou_5 = torch.norm(s[:, 0:2] - torch.tensor([5,5]).to(s.device).reshape(1, 2), dim=1, keepdim=True) - 1.8
        
        rou_6 = torch.norm(s[:, 0:2] - s[:, 4:6], dim=1, keepdim=True) - 0.6 

        return torch.hstack( (rou_1, rou_2, rou_3, rou_4, rou_5, rou_6) ) 

    def rou_n(s: torch.Tensor) -> torch.Tensor:
        s_norm = torch.norm(s, dim=1)

        return - s_norm + 0.6

    point_robots_dis.set_domain_limits(domain_lower_bd, domain_upper_bd)
    point_robots_dis.set_control_limits(control_lower_bd, control_upper_bd)
    point_robots_dis.set_disturbance_limits(disturbance_lower_bd, disturbance_upper_bd)
    point_robots_dis.set_state_constraints(rou)
    point_robots_dis.set_nominal_state_constraints(rou_n)

    
    x = torch.tensor([5, 5, 0, 1], dtype=torch.float).reshape(1, 4)
    # x = torch.rand(3,3, dtype=torch.float)
    u_ref = torch.rand(1, 2, dtype=torch.float)
    disturb = torch.rand(1, 2, dtype=torch.float) 


    f = point_robots_dis.f(x)
    print(f"the shape of f is {f.shape} \n f is {f} \n ")
    g = point_robots_dis.g(x)
    print(f"the shape of g is {g.shape} \n g is {g} \n ")

    d = point_robots_dis.d(x)
    print(f"the shape of d is {d.shape} \n d is {d} \n ")

    dsdt =point_robots_dis.dsdt(x, u_ref)
    print(f"dsdt is {dsdt}")

    dsdt =point_robots_dis.dsdt(x, u_ref, disturb)
    print(f"dsdt with disturbance is {dsdt} \n ")

    x_next = point_robots_dis.step(x, u_ref, disturb)
    print(f"x is {x} \n")
    print(f"x_nest is {x_next}")

    rou = point_robots_dis.state_constraints(x)