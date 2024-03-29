from typing import Tuple, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

import numpy as np
import control

from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem

class TwoVehicleAvoidance(ControlAffineSystem):
    """
    Represents a damped inverted pendulum.

    The system has state

        s = [x_e, y_e, theta_e, x_c, y_c, theta_c]

    representing the angle and velocity of the pendulum, and it
    has control inputs

        u = [w_e]

    representing the torque applied. and the disturbance is
        d = [w_c]

    """
    
    # Number of states and controls
    N_DIMS = 6
    N_CONTROLS = 1
    N_DISTURBANCES = 1

    # State indices
    XE = 0
    YE = 1
    THETAE = 2
    XC = 3
    YC = 4
    THETAC = 5
  
    
    # Control indices
    WE = 0

    # Disturbance indices
    WC = 0

    def __init__(self, ns=N_DIMS, nu=N_CONTROLS, nd=N_DISTURBANCES,  v_e=0.4, v_c=0.4, dt=0.05):
        super().__init__(ns, nu, nd, dt)
        self.v_e = v_e
        self.v_c = v_c
        self.period_state_index = [TwoVehicleAvoidance.THETAE, TwoVehicleAvoidance.THETAC]


    def f(self, s: torch.Tensor) -> torch.Tensor:
        batch_size = s.shape[0]
        f = torch.zeros((batch_size, self.ns, 1))
        f = f.type_as(s)
        
        f[:, TwoVehicleAvoidance.XE, 0] = self.v_e * torch.cos(s[:, TwoVehicleAvoidance.THETAE])
        f[:, TwoVehicleAvoidance.YE, 0] = self.v_e * torch.sin(s[:, TwoVehicleAvoidance.THETAE])
        f[:, TwoVehicleAvoidance.XC, 0] = self.v_c * torch.cos(s[:, TwoVehicleAvoidance.THETAC])
        f[:, TwoVehicleAvoidance.YC, 0] = self.v_c * torch.sin(s[:, TwoVehicleAvoidance.THETAC])

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
        g[:, TwoVehicleAvoidance.THETAE, TwoVehicleAvoidance.WE] = 1

        return g

    def d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the disturbance dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            d: bs x self.n_dims x self.n_disturbances tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        d = torch.zeros((batch_size, self.ns, self.nd))
        d = d.type_as(x)

        d[:, TwoVehicleAvoidance.THETAC, TwoVehicleAvoidance.WC] = 1

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

    two_vehicle_avoidance = TwoVehicleAvoidance()

    domain_lower_bd = torch.Tensor([-1, -1, -4, -1, -1, -4]).float()
    domain_upper_bd = torch.Tensor([9, 9, 4, 9, 9, 4]).float()

    control_lower_bd = torch.Tensor([-1]).float()
    control_upper_bd = -control_lower_bd

    disturbance_lower_bd = torch.Tensor([-1]).float()
    disturbance_upper_bd = -disturbance_lower_bd
        
    def rou(s: torch.Tensor) -> torch.Tensor:
        rou_1 = torch.unsqueeze(s[:, 0] + 0, dim=1)
        rou_2 = torch.unsqueeze( - s[:, 0] + 8, dim=1)
        rou_3 = torch.unsqueeze(s[:, 1] + 0, dim=1)
        rou_4 = torch.unsqueeze( -s[:, 1] + 8, dim=1)
        rou_5 = torch.norm(s[:, 0:2] - torch.tensor([5,5]).to(s.device).reshape(1, 2), dim=1, keepdim=True) - 1.5
        
        rou_6 = torch.norm(s[:, 0:2] - s[:, 3:5], dim=1, keepdim=True) - 0.6

        return torch.hstack( (rou_1, rou_2, rou_3, rou_4, rou_5, rou_6) ) 

    def rou_n(s: torch.Tensor) -> torch.Tensor:
        s_norm = torch.norm(s, dim=1, keepdim=True)

        return - s_norm + 0.6

    two_vehicle_avoidance.set_domain_limits(domain_lower_bd, domain_upper_bd)
    two_vehicle_avoidance.set_control_limits(control_lower_bd, control_upper_bd)
    two_vehicle_avoidance.set_disturbance_limits(disturbance_lower_bd, disturbance_upper_bd)
    two_vehicle_avoidance.set_state_constraints(rou)
    two_vehicle_avoidance.set_nominal_state_constraints(rou_n)

    
    x = torch.tensor([5, 5, 0, 1, 1, 0], dtype=torch.float).reshape(1, two_vehicle_avoidance.ns)
    # x = torch.rand(3,3, dtype=torch.float)
    u_ref = torch.rand(1, 1, dtype=torch.float)
    
    f = two_vehicle_avoidance.f(x)
    print(f"the shape of f is {f.shape} \n f is {f} \n ")
    g = two_vehicle_avoidance.g(x)
    print(f"the shape of g is {g.shape} \n g is {g} \n ")

    dsdt =two_vehicle_avoidance.dsdt(x, u_ref)
    print(f"the shape of dsdt is {dsdt.shape} \n dsdt is {dsdt} \n ")

    x_next = two_vehicle_avoidance.step(x, u_ref)
    print(f"x is {x} \n")
    print(f"x_nest is {x_next}")

    rou = two_vehicle_avoidance.state_constraints(x)