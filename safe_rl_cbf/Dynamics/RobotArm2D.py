from typing import Tuple, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

import numpy as np
import control

from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem

class RobotArm2D(ControlAffineSystem):
    """
    Represents a 2d robot arm.

    The system has state

        s = [theta_1, theta_2]

    representing the angle and velocity of the pendulum, and it
    has control inputs

        u = [w_1, w_2]

    representing the torque applied.

    """
    
    # Number of states and controls
    N_DIMS = 2
    N_CONTROLS = 2
    N_DISTURBANCES = 0

    # State indices
    THETA_1 = 0
    THETA_2 = 1
    
    # Control indices
    W_1 = 0
    W_2 = 1

    def __init__(self, ns=N_DIMS, nu=N_CONTROLS, nd=N_DISTURBANCES , dt=0.01):
        super().__init__(ns, nu, nd, dt)
        self.l1 = 2
        self.l2 = 2

        self.period_state_index = [RobotArm2D.THETA_1, RobotArm2D.THETA_2]


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
        # g[:, RobotArm2D.X, RobotArm2D.W_1] = - self.l1 * torch.sin(x[:, RobotArm2D.THETA_1]) - self.l2 * torch.sin(x[:, RobotArm2D.THETA_1] + x[:, RobotArm2D.THETA_2])
        # g[:, RobotArm2D.Y, RobotArm2D.W_1] = self.l1 * torch.cos(x[:, RobotArm2D.THETA_1]) + self.l2 * torch.cos(x[:, RobotArm2D.THETA_1] + x[:, RobotArm2D.THETA_2])
        # g[:, RobotArm2D.X, RobotArm2D.W_2] = - self.l2 * torch.sin(x[:, RobotArm2D.THETA_1] + x[:, RobotArm2D.THETA_2])
        # g[:, RobotArm2D.Y, RobotArm2D.W_2] = self.l2 * torch.cos(x[:, RobotArm2D.THETA_1] + x[:, RobotArm2D.THETA_2])
        g[:, RobotArm2D.THETA_1, RobotArm2D.W_1] = 1
        g[:, RobotArm2D.THETA_2, RobotArm2D.W_2] = 1

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

    robot_arm_2d = RobotArm2D()

    domain_lower_bd = torch.Tensor([-4, -4]).float()
    domain_upper_bd = torch.Tensor([4, 4]).float()

    control_lower_bd = torch.Tensor([-0.5, -0.5]).float()
    control_upper_bd = -control_lower_bd
        
    def rou(s: torch.Tensor) -> torch.Tensor:
        theta_1 = s[:, 0]
        theta_2 = s[:, 1]
        l1 = 2
        l2 = 2
        x_e = l2 * torch.cos(theta_1 + theta_2) + l1 * torch.cos(theta_1)
        y_e = l2 * torch.sin(theta_1 + theta_2) + l1 * torch.sin(theta_1) 

        rou_1 = torch.unsqueeze(y_e + 0.5, dim=1)
        rou_2 = torch.unsqueeze( - y_e + 3, dim=1)
        rou_3 = torch.unsqueeze( theta_1 + 3, dim=1)
        rou_4 = torch.unsqueeze( - theta_1 + 3, dim=1)
        rou_5 = torch.unsqueeze( theta_2 + 3, dim=1)
        rou_6 = torch.unsqueeze( - theta_2 + 3, dim=1)
        
        return torch.hstack( (rou_1, rou_2, rou_3, rou_4, rou_5, rou_6) )

    def rou_n(s: torch.Tensor) -> torch.Tensor:
        s_norm = torch.norm(s, dim=1, keepdim=True)

        return - s_norm + 0.6

    robot_arm_2d.set_domain_limits(domain_lower_bd, domain_upper_bd)
    robot_arm_2d.set_control_limits(control_lower_bd, control_upper_bd)
    robot_arm_2d.set_state_constraints(rou)
    robot_arm_2d.set_nominal_state_constraints(rou_n)

    
    x = torch.tensor([0, 1, 2, -3], dtype=torch.float).reshape(2, 2)
    # x = torch.rand(3,3, dtype=torch.float)
    u_ref = torch.rand(2, 2, dtype=torch.float)
    
    f = robot_arm_2d.f(x)
    print(f"the shape of f is {f.shape} \n f is {f} \n ")
    g = robot_arm_2d.g(x)
    print(f"the shape of g is {g.shape} \n g is {g} \n ")

    dsdt =robot_arm_2d.dsdt(x, u_ref)
    print(f"the shape of dsdt is {dsdt.shape} \n dsdt is {dsdt} \n ")

    x_next = robot_arm_2d.step(x, u_ref)
    print(f"x is {x} \n")
    print(f"x_nest is {x_next}")

    rou = robot_arm_2d.state_constraints(x)