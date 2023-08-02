from typing import Tuple, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

import numpy as np
import control

from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem

class CartPole(ControlAffineSystem):
    """
    Represents a friction free CartPole.

    The system has state

        s = [x, dot_x, theta, theta_dot]

    representing the position of cart and angle of pendulum, and it
    has control inputs

        u = [u]

    representing the force applied.

    The system is parameterized by
        m_c: mass of the cart
        m_p: mass of the pole
        L: length of the pole
        g: gravity
    """
    
    # Number of states and controls
    N_DIMS = 4
    N_CONTROLS = 1
    N_DISTURBANCES = 0

    # State indices
    X = 0
    X_DOT = 1
    THETA = 2
    THETA_DOT = 3
    # Control indices
    U = 0

    def __init__(self, ns=N_DIMS, nu=N_CONTROLS, nd=N_DISTURBANCES, dt=0.01, m_c=1.0, m_p=0.1, L=0.5):
        super().__init__(ns, nu, nd, dt)
        self.masscart = m_c
        self.masspole = m_p
        self.total_mass = self.masspole + self.masscart
        self.length = L
        self.gravity = 9.81
        self.polemass_length = self.masspole * self.length
        self.force_mag = 15.0
        self.tau = dt  
        

    def f(self, s: torch.Tensor) -> torch.Tensor:
        batch_size = s.shape[0]
        f = torch.zeros((batch_size, self.ns, 1))
        f = f.type_as(s)

        # and state variables
        x = s[:, CartPole.X]
        x_dot = s[:, CartPole.X_DOT]
        theta = s[:, CartPole.THETA]
        theta_dot = s[:, CartPole.THETA_DOT]

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (self.polemass_length * theta_dot**2 * sintheta ) / self.total_mass
        l_4_3 = self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)

        # xacc2 = temp2 - self.polemass_length * costheta / self.total_mass * (self.gravity * sintheta - costheta * temp2) / l_4_3 + (1/self.total_mass + self.polemass_length * costheta ** 2 /(self.total_mass**2 * l_4_3)  ) * force
        # thetaacc2 = (self.gravity * sintheta - costheta * temp2) / l_4_3 - costheta / (self.total_mass * l_4_3) * force 
        

        # The derivatives of theta is just its velocity
        f[:, CartPole.X, 0] = x_dot
        f[:, CartPole.X_DOT, 0] = temp - self.polemass_length * costheta / self.total_mass * (self.gravity * sintheta - costheta * temp) / l_4_3
        f[:, CartPole.THETA, 0] = theta_dot
        # Acceleration in theta depends on theta via gravity and theta_dot via damping
        f[:, CartPole.THETA_DOT, 0] = (self.gravity * sintheta - costheta * temp) / l_4_3

        return f.squeeze(dim=-1)
    
    
    def g(self, s: torch.Tensor) -> torch.Tensor:
        
        # Extract batch size and set up a tensor for holding the result
        batch_size = s.shape[0]

        g = torch.zeros((batch_size, self.ns, self.nu))
        g = g.type_as(s)

        x = s[:, CartPole.X]
        x_dot = s[:, CartPole.X_DOT]
        theta = s[:, CartPole.THETA]
        theta_dot = s[:, CartPole.THETA_DOT]

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (self.polemass_length * theta_dot**2 * sintheta ) / self.total_mass
        l_4_3 = self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)



        # Effect on theta dot
        g[:, CartPole.X, CartPole.U] = 0.0
        g[:, CartPole.X_DOT, CartPole.U] = (1/self.total_mass + self.polemass_length * costheta ** 2 /(self.total_mass**2 * l_4_3)  )
        g[:, CartPole.THETA, CartPole.U] = 0.0
        g[:, CartPole.THETA_DOT, CartPole.U] = costheta / (self.total_mass * l_4_3)


        return g

        
    def d(self, s: torch.Tensor) -> torch.Tensor:
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

    cart_pole = CartPole()

    domain_lower_bd = torch.Tensor([-2.5, -5, -torch.pi * 3 / 2 , -5]).float()
    domain_upper_bd = -domain_lower_bd

    control_lower_bd =torch.Tensor([-15]).float()
    control_upper_bd = -control_lower_bd
        
    def rou(s: torch.Tensor) -> torch.Tensor:
        rou_1 = torch.unsqueeze(s[:, 0] + 2, dim=1)
        rou_2 = torch.unsqueeze( - s[:, 0] + 2, dim=1)
        
        return torch.hstack( (rou_1, rou_2) ) 

    def rou_n(s: torch.Tensor) -> torch.Tensor:
        s_norm = torch.norm(s, dim=1, keepdim=True)

        return - s_norm + 0.6

    cart_pole.set_domain_limits(domain_lower_bd, domain_upper_bd)
    cart_pole.set_control_limits(control_lower_bd, control_upper_bd)
    cart_pole.set_state_constraints(rou)
    cart_pole.set_nominal_state_constraints(rou_n)

    

    x = torch.rand(3,4, dtype=torch.float)
    u_ref = torch.rand(3, 1, dtype=torch.float)
    
    f = cart_pole.f(x)
    print(f"the shape of f is {f.shape} \n f is {f} \n ")
    g = cart_pole.g(x)
    print(f"the shape of g is {g.shape} \n g is {g} \n ")

    dsdt =cart_pole.dsdt(x, u_ref)
    print(f"the shape of dsdt is {dsdt.shape} \n dsdt is {dsdt} \n ")

    x_next = cart_pole.step(x, u_ref)
    print(f"x is {x} \n")
    print(f"x_nest is {x_next}")
