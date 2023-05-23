from typing import Tuple, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

import numpy as np
import control

from control_affine_system import ControlAffineSystem

class InvertedPendulum(ControlAffineSystem):
    """
    Represents a damped inverted pendulum.

    The system has state

        s = [theta, theta_dot]

    representing the angle and velocity of the pendulum, and it
    has control inputs

        u = [u]

    representing the torque applied.

    The system is parameterized by
        m: mass
        L: length of the pole
        b: damping
    """
    
    # Number of states and controls
    N_DIMS = 2
    N_CONTROLS = 1

    # State indices
    THETA = 0
    THETA_DOT = 1
    # Control indices
    U = 0

    def __init__(self, ns=N_DIMS, nu=N_CONTROLS, dt=0.01, m=1.0, L=1.0, b=0.1):
        super().__init__(ns, nu, dt)
        self.m = m
        self.L = L
        self.b = b
        self.gravity = 9.81

        A = np.array([[0, 1], [0, -self.b/(self.m * self.L**2)]]).reshape((2,2))
        B = np.array([0, 1/(self.m * self.L **2)]).reshape((self.ns, self.nu))
        Q = np.eye(self.ns)
        R = np.eye(self.nu)
        K, _, _ = control.lqr(A, B, Q, R)
        self.K_lqr = torch.from_numpy(K).float()

    def f(self, s: torch.Tensor) -> torch.Tensor:
        batch_size = s.shape[0]
        f = torch.zeros((batch_size, self.ns, 1))
        f = f.type_as(s)

        # and state variables
        theta = s[:, InvertedPendulum.THETA]
        theta_dot = s[:, InvertedPendulum.THETA_DOT]

        # The derivatives of theta is just its velocity
        f[:, InvertedPendulum.THETA, 0] = theta_dot

        # Acceleration in theta depends on theta via gravity and theta_dot via damping
        f[:, InvertedPendulum.THETA_DOT, 0] = (
            9.81 / self.L * torch.sin(theta) - self.b / (self.m * self.L ** 2) * theta_dot
        )

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
        g[:, InvertedPendulum.THETA_DOT, InvertedPendulum.U] = 1 / (self.m * self.L ** 2)

        return g

if __name__ == "__main__":

    inverted_pendulum = InvertedPendulum()

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

    inverted_pendulum.set_domain_limits(domain_lower_bd, domain_upper_bd)
    inverted_pendulum.set_control_limits(control_lower_bd, control_upper_bd)
    inverted_pendulum.set_state_constraints(rou)
    inverted_pendulum.set_nominal_state_constraints(rou_n)

    

    x = torch.rand(3,2, dtype=torch.float)
    u_ref = torch.rand(3, 1, dtype=torch.float)
    
    f = inverted_pendulum.f(x)
    print(f"the shape of f is {f.shape} \n f is {f} \n ")
    g = inverted_pendulum.g(x)
    print(f"the shape of g is {g.shape} \n g is {g} \n ")

    dsdt =inverted_pendulum.dsdt(x, u_ref)
    print(f"the shape of dsdt is {dsdt.shape} \n dsdt is {dsdt} \n ")

    x_next = inverted_pendulum.step(x, u_ref)
    print(f"x is {x} \n")
    print(f"x_nest is {x_next}")

    print(f"the K is {inverted_pendulum.K}, the shape is {inverted_pendulum.K.shape}")
    print(f"u= - K*s is { -inverted_pendulum.K @ x[0,:]} ")