from typing import Tuple, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gurobipy as gp
from gurobipy import GRB

class Car:
    def __init__(self, ns=2, dt=0.01):
        self.ns = ns
        self.dt = dt

    def f(self, s):
        A = torch.tensor([[0, 1],[0, 0]], dtype=torch.float).to(s.device)
        result = A @ s.T 
        return result.T

    def g(self, s, m=0.5):
        result =  torch.tensor([0, m], dtype=torch.float).reshape((1,2)).to(s.device) * torch.ones((s.shape[0], 2), dtype=torch.float).to(s.device)
        return result

    def set_domain_limits(self, lower_bd: torch.Tensor,upper_bd: torch.Tensor):
        self.domain_lower_bd = lower_bd
        self.domain_upper_bd = upper_bd
    
    def set_control_limits(self, lower_bd: torch.Tensor,upper_bd: torch.Tensor):
        self.control_lower_bd = lower_bd
        self.control_upper_bd = upper_bd

    def set_state_constraints(self, rou):
        """
        rou(s) is a function such that rous(s) >= 0 implies system is safe.
        """
        self.state_constraints = rou

    def set_barrier_function(self, h):
        self.h = h


    def safe_mask(self, s: torch.Tensor) -> Tuple[bool]:
        rou_s = self.state_constraints(s)
        x = rou_s >=0
        safe_mask = torch.all(x, dim=1)
        return safe_mask

    def unsafe_mask(self, s:torch.Tensor) -> Tuple[bool]:
        rou_s = self.state_constraints(s)
        x= rou_s >= 0
        unsafe_mask = torch.all(x, dim=1)
        return ~unsafe_mask

    def step(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ds = self.f(s) + self.g(s) * u
        s_next = s + ds * self.dt
        return s_next

    def V_lie_derivatives(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the Lie derivatives of the CLF V along the control-affine dynamics

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            
        returns:
            Lf_V: bs x len(scenarios) x 1 tensor of Lie derivatives of V
                  along f
            Lg_V: bs x len(scenarios) x self.dynamics_model.n_controls tensor
                  of Lie derivatives of V along g
        """

        # Get the Jacobian of V for each entry in the batch
        _, gradh = self.h.V_with_jacobian(x)
        # print(f"gradh shape is {gradh.shape}")
        # We need to compute Lie derivatives for each scenario
        batch_size = x.shape[0]
        Lf_V = torch.zeros(batch_size, 2, 1)
        Lg_V = torch.zeros(batch_size, 2, 1)

        f = torch.unsqueeze(self.f(x), dim=-1)
        g = torch.unsqueeze(self.g(x), dim=-1)
        # print(f"f shape is {f.shape}")
        # print(f"g shape is {g.shape}")

        Lf_V = torch.bmm(gradh, f).squeeze(1)
        Lg_V = torch.bmm(gradh, g).squeeze(1)
        # print(f"Lf_V shape is {Lf_V.shape}")
        # print(f"Lg_V shape is {Lg_V.shape}")
        
        return Lf_V, Lg_V

    def solve_CLF_QP(
        self,
        x,
        u_ref: Optional[torch.Tensor] = None,
        relaxation_penalty: Optional[float] = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            relaxation_penalty: the penalty to use for CLF relaxation, defaults to
                                self.clf_relaxation_penalty
            u_ref: allows the user to supply a custom reference input, which will
                   bypass the self.u_reference function. If provided, must have
                   dimensions bs x self.dynamics_model.n_controls. If not provided,
                   default to calling self.u_reference.
            requires_grad: if True, use a differentiable layer
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """
        # Get the value of the CLF and its Lie derivatives
        H, _ = self.h(x)
        Lf_V, Lg_V = self.V_lie_derivatives(x)

        # Get the reference control input as well
        if u_ref is not None:
            err_message = f"u_ref must have {x.shape[0]} rows, but got {u_ref.shape[0]}"
            assert u_ref.shape[0] == x.shape[0], err_message
            err_message = f"u_ref must have {1} cols,"
            err_message += f" but got {u_ref.shape[1]}"
            assert u_ref.shape[1] == 1, err_message
        else:
            err_message = f"u_ref shouldn't be None!!!!"
            assert u_ref is not None, err_message
    
        return self._solve_CLF_QP_gurobi(
            x, u_ref, H, Lf_V, Lg_V, relaxation_penalty
        )
    

    def _solve_CLF_QP_gurobi(
        self,
        x: torch.Tensor,
        u_ref: torch.Tensor,
        V: torch.Tensor,
        Lf_V: torch.Tensor,
        Lg_V: torch.Tensor,
        relaxation_penalty: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP. Solves the QP using
        Gurobi, which does not allow for backpropagation.

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            u_ref: bs x self.dynamics_model.n_controls tensor of reference controls
            V: bs x 1 tensor of CLF values,
            Lf_V: bs x 1 tensor of CLF Lie derivatives,
            Lg_V: bs x self.dynamics_model.n_controls tensor of CLF Lie derivatives,
            relaxation_penalty: the penalty to use for CLF relaxation.
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """
        # To find the control input, we want to solve a QP constrained by
        #
        # -(L_f V + L_g V u + lambda V) <= 0
        #
        # To ensure that this QP is always feasible, we relax the constraint
        #
        #  -(L_f V + L_g V u + lambda V) - r <= 0
        #                              r >= 0
        #
        # and add the cost term relaxation_penalty * r.
        #
        # We want the objective to be to minimize
        #
        #           ||u||^2 + relaxation_penalty * r^2
        #
        # This reduces to (ignoring constant terms)
        #
        #           u^T I u + relaxation_penalty * r^2

        n_controls = 1
        
        allow_relaxation = not (relaxation_penalty == float("inf"))

        # Solve a QP for each row in x
        bs = x.shape[0]
        u_result = torch.zeros(bs, n_controls)
        r_result = torch.zeros(bs, 1)
        for batch_idx in range(bs):
            # Skip any bad points
            if (
                torch.isnan(x[batch_idx]).any()
                or torch.isinf(x[batch_idx]).any()
                or torch.isnan(Lg_V[batch_idx]).any()
                or torch.isinf(Lg_V[batch_idx]).any()
                or torch.isnan(Lf_V[batch_idx]).any()
                or torch.isinf(Lf_V[batch_idx]).any()
            ):
                continue

            # Instantiate the model
            model = gp.Model("clf_qp")
            # Create variables for control input and (optionally) the relaxations
            lower_lim, upper_lim = self.control_limits
            upper_lim = upper_lim.cpu().numpy()
            lower_lim = lower_lim.cpu().numpy()
            u = model.addMVar(n_controls, lb=lower_lim, ub=upper_lim)
            if allow_relaxation:
                r = model.addMVar(1, lb=0, ub=GRB.INFINITY)

            # Define the cost
            Q = np.eye(n_controls)
            u_ref_np = u_ref[batch_idx, :].detach().cpu().numpy()
            objective = u @ Q @ u - 2 * u_ref_np @ Q @ u + u_ref_np @ Q @ u_ref_np
            if allow_relaxation:
                relax_penalties = relaxation_penalty * np.ones(1)
                objective += relax_penalties @ r

            # Now build the CLF constraints
        
            Lg_V_np = Lg_V[batch_idx, 0].detach().cpu().numpy()
            Lf_V_np = Lf_V[batch_idx, 0].detach().cpu().numpy()
            V_np = V[batch_idx, 0].detach().cpu().numpy()
            clf_constraint = -(Lf_V_np + Lg_V_np * u + 0.5 * V_np)
            if allow_relaxation:
                clf_constraint -= r[0]
            model.addConstr(clf_constraint <= 0.0, name=f"Scenario {0} Decrease")

            # Optimize!
            model.setParam("DualReductions", 0)
            model.setObjective(objective, GRB.MINIMIZE)
            model.optimize()

            if model.status != GRB.OPTIMAL:
                # Make the relaxations nan if the problem was infeasible, as a signal
                # that something has gone wrong
                if allow_relaxation:
                    for i in range(1):
                        r_result[batch_idx, i] = torch.tensor(float("nan"))
                continue

            # Extract the results
            for i in range(n_controls):
                u_result[batch_idx, i] = torch.tensor(u[i].x)
            if allow_relaxation:
                for i in range(0):
                    r_result[batch_idx, i] = torch.tensor(r[i].x)

        return u_result.type_as(x), r_result.type_as(x)


    @property
    def domain_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of states for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = self.domain_upper_bd
        lower_limit = self.domain_lower_bd

        return (lower_limit, upper_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = self.control_upper_bd
        lower_limit = self.control_lower_bd

        return (lower_limit, upper_limit)

    
if __name__ == "__main__":
    car = Car()
    h = torch.load("h.pt")
    car.set_barrier_function(h)

    x = torch.rand(3,2, dtype=torch.float)
    u_ref = torch.rand(3, 1, dtype=torch.float)
    u_result, r_result = car.solve_CLF_QP(x, u_ref, relaxation_penalty=1000)
    print(f"u_result is {u_result}")
    print(f"r_result is {r_result}")

    s0 = torch.tensor([0.4, 0.1], dtype=torch.float).reshape((1,2))
    u_ref = torch.tensor([-0.5], dtype=torch.float).reshape((1,1))
    u0, r0 = car.solve_CLF_QP(s0,u_ref)
    assert r0 == 0.0, f"not feasible"
    s1 = car.step(s=s0, u=u0)
    print(f"s0 = {s0} \n the input is {u0},  next state is {s1}")


