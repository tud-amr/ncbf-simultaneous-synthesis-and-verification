import numpy as np
import itertools
from typing import Tuple, List, Optional
import json

import time
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gurobipy as gp
from gurobipy import GRB

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt

from DataModule import DataModule
from control_affine_system import ControlAffineSystem

from dynamic_system_instances import car1, inverted_pendulum_1

######################### define neural network #########################


class NeuralNetwork(pl.LightningModule):
    def __init__(
        self,
        dynamic_system: ControlAffineSystem,
        data_module: DataModule,
        learn_shape_epochs: int = 10,
        primal_learning_rate: float = 1e-3,
        gamma: float = 0.9,
        clf_lambda: float = 1.0,
        clf_relaxation_penalty: float = 50.0,
        ):

        super(NeuralNetwork, self).__init__()
        self.dynamic_system = dynamic_system
        self.data_module = data_module
        self.learn_shape_epochs = learn_shape_epochs
        self.primal_learning_rate = primal_learning_rate
        self.dt = self.dynamic_system.dt
        self.gamma = gamma
        self.flatten = nn.Flatten()
        self.h = nn.Sequential(
            nn.Linear(2, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        
        u_lower, u_upper = self.dynamic_system.control_limits
        self.u_v =[u_lower.item(), u_upper.item()]
        self.train_loss = []
        self.val_loss = []
        
        self.K = self.dynamic_system.K
        self.clf_lambda = clf_lambda
        self.clf_relaxation_penalty = clf_relaxation_penalty

        self.generate_cvx_solver()
    
    def prepare_data(self):
        return self.data_module.prepare_data()

    def setup(self, stage: Optional[str] = None):
        return self.data_module.setup(stage)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def forward(self, s):
        hs = self.h(s)

        return hs

    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CLBF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLBF
        returns:
            V: bs tensor of CLBF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """

        # Compute the CLBF layer-by-layer, computing the Jacobian alongside

        # We need to initialize the Jacobian to reflect the normalization that's already
        # been done to x

        x_norm = x # self.normalize(x)

        bs = x_norm.shape[0]
        JV = torch.zeros(
            (bs, self.dynamic_system.ns, self.dynamic_system.ns)
        ).type_as(x)
        # and for each non-angle dimension, we need to scale by the normalization
        for dim in range(self.dynamic_system.ns):
            JV[:, dim, dim] = 1.0

        # Now step through each layer in V
        V = x_norm
        for layer in self.h:
            V = layer(V)

            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
            elif isinstance(layer, nn.Tanh):
                JV = torch.matmul(torch.diag_embed(1 - V ** 2), JV)
            elif isinstance(layer, nn.ReLU):
                JV = torch.matmul(torch.diag_embed(torch.sign(V)), JV)

        return V, JV


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
        _, gradh = self.V_with_jacobian(x)
        # print(f"gradh shape is {gradh.shape}")
        # We need to compute Lie derivatives for each scenario
        batch_size = x.shape[0]
        Lf_V = torch.zeros(batch_size, 2, 1)
        Lg_V = torch.zeros(batch_size, 2, 1)

        f = torch.unsqueeze(self.dynamic_system.f(x), dim=-1)
        g = torch.unsqueeze(self.dynamic_system.g(x), dim=-1)
        # print(f"f shape is {f.shape}")
        # print(f"g shape is {g.shape}")

        Lf_V = torch.bmm(gradh, f).squeeze(1)
        Lg_V = torch.bmm(gradh, g).squeeze(1)
        # print(f"Lf_V shape is {Lf_V.shape}")
        # print(f"Lg_V shape is {Lg_V.shape}")
        
        return Lf_V, Lg_V

    def boundary_loss(
        self,
        s : torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to boundary conditions

        args:
            s: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        eps = 1e-2
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        hs = self.h(s)

        # 1.) h > 0 in the safe region
        hs_safe = hs[safe_mask]
        safe_violation = 1e2 *  F.relu(eps - hs_safe)
        safe_hs_term =  safe_violation.mean()
        loss.append(("CLBF safe region term", safe_hs_term))
        if accuracy:
            safe_V_acc = (safe_violation >= eps).sum() / safe_violation.nelement()
            loss.append(("CLBF safe region accuracy", safe_V_acc))

        #   3.) h < 0 in the unsafe region
        hs_unsafe = hs[unsafe_mask]
        unsafe_violation = 1e2 *  F.relu(eps + hs_unsafe)
        unsafe_hs_term = unsafe_violation.mean()
        loss.append(("CLBF unsafe region term", unsafe_hs_term))
        if accuracy:
                unsafe_V_acc = (
                    unsafe_violation >= eps 
                ).sum() / unsafe_violation.nelement()
                loss.append(("CLBF unsafe region accuracy", unsafe_V_acc))

        return loss
    
    def descent_loss(
        self,
        s: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
        requires_grad: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to the descent condition

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        loss = []
        
        # The CLBF decrease condition requires that V is decreasing everywhere where
        # V <= safe_level. We'll encourage this in three ways:
        #
        #   1) Minimize the relaxation needed to make the QP feasible.
        #   2) Compute the CLBF decrease at each point by linearizing
        #   3) Compute the CLBF decrease at each point by simulating

        # First figure out where this condition needs to hold
        eps = 0.1
        H = self.h(s)
        condition_active = torch.sigmoid(10 * (1.0 + eps - H))
 

    
        u_qp, qp_relaxation = self.solve_CLF_QP(s, requires_grad=False)
        qp_relaxation = torch.mean(qp_relaxation, dim=-1)

        # Minimize the qp relaxation to encourage satisfying the decrease condition
        qp_relaxation_loss = qp_relaxation.mean()
        # loss.append(("QP relaxation", qp_relaxation_loss))

        # Now compute the decrease using linearization
        eps = 1.0
        clbf_descent_term_lin = torch.tensor(0.0).type_as(s)
        clbf_descent_acc_lin = torch.tensor(0.0).type_as(s)
        # Get the current value of the CLBF and its Lie derivatives
        Lf_V, Lg_V = self.V_lie_derivatives(s)
    
        # Use the dynamics to compute the derivative of V
        Vdot = Lf_V[:, :].unsqueeze(1) + torch.bmm(
            Lg_V[:, :].unsqueeze(1),
            u_qp.reshape(-1, self.dynamic_system.nu, 1),
        )
        Vdot = Vdot.reshape(H.shape)
        violation = F.relu(eps - (Vdot + self.clf_lambda * H))
        violation = violation * condition_active
        clbf_descent_term_lin = clbf_descent_term_lin + violation.mean()
        clbf_descent_acc_lin = clbf_descent_acc_lin + (violation >= eps).sum() / (
            violation.nelement()
        )

        loss.append(("CLBF descent term (linearized)", clbf_descent_term_lin))
        if accuracy:
            loss.append(("CLBF descent accuracy (linearized)", clbf_descent_acc_lin))


        # Now compute the decrease using simulation
        eps = 1.0
        clbf_descent_term_sim = torch.tensor(0.0).type_as(s)
        clbf_descent_acc_sim = torch.tensor(0.0).type_as(s)
       
        # xdot = self.dynamics_model.closed_loop_dynamics(x, u_qp, params=s)

        x_next = self.dynamic_system.step(s, u_qp)
        H_next = self.h(x_next)
        violation = F.relu(
            eps - ((H_next - H) / self.dynamic_system.dt + self.clf_lambda * H)
        )
        violation = violation * condition_active

        clbf_descent_term_sim = clbf_descent_term_sim + violation.mean()
        clbf_descent_acc_sim = clbf_descent_acc_sim + (violation >= eps).sum() / (
            violation.nelement() 
        )

        loss.append(("CLBF descent term (simulated)", clbf_descent_term_sim))
        if accuracy:
            loss.append(("CLBF descent accuracy (simulated)", clbf_descent_acc_sim))


        return loss


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
        _, gradh = self.V_with_jacobian(x)
        # print(f"gradh shape is {gradh.shape}")
        # We need to compute Lie derivatives for each scenario
        batch_size = x.shape[0]
        Lf_V = torch.zeros(batch_size, 2, 1)
        Lg_V = torch.zeros(batch_size, 2, 1)

        f = self.dynamic_system.f(x)
        g = self.dynamic_system.g(x)
        # print(f"f shape is {f.shape}")
        # print(f"g shape is {g.shape}")

        Lf_V = torch.bmm(gradh, f.unsqueeze(dim=-1)).squeeze(1)
        Lg_V = torch.bmm(gradh, g).squeeze(1)
        # print(f"Lf_V shape is {Lf_V.shape}")
        # print(f"Lg_V shape is {Lg_V.shape}")
        
        return Lf_V, Lg_V

    def generate_cvx_solver(self):
        # Save the other parameters
        

        # Since we want to be able to solve the CLF-QP differentiably, we need to set
        # up the CVXPyLayers optimization. First, we define variables for each control
        # input and the relaxation in each scenario
        u = cp.Variable(self.dynamic_system.nu)
        clf_relaxations = []
        
        clf_relaxations.append(cp.Variable(1, nonneg=True))

        # Next, we define the parameters that will be supplied at solve-time: the value
        # of the Lyapunov function, its Lie derivatives, the relaxation penalty, and
        # the reference control input   
        V_param = cp.Parameter(1, nonneg=True)
        Lf_V_params = []
        Lg_V_params = []
        
        Lf_V_params.append(cp.Parameter(1))
        Lg_V_params.append(cp.Parameter(self.dynamic_system.nu))

        clf_relaxation_penalty_param = cp.Parameter(1, nonneg=True)
        u_ref_param = cp.Parameter(self.dynamic_system.nu)


        # These allow us to define the constraints
        constraints = []
        
        # CLF decrease constraint (with relaxation)
        constraints.append(
            Lf_V_params[0]
            + Lg_V_params[0] @ u
            + self.clf_lambda * V_param
            - clf_relaxations[0]
            <= 0
        )


        # Control limit constraints
        lower_lim,  upper_lim  = self.dynamic_system.control_limits
        for control_idx in range(self.dynamic_system.nu):
            constraints.append(u[control_idx] >= lower_lim[control_idx])
            constraints.append(u[control_idx] <= upper_lim[control_idx])


        # And define the objective
        objective_expression = cp.sum_squares(u - u_ref_param)
        for r in clf_relaxations:
            objective_expression += cp.multiply(clf_relaxation_penalty_param, r)
        objective = cp.Minimize(objective_expression)

        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        variables = [u] + clf_relaxations
        parameters = Lf_V_params + Lg_V_params
        parameters += [V_param, u_ref_param, clf_relaxation_penalty_param]
        self.differentiable_qp_solver = CvxpyLayer(
            problem, variables=variables, parameters=parameters
        )


    def _solve_CLF_QP_cvxpylayers(
        self,
        x: torch.Tensor,
        u_ref: torch.Tensor,
        V: torch.Tensor,
        Lf_V: torch.Tensor,
        Lg_V: torch.Tensor,
        relaxation_penalty: float,
        epsilon : float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP. Solves the QP using
        CVXPyLayers, which does allow for backpropagation, but is slower and less
        accurate than Gurobi.

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
        # The differentiable solver must allow relaxation
        relaxation_penalty = min(relaxation_penalty, 1e6)

        # Assemble list of params
        self.n_scenarios = 1
        params = []
        for i in range(self.n_scenarios):
            params.append(Lf_V[:, :])
        for i in range(self.n_scenarios):
            params.append(Lg_V[:, :])
        params.append(V.reshape(-1, 1))
        params.append(u_ref)
        params.append(torch.tensor([relaxation_penalty]).type_as(x))

        # We've already created a parameterized QP solver, so we can use that
        result = self.differentiable_qp_solver(
            *params,
            solver_args={"max_iters": 50000000},
        )

        # Extract the results
        u_result = result[0]
        r_result = torch.hstack(result[1:])

        return u_result.type_as(x), r_result.type_as(x)


    def solve_CLF_QP(
        self,
        x,
        u_ref: Optional[torch.Tensor] = None,
        relaxation_penalty: Optional[float] = 1000,
        requires_grad: bool = False,
        epsilon: float = 0.1,
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
        H = self.h(x)
        Lf_V, Lg_V = self.V_lie_derivatives(x)

        # Get the reference control input as well
        if u_ref is not None:
            err_message = f"u_ref must have {x.shape[0]} rows, but got {u_ref.shape[0]}"
            assert u_ref.shape[0] == x.shape[0], err_message
            err_message = f"u_ref must have {1} cols,"
            err_message += f" but got {u_ref.shape[1]}"
            assert u_ref.shape[1] == 1, err_message
        else:
            u_ref = self.nominal_controller(x)
            err_message = f"u_ref shouldn't be None!!!!"
            assert u_ref is not None, err_message
        
        if requires_grad:
            return self._solve_CLF_QP_cvxpylayers(
                x, u_ref, H, Lf_V, Lg_V, relaxation_penalty, epsilon=epsilon
            )
        else:
            return self._solve_CLF_QP_gurobi(
            x, u_ref, H, Lf_V, Lg_V, relaxation_penalty, epsilon=epsilon
        )
    

    def _solve_CLF_QP_gurobi(
        self,
        x: torch.Tensor,
        u_ref: torch.Tensor,
        V: torch.Tensor,
        Lf_V: torch.Tensor,
        Lg_V: torch.Tensor,
        relaxation_penalty: float,
        epsilon : float = 0.1,
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
            lower_lim, upper_lim = self.dynamic_system.control_limits
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
            clf_constraint = -(Lf_V_np + Lg_V_np * u + 0.5 * V_np - epsilon)
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




    def V_loss(self, 
        s: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        ) -> List[Tuple[str, torch.Tensor]]:


        C_mask = self.h(s) >= 0
        C_mask = torch.squeeze(C_mask, dim=-1)
        
        #C_C_mask = ~C_mask

        unsafe_mask = self.dynamic_system.unsafe_mask(s)
        unsafe_mask = torch.squeeze(unsafe_mask, dim=-1)
        
        safe_mask = self.dynamic_system.safe_mask(s)
        safe_mask = torch.squeeze(safe_mask, dim=-1)

        unit_safe_C = torch.hstack(( torch.unsqueeze(safe_mask,dim=1)  , torch.unsqueeze(~C_mask, dim=1)))
        A_slash_C_mask = torch.all(unit_safe_C, dim=-1)

        V_s = self.V(s)
        loss = []

        if C_mask.sum() > 0:
            V_s_C = V_s[C_mask]
            loss_1 = torch.mean( torch.pow(V_s_C - 1, 2) )
            loss.append(("V loss in safe set", loss_1))
        
        if unsafe_mask.sum() > 0:
            V_s_A_C =V_s[unsafe_mask]
            loss_2 = torch.mean( torch.pow(V_s_A_C + 1, 2) )
            loss.append(("V loss in unsafe set", loss_2))

        if A_slash_C_mask.sum() > 0:
            V_s_A_slash_C = V_s[A_slash_C_mask]

            u_star_list = [ torch.Tensor([u_v[i]]).float() for i in self.u_star_index]
            u_star = torch.unsqueeze(torch.hstack(u_star_list), dim=1)
            u_star_CC = torch.unsqueeze(u_star[A_slash_C_mask], dim=1).to(s.device)

            s_else = s[A_slash_C_mask]

            ds = self.dynamic_system.f(s_else) + self.dynamic_system.g(s_else) * u_star_CC
            s_plus = s_else + ds * self.dt
            loss_3 = torch.mean( torch.pow( V_s_A_slash_C - (0.5 + self.gamma * self.V(s_plus))  , 2) ) 
            # loss.append(("V loss in feasible set", loss_3))


        # V_s = self.V(s)

        # s_C = s[C_mask, :]
        # V_s_C = V_s[C_mask, :]

        # s_else = s[~C_mask, :] 
        # V_s_else = V_s[~C_mask, :]

        # if C_mask.sum() > 0:
        #     V_s_C_bar = torch.mean(torch.sign( self.dynamic_system.state_constraints(s_C) ) + 1, dim=-1)
        #     loss_1 = torch.mean( torch.pow(V_s_C - V_s_C_bar, 2) ) 
        #     # loss_1 = torch.mean( torch.pow(V_s_C - 1, 2) ) 
        # else:
        #     loss_1 = None

        # if  C_C_mask.sum() > 0:
        #     u_star_list = [ torch.Tensor([u_v[i]]).float() for i in self.u_star_index]
        #     u_star = torch.unsqueeze(torch.hstack(u_star_list), dim=1)
        #     u_star_CC = torch.unsqueeze(u_star[C_C_mask], dim=1).to(s.device)

        #     ds = self.dynamic_system.f(s_else) + self.dynamic_system.g(s_else) * u_star_CC
        #     s_plus = s_else + ds * self.dt
        #     V_s_else_bar = torch.mean(torch.sign( self.dynamic_system.state_constraints(s_else) ) + 1, dim=-1) + self.gamma * self.V(s_plus)
        #     loss_2 = torch.mean( torch.pow( V_s_else - V_s_else_bar, 2 ) )
        #     # loss_2 = torch.mean( torch.pow( V_s_else - (-1), 2 ) )
        # else:
        #     loss_2 = None

        # loss = []
        # if loss_1 is not None:
        #     loss.append(("V loss in safe set", loss_1))
        # if loss_2 is not None:
        #     loss.append(("V loss outside safe set", loss_2))
        
        return loss

    def Dt(self, s, u_vi):
        Lf_h, Lg_h = self.V_lie_derivatives(s)

        return Lf_h + Lg_h * u_vi

    def gradient_descent_condition(self, s, u_vi, alpha=0.5):
        output = self.h(s)
        dt = self.Dt(s, u_vi)

        result = dt + alpha * output
        
        return result

    def gradient_descent_violation(self, s, u_pi):
        
        violation = self.gradient_descent_condition(s, u_pi)
        
        return violation


    def normalize(
        self, x: torch.Tensor, k: float = 1.0
    ) -> torch.Tensor:
        """Normalize the state input to [-k, k]

        args:
            dynamics_model: the dynamics model matching the provided states
            x: bs x self.dynamics_model.n_dims the points to normalize
            k: normalize non-angle dimensions to [-k, k]
        """
        x_min, x_max = self.dynamic_system.domain_limits
        x_center = (x_max + x_min) / 2.0
        x_range = (x_max - x_min) / 2.0
        # Scale to get the input between (-k, k), centered at 0
        x_range = x_range / k


        # Do the normalization
        return (x - x_center.type_as(x)) / x_range.type_as(x)

    def nominal_controller(self, s):
        batch_size = s.shape[0]

        K = torch.ones(batch_size, self.dynamic_system.nu, self.dynamic_system.ns) * torch.unsqueeze(self.K, dim=0)
        K = K.to(s.device)

        u = -torch.bmm(K, s.unsqueeze(dim=-1))
        # u_lower_bd ,  u_upper_bd = self.dynamic_system.control_limits
        # u = torch.clip(u, u_lower_bd.to(s.device), u_upper_bd.to(s.device))
        return u.squeeze(dim=-1)

    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, safe_mask, unsafe_mask = batch

        # Compute the losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, safe_mask, unsafe_mask)
        )

        if self.current_epoch > self.learn_shape_epochs:
            component_losses.update(
                self.descent_loss(
                    x, safe_mask, unsafe_mask,requires_grad=False
                )
            )
            # component_losses.update(
            #      self.V_loss(x, safe_mask, unsafe_mask)
            # )

        # component_losses = {}
        # component_losses.update(
        #     self.test_loss(x, safe_mask, unsafe_mask)
        # )

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        batch_dict = {"loss": total_loss, **component_losses}

        return batch_dict
    

    def training_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Outputs contains a list for each optimizer, and we need to collect the losses
        # from all of them if there is a nested list
        if isinstance(outputs[0], list):
            outputs = itertools.chain(*outputs)

        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        if self.current_epoch > self.learn_shape_epochs:
            self.log("Total loss / train", avg_losses["loss"], sync_dist=True)
            print(f"\n the overall loss of this training epoch {avg_losses['loss']}\n")
            self.train_loss.append(avg_losses['loss'])
            # And all component losses
            for loss_key in avg_losses.keys():
                # We already logged overall loss, so skip that here
                if loss_key == "loss":
                    continue
                # Log the other losses
                self.log(loss_key + " / train", avg_losses[loss_key], sync_dist=True)


    def validation_step(self, batch, batch_idx):
        """Conduct the validation step for the given batch"""
        # Extract the input and masks from the batch
        x, safe_mask, unsafe_mask = batch

        # Get the various losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, safe_mask, unsafe_mask, accuracy=True)
        )
        if self.current_epoch > self.learn_shape_epochs:
            component_losses.update(
                self.descent_loss(x, safe_mask, unsafe_mask, accuracy=True)
            )

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        # Also compute the accuracy associated with each loss
        # component_losses.update(
        #     self.boundary_loss(x, safe_mask, unsafe_mask)
        # )
        # if self.current_epoch > self.learn_shape_epochs:
        #     component_losses.update(
        #         self.descent_loss(x, safe_mask, unsafe_mask)
        #     )

        batch_dict = {"val_loss": total_loss, **component_losses}

        return batch_dict

    
    def validation_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        if self.current_epoch > self.learn_shape_epochs:
            self.log("Total loss / val", avg_losses["val_loss"], sync_dist=True)
            print(f"\n the overall loss of this validation epoch {avg_losses['val_loss']}\n")
            print(f"\n the descent accuracy of this validation epoch {avg_losses['CLBF descent accuracy (linearized)']}\n")
            
            self.val_loss.append(avg_losses['val_loss'])
            # And all component losses
            for loss_key in avg_losses.keys():
                # We already logged overall loss, so skip that here
                if loss_key == "val_loss":
                    continue
                # Log the other losses
                self.log(loss_key + " / val", avg_losses[loss_key], sync_dist=True)
        else:
            self.log("Total loss / val", 10, sync_dist=True)
        # **Now entering spicetacular automation zone**
        # We automatically run experiments every few epochs

        # Only plot every 5 epochs
        if self.current_epoch % 5 != 0:
            return

        # self.experiment_suite.run_all_and_log_plots(
        #     self, self.logger, self.current_epoch
        # )

    def test_step(self, batch, batch_idx):
        """Conduct the validation step for the given batch"""
        # Extract the input and masks from the batch
        x, safe_mask, unsafe_mask = batch

        # Get the various losses
        batch_dict = {"shape_h": {},"safe_violation": {}, "unsafe_violation": {}, "descent_violation": {} }
        
        # record shape_h
        h_x = self.h(x)
        batch_dict["shape_h"]["state"] = x
        batch_dict["shape_h"]["val"] = h_x

        # record safe_violation
        

        h_x_neg_mask = h_x < 0 
        unit_index= torch.hstack((safe_mask.unsqueeze(dim=1), h_x_neg_mask))

        h_x_safe_violation_indx = torch.all(unit_index, dim=1)
        ## record states and its value
        s_safe_violation = x[h_x_safe_violation_indx]
        h_s_safe_violation = h_x[h_x_safe_violation_indx]

        batch_dict["safe_violation"]["state"] = s_safe_violation
        batch_dict["safe_violation"]["val"] = h_s_safe_violation

        # record unsafe_violation
        h_x_pos_mask = h_x >= 0
        unit_index = torch.hstack((unsafe_mask.unsqueeze(dim=1), h_x_pos_mask))

        h_x_unsafe_violation_indx = torch.all(unit_index, dim=1)
        ##  record states and its value
        s_unsafe_violation = x[h_x_unsafe_violation_indx]
        h_s_unsafe_violation = h_x[h_x_unsafe_violation_indx]

        batch_dict["unsafe_violation"]["state"] = s_safe_violation
        batch_dict["unsafe_violation"]["val"] = h_s_safe_violation

        # record descent_violation
        c_list = [ self.gradient_descent_violation(x, u_i) for u_i in self.u_v ]
        c_list = torch.hstack(c_list)

        descent_violation, u_star_index = torch.max(c_list, dim=1)

        descent_violation_mask = torch.logical_and(descent_violation < 0, torch.logical_not(unsafe_mask))  

        s_descent_violation = x[descent_violation_mask]
        u_star_index_descent_violation = u_star_index[descent_violation_mask]
        h_s_descent_violation = h_x[descent_violation_mask]
        Lf_h_descent_violation, Lg_h_descent_violation = self.V_lie_derivatives(s_descent_violation)

        batch_dict["descent_violation"]["state"] = s_descent_violation
        batch_dict["descent_violation"]["val"] = h_s_descent_violation
        batch_dict["descent_violation"]["Lf_h"] =  Lf_h_descent_violation
        batch_dict["descent_violation"]["Lg_h"] = Lg_h_descent_violation
        batch_dict["descent_violation"]["u_star_index"] = u_star_index_descent_violation

        return batch_dict

    def test_epoch_end(self, outputs):
        
        print("test_epoch_end")
        torch.save(outputs, "test_results.pt")
        
        print("results is save to test_result.pt")
        


    def configure_optimizers(self):
        clbf_params = list(self.h.parameters())  #+ list(self.V.parameters())
        clbf_opt = torch.optim.SGD(
            clbf_params,
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        self.opt_idx_dict = {0: "clbf"}

        return [clbf_opt]



if __name__ == "__main__":

        
    s0 = -torch.rand(3, 2, dtype=torch.float)

    data_module = DataModule(system=inverted_pendulum_1, train_grid_gap=0.5, test_grid_gap=0.3)

    NN = NeuralNetwork(dynamic_system=inverted_pendulum_1, data_module=data_module)

    NN.prepare_data()

    u_nominal = NN.nominal_controller(s0)
    NN.boundary_loss(NN.data_module.s_training, NN.data_module.safe_mask_training, NN.data_module.unsafe_mask_training, accuracy=True)
    NN.descent_loss(NN.data_module.s_training, NN.data_module.safe_mask_training, NN.data_module.unsafe_mask_training, requires_grad=True)
    # NN.V_loss(NN.data_module.s_training, NN.data_module.safe_mask_training, NN.data_module.unsafe_mask_training)

    del NN

# def myCustomLoss(h_bar_s, C_bar_s, epsilon = 0.01):
    
#     return (torch.sgn(h_bar_s) + 1)/2 * torch.relu(-C_bar_s+epsilon)





