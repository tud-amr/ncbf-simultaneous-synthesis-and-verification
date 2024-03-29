import numpy as np
import itertools
from typing import Tuple, List, Optional
import json
import time
import copy
import itertools
from itertools import product

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.autograd import grad

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gurobipy as gp
from gurobipy import GRB
from qpth.qp import QPFunction

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from collections import defaultdict

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt

from safe_rl_cbf.Dataset.DataModule import DataModule
# from ValueFunctionNeuralNetwork import ValueFunctionNeuralNetwork
from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem
from safe_rl_cbf.Models.Transformer import Transformer

from safe_rl_cbf.Dynamics.dynamic_system_instances import car1, inverted_pendulum_1

######################### define neural network #########################


class NeuralCLBF(pl.LightningModule):
    def __init__(
        self,
        dynamic_system: ControlAffineSystem,
        data_module: DataModule,
        # value_function: ValueFunctionNeuralNetwork,
        require_grad_descent_loss :bool = True,
        learn_shape_epochs: int = 10,
        train_mode: int = 0,
        primal_learning_rate: float = 1e-3,
        gamma: float = 0.9,
        clf_lambda: float = 0.5,
        clf_relaxation_penalty: float = 50.0,
        ):

        super(NeuralCLBF, self).__init__()
        self.dynamic_system = dynamic_system
        self.data_module = data_module
        self.learn_shape_epochs = learn_shape_epochs
        self.train_mode = train_mode
        if train_mode == 0:
            self.learning_rate = primal_learning_rate
        elif train_mode == 1:
            self.learning_rate = 1e-5
        elif train_mode == 2:
            self.learning_rate = 5e-6

        self.dt = self.dynamic_system.dt
        self.gamma = gamma
        self.beta = 0.99
       
        self.require_grad_descent_loss = require_grad_descent_loss
        # self.h = nn.Sequential(
        #         nn.Linear(self.dynamic_system.ns, 256),
        #         nn.ReLU(),
        #         nn.Linear(256, 256),
        #         nn.ReLU(),
        #         nn.Linear(256, 256),
        #         nn.ReLU(),
        #         nn.Linear(256,1)
        #     )

        # self.h = nn.Sequential(
        #         nn.Linear(self.dynamic_system.ns, 256),
        #         nn.Tanh(),
        #         nn.Linear(256, 256),
        #         nn.Tanh(),
        #         nn.Linear(256, 256),
        #         nn.Tanh(),
        #         nn.Linear(256,1)
        #     )

        self.h = nn.Sequential(
                nn.Linear(self.dynamic_system.ns, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )
        
        self.param_init(self.h)
    
        self.h0 = copy.deepcopy(self.h)
        
        u_lower, u_upper = self.dynamic_system.control_limits
        dir = torch.tensor([0, 1])
        combine = list(product(dir, repeat=self.dynamic_system.nu))
        self.u_v = []
        if self.dynamic_system.nu != 0:
            for i in range(len(combine)):
                coefficent = torch.tensor(combine[i])
                u_i = coefficent * u_upper + (1 - coefficent) * u_lower
                self.u_v.append(u_i.reshape(1, -1))

        d_lower, d_upper = self.dynamic_system.disturbance_limits
        dir = torch.tensor([0, 1])
        combine = list(product(dir, repeat=self.dynamic_system.nd))
        self.d_v = []
        if self.dynamic_system.nd != 0:
            for i in range(len(combine)):
                coefficent = torch.tensor(combine[i])
                d_i = coefficent * d_upper + (1 - coefficent) * d_lower
                self.d_v.append(d_i.reshape(1, -1))
        
        
        # self.u_v = torch.cat(self.u_v, dim=0)
        
        # self.K = self.dynamic_system.K
        self.clf_lambda = clf_lambda
    
        self.clf_relaxation_penalty = clf_relaxation_penalty
        self.training_stage = 0

        # self.generate_cvx_solver()
        # self.generate_cvx_solver2()
        # self.generate_convex_relaxation_solver()

        self.use_h0 = False
        self.training_record_dict = {"performance_loss": [], "safety_loss": [], "descent_loss": [],
                                      "performance_grad":[], "safety_grad": [], "descent_grad_mean": [], "descent_grad_max": [],
                                       "angles":[], "angles2": [], 
                                       "violation_states": [], "violation_value": []}
    
        self.descent_loss_name = ['descent_term_linearized', 'descent_term_simulated', 'hji_vi_descent_loss_term']
        self.safety_loss_name = ['unsafe_region_term','unsafe_area_hji_term', 'regulation_loss_term', 'below_violation']
        self.performance_lose_name = ['safe_region_term', 'descent_boundary_term', 'hji_vi_boundary_loss_term']

        self.coefficient_for_performance_loss = 1
        self.coefficient_for_inadmissible_state_loss = 5
        self.coefficients_for_descent_loss = 5

        self.hji_vi_boundary_loss_term = torch.tensor([0.0])
        self.hji_vi_descent_loss_term = torch.tensor([0.0])

    @staticmethod
    def param_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform(m.weight) 

    def prepare_data(self):
        return self.data_module.prepare_data()

    def setup(self, stage: Optional[str] = None):
        return self.data_module.setup(stage)

    def train_dataloader(self):
        self.data_module.set_dataset()
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def set_previous_cbf(self, h):
        self.h0.load_state_dict(h.state_dict())
        self.use_h0 = True
        
    def forward(self, s):
        hs = self.h(s)

        return hs

    def jacobian(self, y, x):
        ''' jacobian of y wrt x '''
        # meta_batch_size, num_observations = y.shape[:2]
        # jac = torch.zeros(meta_batch_size, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size*num_points, 2, 2)
        # for i in range(y.shape[-1]):
        #     # calculate dydx over batches for each feature value of y
        #     y_flat = y[...,i].view(-1, 1)
        #     jac[:, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

        # status = 0
        # if torch.any(torch.isnan(jac)):
        #     status = -1
        x_norm = x 

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

       
        return JV
    
    def get_max_grad_with_params(self, loss):

        grads = []
        for param in self.h.parameters():
            try:
                grads.append( torch.max( torch.abs( torch.flatten(grad(loss, param, torch.ones_like(loss), create_graph=True )[0]) )).detach() )
            except:
                grads.append(torch.zeros(1).to(loss.device))

        grads = torch.hstack(grads)
        grads = [ torch.max(grads) ]
        return grads
    
    def get_mean_grad_with_params(self, loss):

        grads = []
        for param in self.h.parameters():
            try:
                grads.append( torch.mean( torch.abs( torch.flatten(grad(loss, param, torch.ones_like(loss), create_graph=True )[0]) )).detach() )
            except:
                grads.append(torch.zeros(1).to(loss.device))

        grads = torch.hstack(grads)
        grads = [ torch.mean(grads) ]
        return grads

    def get_update_direction_between_losses(self, loss1, loss2):
        grads_1 = []
        grads_2 = []
        for param in self.h.parameters():
            try:
                grads_1.append(  torch.flatten(grad(loss1, param, torch.ones_like(loss1), create_graph=True )[0]).detach() )
                grads_2.append(  torch.flatten(grad(loss2, param, torch.ones_like(loss2), create_graph=True )[0]).detach() )
   
            except:
                grads_1.append(torch.zeros(1).to(loss1.device))
                grads_2.append(torch.zeros(1).to(loss2.device))
        
        angles = []
        for i in range(len(grads_2)):
            theta_1 = grads_1[i]
            theta_2 = grads_2[i]
            inner_product = (theta_1 * theta_2).sum()
            theta_1_norm = torch.norm(theta_1)
            theta_2_norm = torch.norm(theta_2)
            cos = inner_product / (2*theta_1_norm * theta_2_norm)
            if torch.isnan(cos) or torch.isinf(cos) or cos.nelement() == 0:
                cos = torch.zeros(1).to(loss1.device)    
            
            angles.append(cos)
        
        angles = torch.stack(angles)
        angles = angles.mean()
        return angles


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

        x_norm = x 

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
        self, 
        x: torch.Tensor,
        gradh: torch.Tensor
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
        # _, gradh = self.V_with_jacobian(x)
        # print(f"gradh shape is {gradh.shape}")
        # We need to compute Lie derivatives for each scenario
        batch_size = x.shape[0]
        Lf_V = torch.zeros(batch_size, 2, 1)
        Lg_V = torch.zeros(batch_size, 2, 1)

        f = self.dynamic_system.f(x)
        g = self.dynamic_system.g(x)
        d = self.dynamic_system.d(x)
        # print(f"f shape is {f.shape}")
        # print(f"g shape is {g.shape}")

        Lf_V = torch.bmm(gradh, f.unsqueeze(dim=-1)).squeeze(1)
        Lg_V = torch.bmm(gradh, g).squeeze(1)
        Ld_V = torch.bmm(gradh, d).squeeze(1)
        # print(f"Lf_V shape is {Lf_V.shape}")
        # print(f"Lg_V shape is {Lg_V.shape}")
        
        return Lf_V, Lg_V, Ld_V

    def boundary_loss(
        self,
        s : torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        coefficient_for_safe_state_loss: float=1e2,
        coefficient_for_unsafe_state_loss: float=1e2,
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
        
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        hs = self.h(s)

        # 1.) h > 0 in the safe region
        hs_safe = hs[safe_mask]
        eps = 1 # torch.min(self.dynamic_system.state_constraints(s[torch.logical_not(unsafe_mask)]), dim=1, keepdim=True).values
        
        safe_violation = coefficient_for_safe_state_loss *  F.relu( eps - hs_safe)
        safe_hs_term =  safe_violation.mean()
        if not torch.isnan(safe_hs_term):
            loss.append(("safe_region_term", safe_hs_term))
        # if accuracy:
        #     safe_V_acc = (safe_violation >= eps).sum() / safe_violation.nelement()
        #     loss.append(("CLBF_safe_region_accuracy", safe_V_acc))

        #   3.) h < 0 in the unsafe region

       

        hs_unsafe = hs[unsafe_mask]
        eps = 1 # -torch.min(self.dynamic_system.state_constraints(s[unsafe_mask]), dim=1, keepdim=True).values
        
        unsafe_violation = coefficient_for_unsafe_state_loss *  F.relu(eps + hs_unsafe)
        
        unsafe_hs_term = unsafe_violation.mean() 
        if not torch.isnan(unsafe_hs_term):
            loss.append(("unsafe_region_term", unsafe_hs_term))
        # if accuracy:
        #         unsafe_V_acc = (
        #             unsafe_violation >= eps 
        #         ).sum() / unsafe_violation.nelement()
        #         loss.append(("CLBF_unsafe_region_accuracy", unsafe_V_acc))

        # print(f"safe_boundary_loss_term: {safe_hs_term}")
        # print(f"unsafe_boundary_loss_term: {unsafe_hs_term}")


        return loss
    
    def descent_loss(
        self,
        s: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        hs: torch.Tensor,
        gradh: torch.Tensor,
        coefficients_descent_loss: float=1e2,
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
        

        
        condition_active = torch.sigmoid(10 * (1.0 + eps - hs))

        # u_qp, qp_relaxation = self.solve_CLF_QP(s, gradh, requires_grad=requires_grad, epsilon=0)
        # u_qp = self.solve_CLF_QP(s, requires_grad=requires_grad, epsilon=0)
        u_qp = self.nominal_controller(s)
        # qp_relaxation = torch.mean(qp_relaxation, dim=-1)

        # ####### Minimize the qp relaxation to encourage satisfying the decrease condition #################
        # qp_relaxation_loss = qp_relaxation[torch.logical_not(unsafe_mask)].mean()
        # loss.append(("QP_relaxation", qp_relaxation_loss))

        ############### Now compute the decrease using linearization #######################
        eps = 0.1
        clbf_descent_term_lin = torch.tensor(0.0).type_as(s)
        clbf_descent_acc_lin = torch.tensor(0.0).type_as(s)
        # Get the current value of the CLBF and its Lie derivatives
        Lf_V, Lg_V, _ = self.V_lie_derivatives(s, gradh)
    
        # Use the dynamics to compute the derivative of V
        Vdot = Lf_V[:, :].unsqueeze(1) + torch.bmm(
            Lg_V[:, :].unsqueeze(1),
            u_qp.reshape(-1, self.dynamic_system.nu, 1),
        )
        Vdot = Vdot.reshape(hs.shape)
        violation = coefficients_descent_loss * F.relu(eps - (Vdot + self.clf_lambda * hs))
        violation = violation * condition_active
        clbf_descent_term_lin = clbf_descent_term_lin + violation[torch.logical_not(unsafe_mask)].mean()
        clbf_descent_acc_lin = clbf_descent_acc_lin + (violation >= eps).sum() / (
            violation.nelement()
        )

        loss.append(("descent_term_linearized", clbf_descent_term_lin))
        # if accuracy:
        #     loss.append(("CLBF_descent_accuracy_linearized", clbf_descent_acc_lin))


        ##################### Now compute the decrease using simulation ##########################
        eps = 0.1
        clbf_descent_term_sim = torch.tensor(0.0).type_as(s)
        clbf_descent_acc_sim = torch.tensor(0.0).type_as(s)
       
        # xdot = self.dynamics_model.closed_loop_dynamics(x, u_qp, params=s)

        x_next = self.dynamic_system.step(s, u_qp)
        H_next = self.h(x_next)
        violation = F.relu(
            eps - ((H_next - hs) / self.dynamic_system.dt + self.clf_lambda * hs)
        )
        violation = coefficients_descent_loss * violation * condition_active

        clbf_descent_term_sim = clbf_descent_term_sim + violation[torch.logical_not(unsafe_mask)].mean()
        clbf_descent_acc_sim = clbf_descent_acc_sim + (violation >= eps).sum() / (
            violation.nelement() 
        )

        loss.append(("descent_term_simulated", clbf_descent_term_sim))
        # if accuracy:
        #     loss.append(("CLBF_descent_accuracy_simulated", clbf_descent_acc_sim))

        

        return loss

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
            + clf_relaxations[0]
            >= 0
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


    def generate_cvx_solver2(self):
        # Save the other parameters
        

        # Since we want to be able to solve the CLF-QP differentiably, we need to set
        # up the CVXPyLayers optimization. First, we define variables for each control
        # input and the relaxation in each scenario
        u = cp.Variable(self.dynamic_system.nu)
       
        # Next, we define the parameters that will be supplied at solve-time: the value
        # of the Lyapunov function, its Lie derivatives, the relaxation penalty, and
        # the reference control input   
        
        
        Lg_V_params = []
        
       
        Lg_V_params.append(cp.Parameter(self.dynamic_system.nu))


        # These allow us to define the constraints
        constraints = []
        
        # CLF decrease constraint (with relaxation)
        # constraints.append(
        #     Lf_V_params[0]
        #     + Lg_V_params[0] @ u
        #     + self.clf_lambda * V_param
        #     + clf_relaxations[0]
        #     >= 0
        # )


        # Control limit constraints
        lower_lim,  upper_lim  = self.dynamic_system.control_limits
        for control_idx in range(self.dynamic_system.nu):
            constraints.append(u[control_idx] >= lower_lim[control_idx])
            constraints.append(u[control_idx] <= upper_lim[control_idx])


        # And define the objective
        objective_expression = cp.sum(Lg_V_params[0] * u)
        
        objective = cp.Minimize(objective_expression)

        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        variables = [u]
        parameters =  Lg_V_params
        
        self.differentiable_qp_solver2 = CvxpyLayer(
            problem, variables=variables, parameters=parameters
        )

    def generate_convex_relaxation_solver(self):
        # define optimization variables
        
        J = cp.Variable(self.dynamic_system.ns)
        dx = cp.Variable(self.dynamic_system.ns)
        h = cp.Variable(1)
        X = cp.Variable(self.dynamic_system.ns)
        
        # define parameters
        
        lb_j = cp.Parameter(self.dynamic_system.ns)
        ub_j = cp.Parameter(self.dynamic_system.ns)
        lb_dx = cp.Parameter(self.dynamic_system.ns)
        ub_dx = cp.Parameter(self.dynamic_system.ns)
        lb_h = cp.Parameter(1)
        ub_h = cp.Parameter(1)
        lb_j_lb_dx = cp.Parameter(self.dynamic_system.ns)
        ub_j_up_dx = cp.Parameter(self.dynamic_system.ns)
        lb_j_up_dx = cp.Parameter(self.dynamic_system.ns)
        ub_j_lb_dx = cp.Parameter(self.dynamic_system.ns)

        # define objective
        objective_expression = cp.sum(X) + 0.5 * h
        objective = cp.Minimize(objective_expression)

       

        # define constraints
        constraints = []
        constraints.append(lb_j <= J)
        constraints.append(J <= ub_j)
        constraints.append(lb_dx <= dx) 
        constraints.append(dx <= ub_dx)
        constraints.append(lb_h <= h)
        constraints.append(h <= ub_h)
        constraints.append(X - cp.multiply(lb_j, dx) - cp.multiply(lb_dx, J) >= - lb_j_lb_dx )
        constraints.append(X - cp.multiply(lb_j, dx) - cp.multiply(ub_dx, J) >= - ub_j_up_dx )
        constraints.append(X - cp.multiply(lb_j, dx) - cp.multiply(ub_dx, J) >= - lb_j_up_dx )
        constraints.append(X - cp.multiply(lb_dx, J) - cp.multiply(ub_j, dx) >= - ub_j_lb_dx )
        

        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        variables = [J, dx, h, X]
        parameters = [lb_j, ub_j, lb_dx, ub_dx, lb_h, ub_h, lb_j_lb_dx,ub_j_up_dx, lb_j_up_dx, ub_j_lb_dx]
        self.convex_relaxation_solver = CvxpyLayer(problem,  variables=variables, parameters=parameters)

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


    def _solve_CLF_QP_cvxpylayers2(
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
            params.append(Lg_V[:, :])

        # We've already created a parameterized QP solver, so we can use that
        result = self.differentiable_qp_solver2(
            *params,
            solver_args={"max_iters": 50000000},
        )

        # Extract the results
        u_result = result[0]
        

        return u_result.type_as(x)

    def _solve_convex_relataxion(
        self,
        lb_j: torch.Tensor,
        ub_j: torch.Tensor,
        lb_dx: torch.Tensor,
        ub_dx: torch.Tensor,
        lb_h: torch.Tensor,
        ub_h: torch.Tensor,
        lb_j_lb_dx : torch.Tensor,
        ub_j_up_dx : torch.Tensor,
        lb_j_up_dx : torch.Tensor,
        ub_j_lb_dx : torch.Tensor,
    )   -> Tuple[torch.Tensor, torch.Tensor]:

        params = []
        params.append(lb_j)
        params.append(ub_j)
        params.append(lb_dx)
        params.append(ub_dx)
        params.append(lb_h)
        params.append(ub_h)
        params.append(lb_j_lb_dx)
        params.append(ub_j_up_dx)
        params.append(lb_j_up_dx)
        params.append(ub_j_lb_dx)

        # We've already created a parameterized QP solver, so we can use that
        result = self.convex_relaxation_solver(
            *params,
            solver_args={"max_iters": 50000000},
        )

        # Extract the results
        J_result = result[0]
        dx_result = result[1]
        h_result = result[2]
        X_result = result[3]
        

        return J_result, dx_result, X_result, h_result


    def solve_CLF_QP(
        self,
        x,
        gradh,
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
        Lf_V, Lg_V, Ld_V = self.V_lie_derivatives(x, gradh)

        # Get the reference control input as well
        if u_ref is not None:
            err_message = f"u_ref must have {x.shape[0]} rows, but got {u_ref.shape[0]}"
            assert u_ref.shape[0] == x.shape[0], err_message
            err_message = f"u_ref must have {self.dynamic_system.nu} cols,"
            err_message += f" but got {u_ref.shape[1]}"
            assert u_ref.shape[1] == self.dynamic_system.nu, err_message
        else:
            u_ref = self.nominal_controller(x)
            err_message = f"u_ref shouldn't be None!!!!"
            assert u_ref is not None, err_message
        
        if requires_grad:
            # return self._solve_CLF_QP_cvxpylayers(
            #     x, u_ref, H, Lf_V, Lg_V, relaxation_penalty, epsilon=epsilon
            # )

            # return self._solve_CLF_QP_cvxpylayers2(
            #     x, u_ref, H, Lf_V, Lg_V, relaxation_penalty, epsilon=epsilon
            # )

            return self._solve_CLF_QP_OptNet(
                x, u_ref, H, Lf_V, Lg_V, relaxation_penalty, epsilon=epsilon
            )

            # return self._solve_CLF_QP_OptNet2(
            #     x, u_ref, H, Lf_V, Lg_V, relaxation_penalty, epsilon=epsilon
            # )
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

        n_controls = self.dynamic_system.nu
        
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
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with gp.Model("clf_qp", env=env) as model:
                    
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
                
                    Lg_V_np = Lg_V[batch_idx, :].detach().cpu().numpy()
                    Lf_V_np = Lf_V[batch_idx, 0].detach().cpu().numpy()
                    V_np = V[batch_idx, 0].detach().cpu().numpy()
                    clf_constraint = -(Lf_V_np + Lg_V_np @ u + 1 * V_np - epsilon)
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

    def _solve_CLF_QP_OptNet(self,
        x: torch.Tensor,
        u_ref: torch.Tensor,
        V: torch.Tensor,
        Lf_V: torch.Tensor,
        Lg_V: torch.Tensor,
        relaxation_penalty: float,
        epsilon : float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        bs = x.shape[0]
        nu = self.dynamic_system.nu
        nv = nu + 1
        nieq = 4

        u_min, u_max = self.dynamic_system.control_limits

        diag_Q =  torch.ones(nv).float().to(x.device)
        diag_Q[-1] = 1000  # 2 * torch.Tensor([1, 1000]).float().to(x.device)
        Q = torch.diag(diag_Q).reshape(nv, nv)
        Q = Variable(Q.expand(bs, nv, nv))
        # print(f"Q shape is {Q.shape} \n")

        p = Variable(torch.zeros(bs, nv).float().to(x.device))
        # print(f"u_ref shape is {u_ref.shape}")
        p[:, 0] = -2 * u_ref.squeeze(-1)
        # print(f"p shape is {p.shape} \n")

        # print(f"Lf_h shape is {Lf_V.shape}")
        # print(f"Lg_h shape is {Lg_V.shape}")
        
        G = Variable(torch.zeros(bs, nieq, nv).to(x.device))
        G[:, 0, 0:nu] = -Lg_V
        G[:, 0, nu:nv] = -1
        G[:, 1, 0:nu] = -1
        G[:, 2, 0:nu] = 1
        G[:, 3, nu:nv] = -1
        # print(f"G shape is {G.shape}")
        # print(f"Lg_V is {Lg_V}")
        # print(f"G is {G}")

        h = Variable(torch.zeros(bs, nieq).to(x.device))
        h[:, 0] = Lf_V.squeeze(-1) + 0.5 * V.squeeze(-1) - epsilon
        h[:, 1] = -u_min
        h[:, 2] = u_max
        h[:, 3] = 0
        # print(f"h shape is {h.shape}")
        # print(f"h is {h}")

        e = Variable(torch.Tensor())

        u_delta = QPFunction(verbose=0)(Q, p, G, h, e, e)
        # print(f"u_delta shape is {u_delta.shape}")
        # print(f"u_delta is {u_delta}")
        
        return u_delta[:, 0:1], u_delta[:, 1:2]


    def _solve_CLF_QP_OptNet2(self,
        x: torch.Tensor,
        u_ref: torch.Tensor,
        V: torch.Tensor,
        Lf_V: torch.Tensor,
        Lg_V: torch.Tensor,
        relaxation_penalty: float,
        epsilon : float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        bs = x.shape[0]
        nu = self.dynamic_system.nu
        nv = nu 
        nieq = 2 * nu

        u_min, u_max = self.dynamic_system.control_limits

        diag_Q =  torch.ones(nv).float().to(x.device)
        Q = 0.0001 * torch.diag(diag_Q).reshape(nv, nv)
        Q = Variable(Q.expand(bs, nv, nv))
        # print(f"Q shape is {Q.shape} \n")

        p = Variable(torch.zeros(bs, nv).float().to(x.device))
        # print(f"u_ref shape is {u_ref.shape}")
        p[:, 0:nv] = -Lg_V
        # print(f"p shape is {p.shape} \n")

        # print(f"Lf_h shape is {Lf_V.shape}")
        # print(f"Lg_h shape is {Lg_V.shape}")
        
        G = Variable(torch.zeros(bs, nieq, nv).to(x.device))
        G[:, 0, 0:nu] = -1
        G[:, 1, 0:nu] = 1
       
        # print(f"G shape is {G.shape}")
        # print(f"Lg_V is {Lg_V}")
        # print(f"G is {G}")

        h = Variable(torch.zeros(bs, nieq).to(x.device))
        h[:, 0] = -u_min
        h[:, 1] = u_max
        # print(f"h shape is {h.shape}")
        # print(f"h is {h}")

        e = Variable(torch.Tensor())

        u_delta = QPFunction(verbose=0)(Q, p, G, h, e, e)
        # print(f"u_delta shape is {u_delta.shape}")
        # print(f"u_delta is {u_delta}")
        
        return u_delta[:, 0:1]


    def _solve_convex_relataxion_OptNet(
        self,
        x : torch.Tensor,
        lb_j: torch.Tensor,
        ub_j: torch.Tensor,
        lb_dx: torch.Tensor,
        ub_dx: torch.Tensor,
        lb_h: torch.Tensor,
        ub_h: torch.Tensor,
        lb_j_lb_dx : torch.Tensor,
        ub_j_ub_dx : torch.Tensor,
        lb_j_ub_dx : torch.Tensor,
        ub_j_lb_dx : torch.Tensor,
    )   -> Tuple[torch.Tensor, torch.Tensor]:

        bs = x.shape[0]
        ns = self.dynamic_system.ns
        nv = 3*ns + 1
        nieq = 8*ns + 2


        diag_Q = 0.00001 * torch.ones(nv).float().to(x.device)
        Q = torch.diag(diag_Q).reshape(nv, nv)
        Q = Variable(Q.expand(bs, nv, nv))

        p = Variable(torch.zeros(bs, nv).float().to(x.device))
        p[:, 2*ns:3*ns] = 1.0
        p[:, 3*ns:3*ns+1] = 0.5

        G = Variable(torch.zeros(bs, nieq, nv).to(x.device))
        # G[:, 0:ns, 0:ns] = torch.diag(lb_dx.squeeze(-1))
        # G[:, 0:ns, ns:2*ns] = torch.diag(lb_j.squeeze(-1))
        G[:, 0:ns, 2*ns:3*ns] = -torch.eye(ns).to(x.device)

        # G[:, ns:2*ns, 0:ns] = torch.diag(ub_dx.squeeze(-1))
        # G[:, ns:2*ns, ns:2*ns] = torch.diag(lb_j.squeeze(-1))
        # G[:, ns:2*ns, 2*ns:3*ns] = -torch.eye(ns).to(x.device)

        # G[:, 2*ns:3*ns, 0:ns] = torch.diag(ub_dx.squeeze(-1))
        # G[:, 2*ns:3*ns, ns:2*ns] = torch.diag(lb_j.squeeze(-1))
        # G[:, 2*ns:3*ns, 2*ns:3*ns] = -torch.eye(ns).to(x.device)

        # G[:, 3*ns:4*ns, 0:ns] = torch.diag(lb_dx.squeeze(-1))
        # G[:, 3*ns:4*ns, ns:2*ns] = torch.diag(ub_j.squeeze(-1))
        # G[:, 3*ns:4*ns, 2*ns:3*ns] = -torch.eye(ns).to(x.device)

        G[:, 4*ns:5*ns, 0:ns] = -torch.eye(ns).to(x.device)

        G[:, 5*ns:6*ns, 0:ns] = torch.eye(ns).to(x.device)

        G[:, 6*ns:7*ns, ns:2*ns] = -torch.eye(ns).to(x.device)

        G[:, 7*ns:8*ns, ns:2*ns] = torch.eye(ns).to(x.device)

        G[:, 8*ns:8*ns+1, 3*ns:3*ns+1] = -torch.eye(1).to(x.device)
        
        G[:, 8*ns+1:8*ns+2, 3*ns:3*ns+1] = torch.eye(1).to(x.device)

        h = Variable(torch.zeros(bs, nieq).to(x.device))

        # h[:, 0:ns] = lb_j_lb_dx.squeeze(-1)
        # h[:, ns:2*ns] = ub_j_ub_dx.squeeze(-1)
        # h[:, 2*ns:3*ns] = lb_j_ub_dx.squeeze(-1)
        # h[:, 3*ns:4*ns] = ub_j_lb_dx.squeeze(-1)
        h[:, 4*ns:5*ns] = -lb_j.squeeze(-1)
        h[:, 5*ns:6*ns] = ub_j.squeeze(-1)
        h[:, 6*ns:7*ns] = -lb_dx.squeeze(-1)
        h[:, 7*ns:8*ns] = ub_dx.squeeze(-1)
        h[:, 8*ns:8*ns+1] = -lb_h
        h[:, 8*ns+1:8*ns+2] = ub_h

        e = Variable(torch.Tensor())

        s_solution = QPFunction( verbose=1, maxIter=1000, notImprovedLim=200)(Q, p, G, h, e, e)
        
        J = s_solution[:, 0:ns]
        dx = s_solution[:, ns:2*ns]
        X = s_solution[:, 2*ns:3*ns]
        h = s_solution[:, 3*ns:3*ns+1]

        return J, dx, X, h

    def test(self, x, relaxation_penalty=1000, epsilon=0.1):
        H = self.h(x)
        Lf_V, Lg_V, Ld_V = self.V_lie_derivatives(x)
        u_ref = self.nominal_controller(x)

        u0, r0 = self._solve_CLF_QP_OptNet(x, u_ref, H, Lf_V, Lg_V, relaxation_penalty, epsilon=epsilon)

        u1, r1 = self._solve_CLF_QP_gurobi(
            x, u_ref, H, Lf_V, Lg_V, relaxation_penalty, epsilon=epsilon
        )

        print(f"u1 is {u1}")
        print(f"r1 is {r1}")
    

    def Dt(self, s, u_vi):
        hs = self.h(s)
        gradh = self.jacobian(hs, s)

        Lf_h, Lg_h, Ld_V = self.V_lie_derivatives(s, gradh)
        u = u_vi.expand(Lg_h.shape[0], -1)
        Lg_h_u = Lg_h * u
        dt = Lf_h + Lg_h_u.sum(dim=1, keepdim=True)
        return dt

    def gradient_descent_condition(self, s, u_vi, alpha=0.5):
        output = self.h(s)
        dt = self.Dt(s, u_vi)

        alpha = self.clf_lambda
        result = dt + alpha * output
        
        return result

    def gradient_descent_violation(self, s, u_pi):
        
        violation = self.gradient_descent_condition(s, u_pi)
        
        return violation

    def get_nominal_control2(
        self,
        x: torch.Tensor,
        Lg_V: torch.Tensor,
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
        
       
        # Solve a QP for each row in x
        bs = x.shape[0]
        u_result = torch.zeros(bs, n_controls)
        for batch_idx in range(bs):
            # Skip any bad points
            if (
                torch.isnan(x[batch_idx]).any()
                or torch.isinf(x[batch_idx]).any()
                or torch.isnan(Lg_V[batch_idx]).any()
                or torch.isinf(Lg_V[batch_idx]).any()
            ):
                continue

            # Instantiate the model
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with gp.Model("clf_qp", env=env) as model:
                    
                    # Create variables for control input and (optionally) the relaxations
                    lower_lim, upper_lim = self.dynamic_system.control_limits
                    upper_lim = upper_lim.cpu().numpy()
                    lower_lim = lower_lim.cpu().numpy()
                    u = model.addMVar(n_controls, lb=lower_lim-2, ub=upper_lim+2)
                    

                    # Define the cost
                    Lg_V_np = Lg_V[batch_idx, 0].detach().cpu().numpy()
                    objective =   -(Lg_V_np * u)

                    # Optimize!
                    model.setParam("DualReductions", 0)
                    model.setObjective(objective, GRB.MINIMIZE)
                    model.optimize()

                    if model.status != GRB.OPTIMAL:
                        # Make the relaxations nan if the problem was infeasible, as a signal
                        # that something has gone wrong
                        continue

                    # Extract the results
                    for i in range(n_controls):
                        u_result[batch_idx, i] = torch.tensor(u[i].x)

        return u_result.type_as(x)


    def get_nominal_control(self, s):
        lower_lim, upper_lim = self.dynamic_system.control_limits
        control_area = torch.hstack((lower_lim.unsqueeze(dim=-1), upper_lim.unsqueeze(dim=-1)))
        # print(control_area.shape)
        
        nu = self.dynamic_system.nu
        vertices = list(itertools.product([0, 1], repeat=nu))
        vertices = torch.tensor(vertices)
        # print(vertices)

        u_list = []
        gs_list = []
        for v in vertices:
            index = torch.hstack( (torch.arange(0, nu, 1, dtype=torch.uint8).unsqueeze(dim=-1), v.unsqueeze(dim=-1)) )
            # print(index)
            u = control_area[index[:,0], index[:,1]]
            u_ex = u.expand(s.shape[0], nu).to(s.device)
            s_next = self.dynamic_system.step(s, u_ex, dt=0.1)
            gs_hat, _ = torch.min(self.dynamic_system.state_constraints(s_next), dim=1, keepdim=True)

            
            u_list.append(u)
            gs_list.append(gs_hat)

        u_list = torch.tensor(u_list)
        gs_list = torch.hstack(gs_list)

        _, index = torch.max(gs_list, dim=1)

        u = u_list[index]

        # print(s)
        # print(gs_list)
        # print(u)
        #print(index)

        # print(gs_list.shape)
        
        # print(u_list.shape)
        return u.unsqueeze(dim=1)
            



    def nominal_controller(self, s):
        batch_size = s.shape[0]
        
        K = torch.ones(batch_size, self.dynamic_system.nu, self.dynamic_system.ns) * torch.unsqueeze(self.dynamic_system.K, dim=0)
        K = K.to(s.device)

        u = -torch.bmm(K, s.unsqueeze(dim=-1))
        # u_lower_bd ,  u_upper_bd = self.dynamic_system.control_limits
        # u = torch.clip(u, u_lower_bd.to(s.device), u_upper_bd.to(s.device))
        
        # _, Lg_V = self.g.G_lie_derivatives(s)

        # u_new = self.get_nominal_control(s)
        
        # print(f"u = {u}, u_new = {u_new}")
        
        return u.squeeze(dim=-1)
    
    def on_train_start(self) -> None:
        print(f"############### Training start #########################")
        # self.param_init(self.h)

    def on_train_epoch_start(self) -> None:
        print("################## the epoch start #####################")
        self.epoch_start_time = time.time()
        print(f"current training stage is {self.training_stage}")
        self.training_record_dict["violation_states"].clear()
        self.training_record_dict["violation_value"].clear()

        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        s, safe_mask, unsafe_mask, grid_gap = batch
        

        # Compute the losses
        component_losses = {}

        s_random = s

        hs = self.h(s_random)
        gradh = self.jacobian(hs, s_random)
        
        if self.training_stage == 0:
            component_losses.update(
                self.boundary_loss(s, safe_mask, unsafe_mask)
            )
        

        if self.training_stage == 1:
            component_losses.update(
                self.boundary_loss(s, safe_mask, unsafe_mask, coefficient_for_safe_state_loss=40, coefficient_for_unsafe_state_loss=100)
            )
            
            component_losses.update(
                self.descent_loss(
                    s, safe_mask, unsafe_mask, hs, gradh, requires_grad=True, coefficients_descent_loss=2
                )
            )


        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(s)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        # print(f"current training stage is {self.training_stage}")

        batch_dict = {"loss": total_loss, **component_losses}


        return batch_dict
    

    def training_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Outputs contains a list for each optimizer, and we need to collect the losses
        # from all of them if there is a nested list

        self.epoch_end_time = time.time()

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

        descent_losses = 0.0
        safety_losses = 0.0
        performance_losses = 0.0
        
        
        for key in avg_losses.keys():
            if key in self.safety_loss_name:
                safety_losses += avg_losses[key]
            if key in self.performance_lose_name:
                performance_losses += avg_losses[key]
            if key in self.descent_loss_name:
                descent_losses += avg_losses[key]
        
        average_loss = safety_losses + performance_losses + descent_losses


        # Log the overall loss...
        # if self.current_epoch > self.learn_shape_epochs:
        self.log("Total_loss/train", average_loss)
        print(f"\n the overall loss of this training epoch {average_loss}\n")
        self.log("Epoch_time/train", self.epoch_end_time - self.epoch_start_time)
        print(f"the epoch time consume is {self.epoch_end_time - self.epoch_start_time}")
        self.log("Safety_loss/train", safety_losses)
        print(f"overall safety loss of this training epoch {safety_losses}")
        self.log("Descent_loss/train", descent_losses)
        print(f"overall descent loss of this training epoch {descent_losses}")
        self.log("Performance_loss/train", performance_losses)
        print(f"overall performance loss of this training epoch {performance_losses}")
        print(f"the adaptive coefficient is {self.coefficients_for_descent_loss}")

        print(f"hji_vi boundary loss is  {self.hji_vi_boundary_loss_term}")
        print(f"hji_vi descent loss is  {self.hji_vi_descent_loss_term}")

        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "loss":
                continue
            # Log the other losses
            self.log(loss_key + "/train", avg_losses[loss_key], sync_dist=True)

           
        if self.training_stage == 0 and (average_loss < 10): # warm up, shape h and g
            self.training_stage = 1  # start to consider condition 3
            self.epoch_record = self.current_epoch
        
        if self.training_stage == 1 and (self.current_epoch + 1) % (self.trainer.reload_dataloaders_every_n_epochs ) == 0:
            self.data_module.augment_dataset()
            torch.save(self.data_module.s_training, "s_training.pt")
        
        print(f"current training stage is {self.training_stage}")
        print(f"current learning rate is {self.trainer.optimizers[0].param_groups[0]['lr']}")

    def on_train_end(self) -> None:

        torch.save(self.training_record_dict, "training_record_dict.pt")

        return super().on_train_end()

    def test_step(self, batch, batch_idx):
        """Conduct the validation step for the given batch"""
        # Extract the input and masks from the batch
    
        x, safe_mask, unsafe_mask = batch
        
        # Get the various losses
        batch_dict = {"shape_h": {}, "unsafe_violation": {}, "descent_violation": {}, "inadmissible_boundary":{} }
        x.requires_grad_(True)
        
        # record shape_h
        h_x = self.h(x)
        gradh = self.jacobian(h_x, x)
        
        
        batch_dict["shape_h"]["state"] = x
        batch_dict["shape_h"]["val"] = h_x

        # record unsafe_violation
        h_x_neg_mask = h_x < -2
        h_x_pos_mask = h_x > 0
        unit_index = torch.hstack((unsafe_mask.unsqueeze(dim=1), h_x_pos_mask))

        h_x_unsafe_violation_indx = torch.all(unit_index, dim=1)
        ##  record states and its value
        s_unsafe_violation = x[h_x_unsafe_violation_indx]
        h_s_unsafe_violation = h_x[h_x_unsafe_violation_indx]

        batch_dict["unsafe_violation"]["state"] = s_unsafe_violation
        batch_dict["unsafe_violation"]["val"] = h_s_unsafe_violation

        # record descent_violation
        c_list = [ self.gradient_descent_violation(x, u_i.to(x.device)) for u_i in self.u_v ]
        c_list = torch.hstack(c_list)

        descent_violation, u_star_index = torch.max(c_list, dim=1)

        descent_violation_mask = torch.logical_and(descent_violation < 0,  torch.logical_not(unsafe_mask.squeeze(dim=-1)))  

        s_descent_violation = x[descent_violation_mask]
        u_star_index_descent_violation = u_star_index[descent_violation_mask]
        h_s_descent_violation = h_x[descent_violation_mask]
        Lf_h_descent_violation, Lg_h_descent_violation, _ = self.V_lie_derivatives(s_descent_violation, gradh[descent_violation_mask])

        batch_dict["descent_violation"]["state"] = s_descent_violation
        batch_dict["descent_violation"]["val"] = h_s_descent_violation
        batch_dict["descent_violation"]["Lf_h"] =  Lf_h_descent_violation
        batch_dict["descent_violation"]["Lg_h"] = Lg_h_descent_violation
        batch_dict["descent_violation"]["u_star_index"] = u_star_index_descent_violation

        # get safety boundary
        baseline = torch.min(self.dynamic_system.state_constraints(x), dim=1, keepdim=True).values
        
        inadmissible_boundary_index = torch.logical_and(baseline < 0, baseline > -0.01).squeeze(dim=-1)
        inadmissible_boundary_state = x[inadmissible_boundary_index]

        batch_dict["inadmissible_boundary"]["state"] = inadmissible_boundary_state

        return batch_dict

    def test_epoch_end(self, outputs):
        
        print("test_epoch_end")
        torch.save(outputs, "test_results.pt")
        
        print("results is save to test_result.pt")
        


    def configure_optimizers(self):
        clbf_params = list(self.h.parameters()) # list(self.g.parameters())
        clbf_opt = torch.optim.SGD(
            clbf_params,
            lr=self.learning_rate,
        )
        lr_scheduler = ExponentialLR(clbf_opt, gamma=0.995)
        self.opt_idx_dict = {0: "clbf"}

        return {"optimizer":clbf_opt, "lr_scheduler": lr_scheduler}



if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    s0 = torch.rand(5, 2, dtype=torch.float).to(device) * 3 - 1.5

    data_module = DataModule(system=inverted_pendulum_1, train_grid_gap=0.3, test_grid_gap=0.3)

    
    G = ValueFunctionNeuralNetwork.load_from_checkpoint("/home/wangxinyu/.mujoco/mujoco210/sunny_test/masterthesis_test/CBF_logs/train_g/lightning_logs/version_0/checkpoints/epoch=399-step=4800.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module)
    G.to(device)
   
    # NN = NeuralNetwork.load_from_checkpoint("/home/wangxinyu/.mujoco/mujoco210/sunny_test/masterthesis_test/CBF_logs/robust_training_maximum/lightning_logs/version_1/checkpoints/epoch=199-step=2400.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module, value_function=G)
    NN = NeuralNetwork(dynamic_system=inverted_pendulum_1, data_module=data_module, value_function=G).to(device)
    NN.to(device)

     # NN.test(s0)

    NN.prepare_data()
    n_sample = 10
    # NN.boundary_loss(NN.data_module.s_training, NN.data_module.safe_mask_training, NN.data_module.unsafe_mask_training, accuracy=True)
    NN.descent_loss(NN.data_module.s_training[0:n_sample,:].to(device), NN.data_module.safe_mask_training[0:n_sample].to(device), NN.data_module.unsafe_mask_training[0:n_sample].to(device), NN.data_module.grid_gap_training[0:n_sample].to(device),requires_grad=True)
    # NN.V_loss(NN.data_module.s_training, NN.data_module.safe_mask_training, NN.data_module.unsafe_mask_training)
    # NN.value_function_loss(NN.data_module.s_training[0:n_sample,:].to(device), NN.data_module.safe_mask_training[0:n_sample].to(device), NN.data_module.unsafe_mask_training[0:n_sample].to(device))
    del NN





