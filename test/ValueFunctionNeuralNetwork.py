import numpy as np
import itertools
from typing import Tuple, List, Optional
import json
import time
import copy

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.autograd import Variable
from torch.nn.parameter import Parameter

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

from Dataset.DataModule import DataModule
from Dynamics.control_affine_system import ControlAffineSystem

from Dynamics.dynamic_system_instances import car1, inverted_pendulum_1

######################### define neural network #########################


class ValueFunctionNeuralNetwork(pl.LightningModule):
    def __init__(
        self,
        dynamic_system: ControlAffineSystem,
        data_module: DataModule,
        require_grad_descent_loss :bool = True,
        learn_shape_epochs: int = 10,
        primal_learning_rate: float = 1e-3,
        gamma: float = 0.9,
        clf_lambda: float = 1.0,
        clf_relaxation_penalty: float = 50.0,
        ):

        super(ValueFunctionNeuralNetwork, self).__init__()
        self.dynamic_system = dynamic_system
        self.data_module = data_module
        self.learn_shape_epochs = learn_shape_epochs
        self.primal_learning_rate = primal_learning_rate
        self.dt = self.dynamic_system.dt
        self.gamma = gamma
        self.require_grad_descent_loss = require_grad_descent_loss
       
        
        self.g = nn.Sequential(
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
        
        self.K = self.dynamic_system.K
        self.clf_lambda = clf_lambda
        self.clf_relaxation_penalty = clf_relaxation_penalty
        self.training_stage = 0

        # self.generate_cvx_solver()


    
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
        gs = self.g(s)

        return gs


    def G_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        for layer in self.g:
            V = layer(V)

            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
            elif isinstance(layer, nn.Tanh):
                JV = torch.matmul(torch.diag_embed(1 - V ** 2), JV)
            elif isinstance(layer, nn.ReLU):
                JV = torch.matmul(torch.diag_embed(torch.sign(V)), JV)

        return V, JV

    
    def value_function_loss(self,
        s : torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
    ):
        loss = []
        gs = self.g(s)

        gs_hat, _ = torch.min(self.dynamic_system.state_constraints(s), dim=1, keepdim=True)
        # print(gs_hat)
        loss_term = F.relu( torch.norm( gs - gs_hat, p=2) - 0.3 )
        loss.append(("value_function_loss_term", loss_term))
        
        return loss

    def G_lie_derivatives(
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
        _, gradh = self.G_with_jacobian(x)
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

    
    def on_train_start(self) -> None:
        print(f"############### Training start #########################")

    def on_train_epoch_start(self) -> None:
        print("################## the epoch start #####################")
        self.epoch_start_time = time.time()
        
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        s, safe_mask, unsafe_mask = batch
    
        # Compute the losses
        component_losses = {}

        component_losses.update( 
                self.value_function_loss(s, safe_mask, unsafe_mask)
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

        safety_losses = 0
        performance_losses = 0
        value_function_loss = 0
        value_function_loss_name = ['value_function_loss_term']
        safety_loss_name = ['unsafe_region_term', 'descent_term_linearized', 
                            'descent_term_simulated','descent_term_epsilon_area','unsafe_region_epsilon_term']
        performance_lose_name = ['safe_region_term']
        for key in avg_losses.keys():
            if key in safety_loss_name:
                safety_losses += avg_losses[key]
            if key in performance_lose_name:
                performance_losses += avg_losses[key]
            if key in value_function_loss_name:
                value_function_loss += avg_losses[key]


        # Log the overall loss...
        # if self.current_epoch > self.learn_shape_epochs:
        self.log("Total_loss/train", avg_losses["loss"])
        print(f"\n the overall loss of this training epoch {avg_losses['loss']}\n")
        self.log("Epoch_time/train", self.epoch_end_time - self.epoch_start_time)
        print(f"the epoch time consume is {self.epoch_end_time - self.epoch_start_time}")
        self.log("Value_function_loss/train", value_function_loss)
        print(f"overall value function loss of this training epoch {value_function_loss}")
        
        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "loss":
                continue
            # Log the other losses
            self.log(loss_key + "/train", avg_losses[loss_key], sync_dist=True)

        

    def validation_step(self, batch, batch_idx):
        """Conduct the validation step for the given batch"""
        # Extract the input and masks from the batch
        x, safe_mask, unsafe_mask = batch

        # Get the various losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, safe_mask, unsafe_mask, accuracy=False)
        )
        if self.current_epoch > self.learn_shape_epochs:
            component_losses.update(
                self.descent_loss(x, safe_mask, unsafe_mask, accuracy=False)
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
            self.log("Total_loss/val", avg_losses["val_loss"])
            print(f"\n the overall loss of this validation epoch {avg_losses['val_loss']}\n")
            #print(f"\n the descent accuracy of this validation epoch {avg_losses['CLBF descent accuracy (linearized)']}\n")
            
            # And all component losses
            for loss_key in avg_losses.keys():
                # We already logged overall loss, so skip that here
                if loss_key == "val_loss":
                    continue
                # Log the other losses
                self.log(loss_key + "/val", avg_losses[loss_key])
        
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
        batch_dict = {"shape_h": {}, "shape_g": {},"safe_violation": {}, "unsafe_violation": {}, "descent_violation": {}, "safe_boundary":{} }
        
        # record shape_h
        h_x = self.h(x)
        batch_dict["shape_h"]["state"] = x
        batch_dict["shape_h"]["val"] = h_x

        # record shape_g
        g_x = self.g(x)
        batch_dict["shape_g"]["state"] = x
        batch_dict["shape_g"]["val"] = g_x

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

        batch_dict["unsafe_violation"]["state"] = s_unsafe_violation
        batch_dict["unsafe_violation"]["val"] = h_s_unsafe_violation

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

        # get safety boundary
        state_constraints = self.dynamic_system.nominal_state_constraints(x)
        state_constraints_norm = torch.norm(state_constraints, dim=1)
        safe_boundary_index = state_constraints_norm < 0.02
        safe_boundary_state = x[safe_boundary_index]

        batch_dict["safe_boundary"]["state"] = safe_boundary_state

        return batch_dict

    def test_epoch_end(self, outputs):
        
        print("test_epoch_end")
        torch.save(outputs, "test_results.pt")
        
        print("results is save to test_result.pt")
        


    def configure_optimizers(self):
        clbf_params =  list(self.g.parameters())
        clbf_opt = torch.optim.SGD(
            clbf_params,
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        self.opt_idx_dict = {0: "clbf"}

        return [clbf_opt]



if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    s0 = torch.rand(1, 2, dtype=torch.float).to(device) * 10 - 5

    data_module = DataModule(system=inverted_pendulum_1, train_grid_gap=0.3, test_grid_gap=0.3)





