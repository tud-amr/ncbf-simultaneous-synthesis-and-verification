import numpy as np
import itertools
from typing import Tuple, List, Optional
import json

import time
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt

from DataModule import DataModule
from system import Car

######################### define neural network #########################


class NeuralNetwork(pl.LightningModule):
    def __init__(
        self,
        dynamic_system: Car,
        data_module: DataModule,
        learn_shape_epochs: int = 10,
        primal_learning_rate: float = 1e-3,
        gamma: float = 0.9
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
        self.V = nn.Sequential(
            nn.Linear(2, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
        u_lower, u_upper = self.dynamic_system.control_limits
        self.u_v =[u_lower.item(), u_upper.item()]
        self.train_loss = []
        self.val_loss = []

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
        Vs = self.V(s)
        # print(Vs.shape)
        # print(h_bar_s.shape)
        # print(beta_s.shape)
        
        # print(torch.norm(Vs, dim=1).reshape(-1,1))
        # print(torch.norm(beta_s, dim=1).reshape(-1,1))

        return hs, Vs

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
        eps = 0.1
        loss = []

        s_feasible = s[torch.logical_not(unsafe_mask)]

        descent_loss, self.u_star_index = self.gradient_descent_violation(s_feasible, self.u_v)
        descent_loss =  F.relu(eps - descent_loss)
        descent_loss_mean = descent_loss.mean()
        loss.append(("descent loss", descent_loss_mean))
        if accuracy:
            descent_loss_acc = (descent_loss > eps).sum()/(descent_loss.nelement())
            loss.append(("CLBF descent accuracy (linearized)", descent_loss_acc))
        
        return loss

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

    def test_loss(self, 
        s: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor
        ):
        hs = self.h(s)
        loss = []
        test_loss = torch.pow(hs - 1, 2)
        test_loss_mean = torch.mean(test_loss)
        loss.append(("test loss", test_loss_mean))
        
        return loss

    def Dt(self, s, u_vi):
        Lf_h, Lg_h = self.V_lie_derivatives(s)

        return Lf_h + Lg_h * u_vi

    def gradient_descent_condition(self, s, u_vi, alpha=0.5):
        output = self.h(s)
        dt = self.Dt(s, u_vi)

        result = dt + alpha * output
        
        return result

    def gradient_descent_violation(self, s, u_v):
        
        c_list = [self.gradient_descent_condition(s, u) for u in u_v]
        c_list = torch.hstack(c_list)
        
        violation_max, u_max_index = torch.max(c_list, dim=1) 

        return violation_max, u_max_index


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

    
    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, safe_mask, unsafe_mask = batch

        # Compute the losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, safe_mask, unsafe_mask, accuracy=True)
        )

        if self.current_epoch > self.learn_shape_epochs:
            component_losses.update(
                self.descent_loss(
                    x, safe_mask, unsafe_mask, accuracy=True,requires_grad=True
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
        descent_violation, u_star_index = self.gradient_descent_violation(x, self.u_v)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    car = Car()
    data_module = DataModule()
    h = NeuralNetwork(dynamic_system=car, data_module=data_module).to(device)
    s0 = torch.tensor([-1.8545, 0.2772, 2.34, -4.23], dtype=torch.float).reshape((2,2)).to(device)
    s0.requires_grad_(True)
    output = h(s0)
    print(output)
    _, gradh = h.V_with_jacobian(s0)
    print(gradh)
    print(gradh.shape)


# def myCustomLoss(h_bar_s, C_bar_s, epsilon = 0.01):
    
#     return (torch.sgn(h_bar_s) + 1)/2 * torch.relu(-C_bar_s+epsilon)





