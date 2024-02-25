import torch
from safe_rl_cbf.Dataset.DataModule import DataModule
from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem

from safe_rl_cbf.Models.NeuralCBF import NeuralCBF
from typing import List, Tuple
import time
import itertools

class NeuralRA(NeuralCBF):
    def __init__(self, dynamic_system: ControlAffineSystem, data_module: DataModule, primal_learning_rate: float = 0.001, gamma: float = 0.9, clf_lambda: float = 0.5):
        super().__init__(dynamic_system, data_module, 0, primal_learning_rate, gamma, clf_lambda)


    def boundary_loss(
        self,
        s : torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        hs : torch.Tensor,
        gradh : torch.Tensor,
        coefficient_for_performance_state_loss: float=1e2,
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

        baseline = torch.min(self.dynamic_system.state_constraints(s), dim=1, keepdim=True).values.detach() - 0.1

        safe_mask = torch.logical_not(unsafe_mask)

        if self.dynamic_system.nu != 0:
            # find the action that maximize the hamiltonian
            hs_next_list = []
            for u in self.u_v:
                with torch.no_grad():
                    x_next = self.dynamic_system.step(s, u=torch.ones(s.shape[0], self.dynamic_system.nu).to(s.device)*u.to(s.device))
                    hs_next = self.h(x_next)
                    hs_next_list.append(hs_next)

            hs_next = torch.stack(hs_next_list, dim=1)
            _ , index_control = torch.max(hs_next, dim=1, keepdim=True)

            index_control = index_control.squeeze()
            u_v = torch.cat(self.u_v, dim=0)
            u_max = u_v[index_control]
            u_max = u_max.to(s.device)
        else:
            u_max = None
        
        if self.dynamic_system.nd != 0:
            # find the disturbance that minimize the hamiltonian
            hs_next_list = []
            for d in self.d_v:
                with torch.no_grad():
                    x_next = self.dynamic_system.step(s, d=torch.ones(s.shape[0], self.dynamic_system.nd).to(s.device)*d.to(s.device))
                    hs_next = self.h(x_next)
                    hs_next_list.append(hs_next)

            hs_next = torch.stack(hs_next_list, dim=1)
            _, index_control = torch.min(hs_next, dim=1, keepdim=True)

            index_control = index_control.squeeze()
            d_v = torch.cat(self.d_v, dim=0)
            d_min = d_v[index_control]
            d_min = d_min.to(s.device)
        else:
            d_min = None
        

        # compute the hamiltonian
        with torch.no_grad():
            x_next = self.dynamic_system.step(s, u=u_max, d=d_min)
            hs_next = self.h(x_next)


        gamma = 0.99
        hs_bar = (1- gamma) * baseline + gamma * torch.min(baseline, hs_next)

        safe_violation = coefficient_for_performance_state_loss * torch.abs(  hs_bar - hs )
        safe_hs_term =  safe_violation.mean()
        if not torch.isnan(safe_hs_term):
            loss.append(("safe_region_term", safe_hs_term))
            

        return loss
    

    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        s, safe_mask, unsafe_mask, grid_gap = batch
        s.requires_grad_(True)
        
        # s = torch.Tensor([1.23, 0.1]).to(s.device).reshape(-1,2)

        # Compute the losses
        component_losses = {}
       
        s_random = s # + torch.mul(grid_gap , torch.rand(s.shape[0], self.dynamic_system.ns, requires_grad=True).to(s.device)) - grid_gap/2
        
        hs = self.h(s_random)
        gradh = self.jacobian(hs, s_random)

       
        if self.train_mode == 0:
         
            component_losses.update(
                self.boundary_loss(s_random, safe_mask, unsafe_mask, hs, gradh, 
                coefficient_for_performance_state_loss=100)
            )
            

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(s)
        safety_loss = torch.tensor(0.0).type_as(s)
        performance_loss = torch.tensor(0.0).type_as(s)
        descent_loss = torch.tensor(0.0).type_as(s)

        # For the objectives, we can just sum them
        for key, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value
            if key in self.performance_lose_name:
                performance_loss += loss_value
            if key in self.safety_loss_name:
                safety_loss += loss_value
            if key in self.descent_loss_name:
                descent_loss += loss_value
        

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

           
        if self.train_mode == 0 and (performance_losses < 1 and safety_losses < 1): # warm up
            self.trainer.should_stop = True
            # pass
        
        print(f"current learning rate is {self.trainer.optimizers[0].param_groups[0]['lr']}")
