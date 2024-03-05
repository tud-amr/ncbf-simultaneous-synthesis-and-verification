from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Dataset.VerificationDataModule import VerificationDataModule
from safe_rl_cbf.Dataset.DataModule import DataModule

class Verifier:
    def __init__(self, model, initial_grid_gap, minimum_grip_gap = 0.005, verify_batch_size=8, prefix="", log_dir="logs"):
        self.model = model
        self.initial_grid_gap = initial_grid_gap
        self.minimum_grid_gap = minimum_grip_gap
        self.verify_batch_size = verify_batch_size
        self.prefix = prefix
        self.log_dir = log_dir
        self.verify_data_module = VerificationDataModule(system=self.model.dynamic_system, 
                                                         initial_grid_gap=self.initial_grid_gap, 
                                                         verify_batch_size=verify_batch_size, prefix=self.prefix+"verify", log_dir=self.log_dir)    
        self.temporary_data_module = DataModule(system=self.model.dynamic_system, batch_size=verify_batch_size, prefix=self.prefix + "temporary", log_dir=self.log_dir)
        self.augment_data_module = DataModule(system=self.model.dynamic_system, batch_size=verify_batch_size, prefix=self.prefix + "augment", log_dir=self.log_dir)

     
    def reset(self):
        self.verify_data_module.reset()
        self.temporary_data_module.reset()
        self.augment_data_module.reset()

        
    def prepare_data(self, initial_grid_gap = None, verify_batch_size = None):
            
        assert initial_grid_gap is not None or self.initial_grid_gap is not None, "Please specify the initial grid gap"
        initial_grid_gap = initial_grid_gap if initial_grid_gap is not None else self.initial_grid_gap

        assert verify_batch_size is not None or self.verify_batch_size is not None, "Please specify the batch size"
        verify_batch_size = verify_batch_size if verify_batch_size is not None else self.verify_batch_size

        self.verify_data_module.set_initial_grid_gap(initial_grid_gap)
        self.verify_data_module.set_batch_size(verify_batch_size)
        self.verify_data_module.reset()

    def verify(self):
        
        print_info("#################### Start verifying the neural network #####################")
        verified_flag = False
        reach_minimum_gap = False 

        while verified_flag == False and reach_minimum_gap == False:
            # if there is a hyperrectangle that is not verified, then set verified_flag to False
            # if the grid gap of a hyperrectangle is larger than the minimum grid gap, then set reach_minimum_gap to False
            verified_flag = True
            reach_minimum_gap = True   

            data_loader = self.verify_data_module.dataloader()
            for batch in data_loader:
                # get batch data
                s, s_gridding_gap, safe_mask, unsafe_mask, satisfied = batch
                # feed to GPU
                s = s.to(self.model.device)
                s_gridding_gap = s_gridding_gap.to(self.model.device)
                safe_mask = safe_mask.to(self.model.device)
                unsafe_mask = unsafe_mask.to(self.model.device)
                satisfied = satisfied.to(self.model.device)

                # verify these hyperrectangles
                unsatisfied_indices = self.model.verify(s, s_gridding_gap, safe_mask, unsafe_mask, satisfied)
                if unsatisfied_indices.shape[0] != 0:
                    verified_flag = False
                   
                # split the hyperrectangles that are not verified
                for i in unsatisfied_indices:
                    
                    leave_node_s = s[i].reshape(1, -1)
                    leave_node_grid_gap = s_gridding_gap[i].reshape(1, -1)
                    
                    if torch.max(leave_node_grid_gap) > self.minimum_grid_gap:
                        # split hyperrectangle
                        reach_minimum_gap = False
                        s_dim = leave_node_s.shape[1]
                        dir = torch.tensor([0.5, -0.5])
                        combine = list(product(dir, repeat=s_dim))

                        for i in range(len(combine)):
                            coefficent = torch.tensor(combine[i]).reshape(1, -1)
                            s_temp = leave_node_s + leave_node_grid_gap/2 * coefficent
                            s_gridding_gap_temp = leave_node_grid_gap / 2
                            safe_mask_temp = self.model.dynamic_system.safe_mask(s_temp).reshape(-1,1)
                            unsafe_mask_temp = self.model.dynamic_system.unsafe_mask(s_temp).reshape(-1,1)
                            satisfied_temp = torch.tensor( [[False]] ).reshape(-1,1)
                            self.temporary_data_module.add_batch_data(s_temp, s_gridding_gap_temp, safe_mask_temp, unsafe_mask_temp, satisfied_temp)
                            
                    else:
                        # keep this hyperrectangle
                        self.temporary_data_module.add_batch_data(leave_node_s, leave_node_grid_gap, safe_mask[i].reshape(-1,1), unsafe_mask[i].reshape(-1,1), satisfied[i].reshape(-1,1))
            
            self.temporary_data_module.push_to_database()
            
            if verified_flag == False:
                print_warning(f"found {len(self.temporary_data_module)} unsatisfied hyperrectangles, will split them")

            if reach_minimum_gap == True:
                print_warning(f"reach minimum grid gap, stop branching")
            
            # delete the previous batch points
            self.verify_data_module.clean()
            
            # insert the new batch points
            self.verify_data_module.clone(self.temporary_data_module)
            self.temporary_data_module.clean()

        self.augment_data_module.concatenate(self.verify_data_module)

        return verified_flag
    
        