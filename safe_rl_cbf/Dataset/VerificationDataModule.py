from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Dataset.DataModule import DataModule   
from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem

class VerificationDataModule(DataModule):
    def __init__(
        self,
        system: ControlAffineSystem,
        initial_grid_gap: List,
        verify_batch_size: int = 64,
        prefix: str = "",
        log_dir: str = "logs",
    ):
        super().__init__(system, verify_batch_size, prefix, log_dir)
        self.initial_grid_gap = torch.tensor(initial_grid_gap)
       
    def prepare_data(self):
        """Prepare the data """
        self.sql_database.clean()
        
        print("Preparing data........")
        
        domain_lower_bd, domain_upper_bd = self.system.domain_limits
        domain_bd_gap = domain_upper_bd - domain_lower_bd

       
        s_train_grid_list = []
        sample_data = []
        sample_data_grid_gap = []

        for i in range(self.system.ns):
            s_i_grid_train = torch.arange(domain_lower_bd[i], domain_upper_bd[i], self.initial_grid_gap[i])
            s_train_grid_list.append(s_i_grid_train.float())
            
        mesh_grids_train = torch.meshgrid(s_train_grid_list)
        unsqueez_mesh_grid_train = [ torch.unsqueeze(mesh_grid, dim=0) for mesh_grid in mesh_grids_train ]
        mesh_grids_train = torch.vstack(unsqueez_mesh_grid_train)
        data_points_train = torch.flatten(mesh_grids_train, start_dim=1)
        
        s = data_points_train.T
        s_gridding_gap = torch.ones(s.shape[0], self.system.ns) * self.initial_grid_gap.reshape(1,-1)
        safe_mask_training = self.system.safe_mask(s).reshape(-1,1)
        unsafe_mask_training = self.system.unsafe_mask(s).reshape(-1,1)
        satisfied = torch.tensor( [[False]] * s.shape[0] ).reshape(-1,1)
        self.sql_database.insert_p_batch(s, s_gridding_gap, safe_mask_training, unsafe_mask_training, satisfied)

    def set_initial_grid_gap(self, initial_grid_gap):
        self.initial_grid_gap = torch.tensor(initial_grid_gap)

        
if __name__ == "__main__":
    from safe_rl_cbf.Dynamics.dynamic_system_instances import inverted_pendulum_1

    verify_data_module = VerificationDataModule(system=inverted_pendulum_1, initial_grid_gap=[1, 1], prefix="verify", verify_batch_size=8)

    verify_data_module.prepare_data()

    # verify_data_module.delete()