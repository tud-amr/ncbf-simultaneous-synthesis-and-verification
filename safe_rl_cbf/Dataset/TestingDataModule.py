from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Dataset.DataModule import DataModule   
from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem

class TestingDataModule(DataModule):
    def __init__(
        self,
        system: ControlAffineSystem,
        test_index: dict,
        test_batch_size: int = 64,
        testing_points_num: int = 1e5,
        prefix: str = "",
        log_dir: str = "logs",
    ):
        super().__init__(system, test_batch_size, prefix, log_dir)
        self.test_index = test_index
        self.testing_points_num = testing_points_num
       
    def prepare_data(self):
        """Prepare the data 
        test_points_num: int, the total number of testing points
        """

        print("Preparing data........")
        
        domain_lower_bd, domain_upper_bd = self.system.domain_limits
        domain_bd_gap = domain_upper_bd - domain_lower_bd

        # generate testing data
        s_test_grid_list = []
        num_each_dim = int(self.testing_points_num ** (1/2))

        for key, value in self.test_index.items():
            key = int(key)
            value = None if value == "None" else value
            
            if value is None:
                s_i_grid_test = torch.linspace( domain_lower_bd[key], domain_upper_bd[key], num_each_dim) 
            else:
                s_i_grid_test = torch.ones(1) * value
            s_test_grid_list.append(s_i_grid_test.float())
        
        mesh_grids_test = torch.meshgrid(s_test_grid_list)
        unsqueez_mesh_grid_test = [ torch.unsqueeze(mesh_grid, dim=0) for mesh_grid in mesh_grids_test ]
        mesh_grids_test = torch.vstack(unsqueez_mesh_grid_test)
        data_points_test = torch.flatten(mesh_grids_test, start_dim=1)
        test_sample = data_points_test.T
        self.s_testing = test_sample
        grid_gap = torch.ones(self.s_testing.shape[0], self.system.ns)
        satisfy_constraint = torch.ones(self.s_testing.shape[0], 1)

        self.safe_mask_testing =  self.system.safe_mask(self.s_testing).unsqueeze(dim=-1) # self.s_testing.norm(dim=-1) <= 0.6
        self.unsafe_mask_testing = self.system.unsafe_mask(self.s_testing).unsqueeze(dim=-1)
        
        self.sql_database.insert_p_batch(self.s_testing, grid_gap, self.safe_mask_testing, self.unsafe_mask_testing, satisfy_constraint)

        print("Full dataset:")
        print("\t----------------------")
        print(f"\t{len(self.sql_database)} testing")
        
    def set_testing_points_num(self, testing_points_num):
        self.testing_points_num = testing_points_num
    
    def set_test_index(self, test_index):
        self.test_index = test_index
        
if __name__ == "__main__":
    from safe_rl_cbf.Dynamics.dynamic_system_instances import point_robot

    data_module = TestingDataModule(system=point_robot, test_index={0: None, 1: None, 2: 0, 3: 0}, prefix="test", test_batch_size=64,)

    data_module.prepare_data(10)

    data_module.delete()