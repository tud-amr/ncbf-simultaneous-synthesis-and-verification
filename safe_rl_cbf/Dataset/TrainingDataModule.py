from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Dataset.DataModule import DataModule   
from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem

class TrainingDataModule(DataModule):
    def __init__(
        self,
        system: ControlAffineSystem,
        train_batch_size: int = 64,
        training_points_num: int = 1e5,
        prefix: str = "",
        log_dir: str = "logs",
    ):
        super().__init__(system, train_batch_size, prefix, log_dir)
        self.training_points_num = training_points_num
        
    def prepare_data(self):
        """Prepare the data 
        training_points_num: int, the total number of training points
        """

        print("Preparing data........")
        ns = self.system.ns

        self.insert_random_data( self.training_points_num )
        print("Full dataset:")
        print("\t----------------------")
        print(f"\t{len(self.sql_database)} training")
        
    def set_training_points_num(self, training_points_num):
        self.training_points_num = training_points_num
    
   
        
if __name__ == "__main__":
    from safe_rl_cbf.Dynamics.dynamic_system_instances import point_robot

    data_module = TrainingDataModule(system=point_robot, prefix="test", train_batch_size=64, training_points_num=1000)

    data_module.prepare_data(100)

    data_module.delete()