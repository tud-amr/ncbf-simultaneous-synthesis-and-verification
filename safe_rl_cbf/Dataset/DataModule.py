from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Dataset.SqlDataSet import SqlDataSet
from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        system: ControlAffineSystem,
        batch_size: int = 64,
        prefix: str = "",
        log_dir: str = "logs",
    ):
        super().__init__()
        self.system = system
        self.prefix = prefix
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.sql_database = SqlDataSet(self.system.ns, self.prefix, self.log_dir)
        self.sql_bandwith = 1e5

    def __len__(self):
        return len(self.sql_database)

    def delete(self):
        print(f"{self.sql_database.db_path} is deleted")
        del self.sql_database
    
    def reset(self):
        self.clean()
        self.prepare_data()

    def clean(self):
        self.sql_database.clean()

    def save_as_tensor(self, file_name="s.pt"):
        s, _, _, _, _  = self.sql_database.to_tensor()

        torch.save(s, os.path.join(self.log_dir, file_name))

    def clone(self, data_module):
        self.clean()
        self.sql_database.clone(data_module.sql_database)

    def concatenate(self, data_module):
        self.sql_database.concatenate(data_module.sql_database)

    def prepare_data(self):
        pass

    
    def add_random_data(self, total_num):
        domain_lower_bd, domain_upper_bd = self.system.domain_limits
        domain_bd_gap = domain_upper_bd - domain_lower_bd

        while total_num > 0:
            num = total_num if total_num < self.sql_bandwith else self.sql_bandwith
            total_num = total_num - num

            s = torch.rand(int(num), self.system.ns) * domain_bd_gap + domain_lower_bd
            s_gridding_gap = torch.zeros(s.shape[0], self.system.ns)
            safe_mask_training = self.system.safe_mask(s).reshape(-1,1)
            unsafe_mask_training = self.system.unsafe_mask(s).reshape(-1,1)
            satisfied = torch.tensor( [[False]] * s.shape[0] ).reshape(-1,1)

            self.sql_database.insert_p_batch(s, s_gridding_gap, safe_mask_training, unsafe_mask_training, satisfied)

    def add_one_data(self, s, s_gridding_gap, safe_mask_training, unsafe_mask_training, satisfied):
        """
        Insert one point to the database
            s: torch.Tensor, shape (ns, )
            grid_gap: torch.Tensor, shape (ns, )
            nominal_safe_mask: bool
            unsafe_mask: bool
            satisfied: bool
        """
        self.sql_database.insert_p(s, s_gridding_gap, safe_mask_training, unsafe_mask_training, satisfied)

    def add_batch_data(self, s, s_gridding_gap, safe_mask_training, unsafe_mask_training, satisfied):
        """
        Insert batch of points to the database
            s: torch.Tensor, shape (n, ns)
            grid_gap: torch.Tensor, shape (n, ns)
            nominal_safe_mask: torch.Tensor, shape (n, 1)
            unsafe_mask: torch.Tensor, shape (n, 1)
            satisfied: torch.Tensor, shape (n, 1)
        """
        self.sql_database.insert_p_batch(s, s_gridding_gap, safe_mask_training, unsafe_mask_training, satisfied)

    def setup(self, stage=None):
        """Setup -- nothing to do here"""
        pass

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def dataloader(self):
        """Make the DataLoader for training data"""
        return DataLoader(
            self.sql_database,
            batch_size=self.batch_size,
            num_workers=2,
        )


        