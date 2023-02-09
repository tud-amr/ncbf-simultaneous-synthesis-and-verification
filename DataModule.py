from typing import List, Callable, Tuple, Dict, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from CARs import car1
from system import Car

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        system: Car,
        training_sample_num: int = 10000,
        val_split: float = 0.1,
        train_batch_size: int = 64,
        test_batch_size: int = 128,
        train_grid_gap: float = 0.01,
        test_grid_gap: float = 0.1,
    ):
        super().__init__()
        self.system = system
        self.training_sample_num = training_sample_num
        self.val_split = val_split
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_grid_gap = train_grid_gap
        self.test_grid_gap = test_grid_gap

    def prepare_data(self):
        domain_lower_bd, domain_upper_bd = self.system.domain_limits
        domain_bd_gap = domain_upper_bd - domain_lower_bd

        s_train_grid_list = []
        s_test_grid_list = []
        for i in range(self.system.ns):
            s_i_grid_train = torch.arange(domain_lower_bd[i], domain_upper_bd[i], self.train_grid_gap)
            s_i_grid_test = torch.arange(domain_lower_bd[i], domain_upper_bd[i], self.test_grid_gap)
            s_train_grid_list.append(s_i_grid_train.float())
            s_test_grid_list.append(s_i_grid_test.float())
        
        mesh_grids_train = torch.meshgrid(s_train_grid_list)
        mesh_grids_test = torch.meshgrid(s_test_grid_list)
        unsqueez_mesh_grid_train = [ torch.unsqueeze(mesh_grid, dim=0) for mesh_grid in mesh_grids_train ]
        unsqueez_mesh_grid_test = [ torch.unsqueeze(mesh_grid, dim=0) for mesh_grid in mesh_grids_test ]
        mesh_grids_train = torch.vstack(unsqueez_mesh_grid_train)
        mesh_grids_test = torch.vstack(unsqueez_mesh_grid_test)
        data_points_train = torch.flatten(mesh_grids_train, start_dim=1)
        data_points_test = torch.flatten(mesh_grids_test, start_dim=1)
        
        s_samples = data_points_train.T
        test_sample = data_points_test.T

        random_indices = torch.randperm(s_samples.shape[0])
        val_pts = int(s_samples.shape[0] * self.val_split)
        validation_indices = random_indices[:val_pts]
        training_indices = random_indices[val_pts:]
        
        self.s_training = s_samples[training_indices]
        self.s_validation = s_samples[validation_indices]
        self.s_testing = test_sample
        
        self.safe_mask_training = self.s_training.norm(dim=-1) <= 0.6
        self.unsafe_mask_training = self.s_training.norm(dim=-1) >= 1.2#self.system.unsafe_mask(self.s_training)
        
        self.safe_mask_validation = self.s_validation.norm(dim=-1) <= 0.6
        self.unsafe_mask_validation = self.s_validation.norm(dim=-1) >= 1.2 #self.system.unsafe_mask(self.s_validation)

        self.safe_mask_testing = self.s_testing.norm(dim=-1) <= 0.6
        self.unsafe_mask_testing = self.s_testing.norm(dim=-1) >= 1.2 #self.system.unsafe_mask(self.s_training)

        print("Full dataset:")
        print(f"\t{self.s_training.shape[0]} training")
        print(f"\t{self.s_validation.shape[0]} validation")
        print("\t----------------------")

        print(f"\t{self.safe_mask_training.sum()} safe points")
        print(f"\t({self.safe_mask_validation.sum()} val)")
        print(f"\t{self.unsafe_mask_training.sum()} unsafe points")
        print(f"\t({self.unsafe_mask_validation.sum()} val)")
        print("\t----------------------")
        print(f"\t{self.s_testing.shape[0]} testing")

        # Turn these into tensor datasets
        self.training_data = TensorDataset(
            self.s_training,
            self.safe_mask_training,
            self.unsafe_mask_training
        )
        self.validation_data = TensorDataset(
            self.s_validation,
            self.safe_mask_validation,
            self.unsafe_mask_validation
            )
        self.testing_data = TensorDataset(
            self.s_testing,
            self.safe_mask_testing,
            self.unsafe_mask_testing
            )
    
    def setup(self, stage=None):
        """Setup -- nothing to do here"""
        pass


    def train_dataloader(self):
        """Make the DataLoader for training data"""
        return DataLoader(
            self.training_data,
            batch_size=self.train_batch_size,
            num_workers=8,
        )


    def val_dataloader(self):
        """Make the DataLoader for validation data"""
        return DataLoader(
            self.validation_data,
            batch_size=self.train_batch_size,
            num_workers=8,
        )
    
    def test_dataloader(self):
        """Make the DataLoader for validation data"""
        return DataLoader(
            self.testing_data,
            batch_size=self.test_batch_size,
            num_workers=8,
        )

        
if __name__ == "__main__":
    
    data_module = DataModule(system=car1, train_grid_gap=0.01, test_grid_gap=0.1)

    data_module.prepare_data()