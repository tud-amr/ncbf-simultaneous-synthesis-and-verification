from typing import List, Callable, Tuple, Dict, Optional

import torch
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader
from safe_rl_cbf.Dynamics.dynamic_system_instances import car1, inverted_pendulum_1, point_robot
from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem
from itertools import product
from treelib import Tree, Node
import matplotlib.pyplot as plt
# from dreal import *
import timeit 
# from safe_rl_cbf.NeuralLC.Functions import *

class TrainingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        system: ControlAffineSystem,
        val_split: float = 0.1,
        train_batch_size: int = 64,
        training_points_num: int = 100000,
        training_grid_gap: torch.tensor = None,
        train_mode: int = 0,
       
    ):
        super().__init__()
        self.system = system
        self.val_split = val_split
        self.train_batch_size = train_batch_size
        self.training_points_num = training_points_num
        self.minimum_training_points_num = 1e5
        self.train_mode = train_mode
        self.training_grid_gap = training_grid_gap
        self.minimum_grid_gap = 0.05
        self.verified = -1
        self.augment_data = torch.zeros(1, self.system.ns)
        self.maximum_augment_data_num = int(5e6)
        # self.initalize_data()
        self.model = None
        self.SMT_verification_time = 0
        self.SMT_CE_num = 0
        self.new_tree = None

    def initalize_data(self):
        domain_lower_bd, domain_upper_bd = self.system.domain_limits
        domain_bd_gap = domain_upper_bd - domain_lower_bd
        
        s = torch.rand(self.training_points_num, self.system.ns) * domain_bd_gap + domain_lower_bd

        # generate training data
        s_samples = s
       

        # split into training and validation
        random_indices = torch.randperm(s_samples.shape[0])
        val_pts = int(s_samples.shape[0] * self.val_split)
        validation_indices = random_indices[:val_pts]
        training_indices = random_indices[val_pts:]
        
        # store the data
        self.s_training = s_samples[training_indices]
        
        
        self.s_validation = s_samples[validation_indices]
        
        
        # generate other information
        self.safe_mask_training = self.system.safe_mask(self.s_training)  # self.s_training.norm(dim=-1) <= 0.6
        self.unsafe_mask_training = self.system.unsafe_mask(self.s_training)

        self.safe_mask_validation =  self.system.safe_mask(self.s_validation) # self.s_validation.norm(dim=-1) <= 0.6
        self.unsafe_mask_validation = self.system.unsafe_mask(self.s_validation)
        
        # generate testing data
        s_test_grid_list = []
       
        for i in range(self.system.ns):
            s_i_grid_test = torch.arange(domain_lower_bd[i], domain_upper_bd[i], self.test_grid_gap)
            s_test_grid_list.append(s_i_grid_test.float())
        
        mesh_grids_test = torch.meshgrid(s_test_grid_list)
        unsqueez_mesh_grid_test = [ torch.unsqueeze(mesh_grid, dim=0) for mesh_grid in mesh_grids_test ]
        mesh_grids_test = torch.vstack(unsqueez_mesh_grid_test)
        data_points_test = torch.flatten(mesh_grids_test, start_dim=1)
        test_sample = data_points_test.T
        self.s_testing = test_sample

        self.safe_mask_testing =  self.system.safe_mask(self.s_testing) # self.s_testing.norm(dim=-1) <= 0.6
        self.unsafe_mask_testing = self.system.unsafe_mask(self.s_testing)
        
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

        # # Turn these into tensor datasets
        # self.training_data = TensorDataset(
        #     self.s_training,
        #     self.safe_mask_training,
        #     self.unsafe_mask_training,
        #     self.grid_gap_training
        # )
        # self.validation_data = TensorDataset(
        #     self.s_validation,
        #     self.safe_mask_validation,
        #     self.unsafe_mask_validation,
        #     self.grid_gap_validation
        #     )
        # self.testing_data = TensorDataset(
        #     self.s_testing,
        #     self.safe_mask_testing,
        #     self.unsafe_mask_testing,
        #     self.grid_gap_testing
        #     )
    

    def expand_leave(self, leave_node):
    
        leave_node_id = leave_node.identifier
        leave_node_s = leave_node.data[0]
        leave_node_grid_gap = leave_node.data[1]
        not_reach_minimum_gap = False
        if torch.max(leave_node_grid_gap) > self.minimum_grid_gap:
            not_reach_minimum_gap = True

            s_dim = leave_node_s.shape[1]
            dir = torch.tensor([0.5, -0.5])
            combine = list(product(dir, repeat=s_dim))

            for i in range(len(combine)):
                coefficent = torch.tensor(combine[i]).reshape(1, -1)
                new_s = leave_node_s + leave_node_grid_gap/2 * coefficent
                new_grid_gap = leave_node_grid_gap / 2
                satisfy_constraint = False
                new_data = [new_s, new_grid_gap, satisfy_constraint]
                self.new_tree.create_node(f"{self.new_tree.size()}", identifier=self.uniname_of_data(new_data), data=new_data, parent=leave_node_id)

        return not_reach_minimum_gap


    def get_minimum_grid_gap(self):
        grid_gaps = []
        for leave_node in self.new_tree.leaves():
            grid_gaps.append( torch.min(leave_node.data[1]))
            
        return min(grid_gaps)


    def uniname_of_data(self, data):
        s = data[0]
        id = str('')
        for i in range(s.shape[1]):
            id = id + str(s[0,i].item())

        return id
        
    def prepare_data(self):
        """Prepare the data """

        print("Preparing data........")
        
        if self.train_mode == 0:
            domain_lower_bd, domain_upper_bd = self.system.domain_limits
            domain_bd_gap = domain_upper_bd - domain_lower_bd

            s = torch.rand(self.training_points_num, self.system.ns) * domain_bd_gap + domain_lower_bd
           
            # generate training data
            s_samples = s
            
            assert self.training_grid_gap is None
            s_gridding_gap = torch.zeros(s_samples.shape[0], self.system.ns)


            # split into training and validation
            random_indices = torch.randperm(s_samples.shape[0])
            val_pts = int(s_samples.shape[0] * self.val_split)
            validation_indices = random_indices[:val_pts]
            training_indices = random_indices[val_pts:]
            


            # store the training data
            self.s_training = s_samples[training_indices]
            self.s_grid_gap_training = s_gridding_gap[training_indices]
            self.s_validation = s_samples[validation_indices]
            self.s_grid_gap_validation = s_gridding_gap[validation_indices]
            
            # generate other information
            self.safe_mask_training = self.system.safe_mask(self.s_training)  # self.s_training.norm(dim=-1) <= 0.6
            self.unsafe_mask_training = self.system.unsafe_mask(self.s_training)
            
            self.safe_mask_validation =  self.system.safe_mask(self.s_validation) # self.s_validation.norm(dim=-1) <= 0.6
            self.unsafe_mask_validation = self.system.unsafe_mask(self.s_validation)
            
        elif (self.train_mode == 1 or self.train_mode == 3):
            
            domain_lower_bd, domain_upper_bd = self.system.domain_limits
            domain_bd_gap = domain_upper_bd - domain_lower_bd

            s = torch.rand(self.training_points_num, self.system.ns) * domain_bd_gap + domain_lower_bd

            if self.augment_data.shape[0] > self.maximum_augment_data_num:
                self.augment_data = self.augment_data[-self.maximum_augment_data_num:]

            # generate training data
            s_samples = torch.cat((s, self.augment_data), dim=0)
            
            assert self.training_grid_gap is None
            s_gridding_gap = torch.zeros(s_samples.shape[0], self.system.ns)


            # split into training and validation
            random_indices = torch.randperm(s_samples.shape[0])
            val_pts = int(s_samples.shape[0] * self.val_split)
            validation_indices = random_indices[:val_pts]
            training_indices = random_indices[val_pts:]
            


            # store the training data
            self.s_training = s_samples[training_indices]
            self.s_grid_gap_training = s_gridding_gap[training_indices]
            self.s_validation = s_samples[validation_indices]
            self.s_grid_gap_validation = s_gridding_gap[validation_indices]
            
            # generate other information
            self.safe_mask_training = self.system.safe_mask(self.s_training)  # self.s_training.norm(dim=-1) <= 0.6
            self.unsafe_mask_training = self.system.unsafe_mask(self.s_training)
            
            self.safe_mask_validation =  self.system.safe_mask(self.s_validation) # self.s_validation.norm(dim=-1) <= 0.6
            self.unsafe_mask_validation = self.system.unsafe_mask(self.s_validation)


        elif self.train_mode == 2:
            
            message = "training_grid_gap is None, please specify the training_grid_gap"
            assert self.training_grid_gap is not None, message
            message = f"training_grid_gap should have shape ({self.system.ns},), but got {self.training_grid_gap.shape}"
            assert self.training_grid_gap.shape == (self.system.ns,), message

            domain_lower_bd, domain_upper_bd = self.system.domain_limits
            domain_bd_gap = domain_upper_bd - domain_lower_bd
            
            del self.new_tree
            self.new_tree = Tree()

            root_s = (domain_lower_bd+ domain_upper_bd )/2
            root_s = root_s.reshape(1, -1)
            root_grid_gap = domain_bd_gap.reshape(1, -1)
            satisfy_constraint = True
            root_data = [root_s, root_grid_gap, satisfy_constraint]

            self.new_tree.create_node(f"{self.new_tree.size()}", identifier=self.uniname_of_data(root_data), data=root_data)  # root node

            s_train_grid_list = []
            sample_data = []
            sample_data_grid_gap = []


            for i in range(self.system.ns):
                s_i_grid_train = torch.arange(domain_lower_bd[i], domain_upper_bd[i], self.training_grid_gap[i])
                s_train_grid_list.append(s_i_grid_train.float())
              
            mesh_grids_train = torch.meshgrid(s_train_grid_list)
            unsqueez_mesh_grid_train = [ torch.unsqueeze(mesh_grid, dim=0) for mesh_grid in mesh_grids_train ]
            mesh_grids_train = torch.vstack(unsqueez_mesh_grid_train)
            data_points_train = torch.flatten(mesh_grids_train, start_dim=1)
         
            s_grid = data_points_train.T
            for i in range(s_grid.shape[0]):
                s = s_grid[i,:].reshape(1, -1)
                grid_gap = self.training_grid_gap.reshape(1, -1)
                satisfy_constraint = True
                data = [s, grid_gap, satisfy_constraint]
                self.new_tree.create_node(f"{self.new_tree.size()}", identifier=self.uniname_of_data(data), data=data, parent=self.new_tree.root)
                
                sample_data.append(s)
                sample_data_grid_gap.append(grid_gap)

            # while( self.get_minimum_grid_gap() > self.training_grid_gap[0]): 
            #     for leave_node in self.new_tree.leaves():
            #         self.expand_leave(leave_node)

               
            # generate training data
            s_samples = torch.cat(sample_data, dim=0)
            sample_data_grid_gap = torch.cat(sample_data_grid_gap, dim=0)

            # split into training and validation
            random_indices = torch.randperm(s_samples.shape[0])
            val_pts = int(s_samples.shape[0] * self.val_split)
            validation_indices = random_indices[:val_pts]
            training_indices = random_indices[val_pts:]
            
            # store the data
            self.s_training = s_samples[training_indices]
            self.s_grid_gap_training = sample_data_grid_gap[training_indices]
            
            self.s_validation = s_samples[validation_indices]
            self.s_grid_gap_validation = sample_data_grid_gap[validation_indices]
            
            # generate other information
            self.safe_mask_training = self.system.safe_mask(self.s_training)  # self.s_training.norm(dim=-1) <= 0.6
            self.unsafe_mask_training = self.system.unsafe_mask(self.s_training)

            self.safe_mask_validation =  self.system.safe_mask(self.s_validation) # self.s_validation.norm(dim=-1) <= 0.6
            self.unsafe_mask_validation = self.system.unsafe_mask(self.s_validation)
            
   
            
    def set_dataset(self):
        print("Full dataset:")
        print(f"\t{self.s_training.shape[0]} training")
        print(f"\t{self.s_validation.shape[0]} validation")
        print("\t----------------------")

        print(f"\t{self.safe_mask_training.sum()} safe points")
        print(f"\t({self.safe_mask_validation.sum()} val)")
        print(f"\t{self.unsafe_mask_training.sum()} unsafe points")
        print(f"\t({self.unsafe_mask_validation.sum()} val)")
        print("\t----------------------")
        

        # Turn these into tensor datasets
        self.training_data = TensorDataset(
            self.s_training,
            self.safe_mask_training,
            self.unsafe_mask_training,
            self.s_grid_gap_training
        )
        self.validation_data = TensorDataset(
            self.s_validation,
            self.safe_mask_validation,
            self.unsafe_mask_validation,
            self.s_grid_gap_validation
            )

    def augment_dataset(self):
        
        print("Augmenting dataset...")
        self.verified = -1
        if self.train_mode == 0:
            domain_lower_bd, domain_upper_bd = self.system.domain_limits
            domain_bd_gap = domain_upper_bd - domain_lower_bd

            s = torch.rand(self.training_points_num, self.system.ns) * domain_bd_gap + domain_lower_bd
           
            # generate training data
            s_samples = s
            
            assert self.training_grid_gap is None
            sample_data_grid_gap = torch.zeros(s_samples.shape[0], self.system.ns)

        if self.train_mode == 1:
            domain_lower_bd, domain_upper_bd = self.system.domain_limits
            domain_bd_gap = domain_upper_bd - domain_lower_bd

            s = torch.rand(self.training_points_num, self.system.ns) * domain_bd_gap + domain_lower_bd
           
            if self.augment_data.shape[0] > self.maximum_augment_data_num:
                self.augment_data = self.augment_data[-self.maximum_augment_data_num:]

            # generate training data
            s_samples = torch.cat((s, self.augment_data), dim=0)
            
            assert self.training_grid_gap is None
            sample_data_grid_gap = torch.zeros(s_samples.shape[0], self.system.ns)

        if self.train_mode == 2:
            sample_data = []
            sample_data_grid_gap = []
            
            test_flag = True
            reach_minimum_gap = True

            for leave_node in self.new_tree.leaves():
                if leave_node.data[2] == False:
                    if self.expand_leave(leave_node):
                        reach_minimum_gap = False
                    test_flag = False
            
            
            if reach_minimum_gap:
                print("all hyperrectangles reach the minimum grid gap")
                self.verified = 0
                self.minimum_grid_gap = max(self.minimum_grid_gap * 0.8, 0.005)
                print("minimum grid gap is updated to be ", self.minimum_grid_gap)
            
            if test_flag:
                print("all hyperrectangles satisfy the constraint")
                self.verified = 1

            for leave_node in self.new_tree.leaves():
                if leave_node.data[2] == False:
                    sample_data.append(leave_node.data[0])
                    sample_data_grid_gap.append(leave_node.data[1])

            try:
                s_samples = torch.cat(sample_data, dim=0)
                sample_data_grid_gap = torch.cat(sample_data_grid_gap, dim=0)
            except:
                print("no new data generated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"sample_data: {sample_data}")
                s_samples = torch.zeros(1, self.system.ns)
                sample_data_grid_gap = torch.zeros(1, self.system.ns)

            self.augment_data = torch.cat( (self.augment_data,s_samples), dim=0)
            # self.training_points_num = int( min(5e5, max(int(5 * self.augment_data.shape[0]), self.minimum_training_points_num)) )

        # if self.train_mode == 3:
        #     G = 9.81 
        #     l = 1
        #     m = 1
        #     b = 0.1

        #     x1 = Variable("x1")
        #     x2 = Variable("x2")
        #     vars_ = [x1,x2]

        #     para = self.model.parameters()
        #     w1 = next(para).data.detach().cpu().numpy()
        #     b1 = next(para).data.detach().cpu().numpy()
        #     w2 = next(para).data.detach().cpu().numpy()
        #     b2 = next(para).data.detach().cpu().numpy()
        #     w3 = next(para).data.detach().cpu().numpy()
        #     b3 = next(para).data.detach().cpu().numpy()

        #     start = timeit.default_timer()
        #     lqr = torch.tensor([[ -2.0,  -3.2699]])    # lqr solution
        #     q = torch.nn.parameter.Parameter(lqr)
            
        #     config = Config()
        #     config.use_polytope_in_forall = True
        #     config.use_local_optimization = True
        #     config.precision = 1e-1

        #     u_NN = (-2.0*x1 - 3.2699*x2) 
        #     print('u_NN',u_NN)
        #     f = [ x2,
        #          (3*m*G*l*sin(x1) + 3*u_NN - 3*b*x2) /(m*l**2)]

        #     # Candidate V
        #     z1 = np.dot(vars_,w1.T)+b1

        #     a1 = []
        #     for j in range(0,len(z1)):
        #         a1.append(tanh(z1[j]))
        #     z2 = np.dot(a1,w2.T)+b2

        #     a2 = []
        #     for j in range(0,len(z2)):
        #         a2.append(tanh(z2[j]))
        #     z3 = np.dot(a2,w3.T)+b3

        #     V_learn = tanh(z3.item())

        #     print('===========Verifying==========')        
        #     start_ = timeit.default_timer() 
        #     result= CheckCBF(vars_, f, V_learn, config)
        #     stop_ = timeit.default_timer() 

        #     domain_lower_bd, domain_upper_bd = self.system.domain_limits
        #     domain_bd_gap = domain_upper_bd - domain_lower_bd

        #     s = torch.rand(self.training_points_num, self.system.ns) * domain_bd_gap + domain_lower_bd
           
        #     if (result): 
        #         print("Not a CBF function. Found counterexample: ")
        #         print(result)
        #         ce = AddCounterexamplesCBF(s,result,10)
        #         self.SMT_verification_time += (stop_ - start_)
        #         self.SMT_CE_num += ce.shape[0]
        #         self.augment_data = torch.cat((self.augment_data,ce), dim=0)
        #         if self.augment_data.shape[0] > self.maximum_augment_data_num:
        #             self.augment_data = self.augment_data[-self.maximum_augment_data_num:]
        #     else:  
        #         self.SMT_verification_time += (stop_ - start_)
        #         self.verified = True
        #         print("Satisfy conditions!!!!!!!!!!!!!")
        #         # print(V_learn, " is a CBF.")
            
        #     print('==============================') 
        #     s_samples = s
        #     s_samples = torch.cat((s_samples, self.augment_data), dim=0)
        #     assert self.training_grid_gap is None
        #     sample_data_grid_gap = torch.zeros(s_samples.shape[0], self.system.ns)
        #     torch.save(self.augment_data, "SMT_augment_data.pt")
            

        # split into training and validation
        random_indices = torch.randperm(s_samples.shape[0])
        val_pts = int(s_samples.shape[0] * self.val_split)
        validation_indices = random_indices[:val_pts]
        training_indices = random_indices[val_pts:]
        
        # store the training data
        self.s_training = s_samples[training_indices]
        self.s_validation = s_samples[validation_indices]
        
        self.s_grid_gap_training = sample_data_grid_gap[training_indices]
        self.s_grid_gap_validation = sample_data_grid_gap[validation_indices]

        # generate other information
        self.safe_mask_training = self.system.safe_mask(self.s_training)  # self.s_training.norm(dim=-1) <= 0.6
        self.unsafe_mask_training = self.system.unsafe_mask(self.s_training)
        
        self.safe_mask_validation =  self.system.safe_mask(self.s_validation) # self.s_validation.norm(dim=-1) <= 0.6
        self.unsafe_mask_validation = self.system.unsafe_mask(self.s_validation)
        

    def train_dataloader(self):
        """Make the DataLoader for training data"""
        return DataLoader(
            self.training_data,
            batch_size=self.train_batch_size,
            num_workers=8,
        )


        
if __name__ == "__main__":
    
    data_module = TrainingDataModule(system=point_robot, training_points_num=100000)

    data_module.prepare_data()