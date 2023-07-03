import torch
from dynamic_system_instances import car1, inverted_pendulum_1
from MyNeuralNetwork_with_OptNet import NeuralNetwork_with_OptNet
from DataModule import DataModule
from treelib import Tree, Node
from dynamic_system_instances import car1, inverted_pendulum_1
from control_affine_system import ControlAffineSystem
from itertools import product
import matplotlib.pyplot as plt

def expand_leave(tree, leave_node):
    leave_node_id = leave_node.identifier
    leave_node_s = leave_node.data[0]
    leave_node_grid_gap = leave_node.data[1]
    s_dim = leave_node_s.shape[1]
    dir = torch.tensor([0.5, -0.5])
    combine = list(product(dir, repeat=s_dim))

    for i in range(len(combine)):
        coefficent = torch.tensor(combine[i])
        new_s = leave_node_s + leave_node_grid_gap * coefficent
        new_grid_gap = leave_node_grid_gap / 2
        new_data = (new_s, new_grid_gap)
        tree.create_node(f"{tree.size()}", identifier=uniname_of_data(new_data), data=new_data, parent=leave_node_id)

def get_minimum_grid_gap(tree):
    grid_gaps = []
    for leave_node in tree.leaves():
        grid_gaps.append( torch.min(leave_node.data[1]))
        
    return min(grid_gaps)

def uniname_of_data(data):
    s = data[0]
    id = str('')
    for i in range(s.shape[1]):
        id = id + str(s[0,i].item())

    return id
    

data_module = DataModule(system=inverted_pendulum_1, train_grid_gap=1, test_grid_gap=0.01)

domain_lower_bd, domain_upper_bd = data_module.system.domain_limits
domain_bd_gap = domain_upper_bd - domain_lower_bd

print(f"domain_lower_bd = {domain_lower_bd}")
print(f"domain_upper_bd = {domain_upper_bd}")
print(f"domain_bd_gap = {domain_bd_gap}")


new_tree = Tree()

root_s = (domain_lower_bd+ domain_upper_bd )/2
root_s = root_s.reshape(1, -1)
root_grid_gap = domain_bd_gap.reshape(1, -1)
root_data = (root_s, root_grid_gap)

new_tree.create_node(f"{new_tree.size()}", identifier=uniname_of_data(root_data), data=root_data)  # root node

while( get_minimum_grid_gap(new_tree) > 0.3): 
    for leave_node in new_tree.leaves():
        expand_leave(new_tree, leave_node)

data_search = (torch.tensor([0.0, 0.0]).reshape(1, -1), torch.tensor([6.2832, 10.0]).reshape(1, -1))
id = uniname_of_data(data_search)
node1 = new_tree.get_node(id)
print(node1.identifier)
exit()
x = []
y = []
training_data = []
for leave_node in new_tree.leaves():
    x.append(leave_node.data[0][0, 0])
    y.append(leave_node.data[0][0, 1])
    training_data.append(leave_node.data[0])

training_data = torch.cat(training_data, dim=0)
print(f"training_data.shape = {training_data.shape}")

plt.figure()
plt.scatter(x, y, s=1, c='#FF00FF')
plt.show()

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")


# data_module = DataModule(system=inverted_pendulum_1, val_split=0.1, train_batch_size=64, test_batch_size=128, train_grid_gap=0.1, test_grid_gap=0.01)

# NN = NeuralNetwork_with_OptNet.load_from_checkpoint("masterthesis_test/OptNet_logs/lightning_logs/version_3/checkpoints/epoch=26-step=4239.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module, learn_shape_epochs=2 )

# s0 = torch.rand(3, 2).float()

# s_star = NN(s0)

# print(s_star)
