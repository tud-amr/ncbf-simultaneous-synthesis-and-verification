import torch
from dynamic_system_instances import car1, inverted_pendulum_1
from MyNeuralNetwork_with_OptNet import NeuralNetwork_with_OptNet
from DataModule import DataModule

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")


data_module = DataModule(system=inverted_pendulum_1, val_split=0.1, train_batch_size=64, test_batch_size=128, train_grid_gap=0.1, test_grid_gap=0.01)

NN = NeuralNetwork_with_OptNet.load_from_checkpoint("masterthesis_test/OptNet_logs/lightning_logs/version_3/checkpoints/epoch=26-step=4239.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module, learn_shape_epochs=2 )

s0 = torch.rand(3, 2).float()

s_star = NN(s0)

print(s_star)
