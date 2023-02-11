import torch
from dynamic_system_instances import car1, inverted_pendulum_1
from MyNeuralNetwork import *
from DataModule import DataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

NN = torch.load("NN.pt")

torch.save({
            'model_state_dict': NN.state_dict(),
            }, "NN_checkpoint.pt")


