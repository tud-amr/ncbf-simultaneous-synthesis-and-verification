from Car import Car
from InvertedPendulum import InvertedPendulum
import torch

####################### create an one-D car object ######################
car1 = Car(ns=2, nu=1)

domain_lower_bd = torch.Tensor([-2, -2]).float()
domain_upper_bd = -domain_lower_bd

control_lower_bd =torch.Tensor([-1]).float()
control_upper_bd = -control_lower_bd
    
def rou(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.unsqueeze(s[:, 0] + 1, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + 1, dim=1)
    rou_3 = torch.unsqueeze(s[:, 1] + 1, dim=1)
    rou_4 = torch.unsqueeze( -s[:, 1] + 1, dim=1)
    return torch.hstack( (rou_1, rou_2, rou_3, rou_4) ) 

def rou_n(s: torch.Tensor) -> torch.Tensor:
    s_norm = torch.norm(s, dim=1).unsqueeze(dim=-1)

    return - s_norm + 0.6

car1.set_domain_limits(domain_lower_bd, domain_upper_bd)
car1.set_control_limits(control_lower_bd, control_upper_bd)
car1.set_state_constraints(rou)
car1.set_nominal_state_constraints(rou_n)

######################## creat inverted pendulum object ##################


domain_lower_bd2 = torch.Tensor([-3, -4]).float()
domain_upper_bd2 = -domain_lower_bd2

control_lower_bd2 =torch.Tensor([-2]).float()
control_upper_bd2 = -control_lower_bd2
    
def rou2(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.unsqueeze(s[:, 0] + torch.pi / 3, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + torch.pi / 3 , dim=1)
    return torch.hstack( (rou_1, rou_2) ) 

def rou_n2(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.pi / 5 -  torch.norm(s, dim=-1)
    
    return rou_1.unsqueeze(dim=1)


inverted_pendulum_1 = InvertedPendulum(m=0.1, b=0.1)

inverted_pendulum_1.set_domain_limits(domain_lower_bd2, domain_upper_bd2)
inverted_pendulum_1.set_control_limits(control_lower_bd2, control_upper_bd2)
inverted_pendulum_1.set_state_constraints(rou2)
inverted_pendulum_1.set_nominal_state_constraints(rou_n2)

