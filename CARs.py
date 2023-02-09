from system import Car
import torch


car1 = Car()

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

car1.set_domain_limits(domain_lower_bd, domain_upper_bd)
car1.set_control_limits(control_lower_bd, control_upper_bd)
car1.set_state_constraints(rou)
