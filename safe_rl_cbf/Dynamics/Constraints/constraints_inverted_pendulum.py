from safe_rl_cbf.Models.common_header import *


domain_lower_bd = torch.Tensor([-torch.pi, -5]).float()
domain_upper_bd = -domain_lower_bd

control_lower_bd = torch.Tensor([-12]).float()
control_upper_bd = -control_lower_bd
    
def rou(s: torch.Tensor) -> torch.Tensor:
    """
    return the boolean tensor for each constraint. The rou(s)>0 means the state is safe
    inputs:
        s: bs*ns torch.Tensor, the state
    return:
        rou_s: (bs,1) torch.Tensor, the constraints value function
    """

    rou_1 = torch.unsqueeze(s[:, 0] + torch.pi * 5 / 6, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + torch.pi * 5 / 6 , dim=1)
    rou_3 = torch.unsqueeze(s[:, 1] + 4, dim=1)
    rou_4 = torch.unsqueeze( - s[:, 1] + 4 , dim=1)

    # get boolean tensor for each constraint
    rou_s = torch.min( torch.hstack( (rou_1, rou_2, rou_3, rou_4) ), dim=1, keepdim=True).values
    assert rou_s.shape == (s.shape[0],1)
    return rou_s
