from safe_rl_cbf.Models.common_header import *


domain_lower_bd = torch.Tensor([-0.3, -0.3, -0.5, -0.5]).float()
domain_upper_bd = torch.Tensor([4.3, 4.3, 0.5, 0.5]).float()

control_lower_bd = torch.Tensor([-1, -1]).float()
control_upper_bd = -control_lower_bd
    

def rou(s: torch.Tensor) -> torch.Tensor:
    """
    return the boolean tensor for each constraint. The rou(s)>0 means the state is safe
    inputs:
        s: bs*ns torch.Tensor, the state
    return:
        rou_s: (bs,1) torch.Tensor, the constraints value function
    """

    rou_1 = torch.unsqueeze(s[:, 0] - 0, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + 4, dim=1)
    rou_3 = torch.unsqueeze(s[:, 1] - 0, dim=1)
    rou_4 = torch.unsqueeze( -s[:, 1] + 4, dim=1)
    
    dist = torch.abs( s[:, 0:2] - torch.tensor([2, 1]).to(s.device).reshape(1, 2) )
    rou_9 = torch.max(dist[:, 0]-0.5, dist[:, 1]-1).reshape(-1, 1) 
    

    
    # get boolean tensor for each constraint
    rou_s = torch.min( torch.hstack( (rou_1, rou_2, rou_3, rou_4, rou_9) ), dim=1, keepdim=True).values
    assert rou_s.shape == (s.shape[0],1)
    return rou_s