
import torch

class ActionNet(torch.nn.Module):
    
    def __init__(self):
        super(ActionNet, self).__init__()
        self.MlpExtractor = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
        )

        self.action_net =  torch.nn.Linear(64, 1)
        

    def forward(self, x):
        x = self.MlpExtractor(x)
        return self.action_net(x)