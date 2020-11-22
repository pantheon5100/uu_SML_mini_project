import torch.nn as nn
import torch


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in=13, H=27, D_out=2):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.drop2 = nn.Dropout(0.8)
        self.linear1Ac = nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(H)
        self.linear2 = torch.nn.Linear(H, 55)
        self.drop1 = torch.nn.Dropout(0.7)
        self.linear2Ac = nn.ReLU()
        self.linear3 = torch.nn.Linear(55, 111)
        self.drop3 = nn.Dropout(0.6)
        self.linear3Ac = nn.ReLU()
        self.linear4 = torch.nn.Linear(111, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.linear1(x)
        x = self.drop2(x)
        x = self.linear1Ac(x)
        # x = self.bn1(x)
        x = self.linear2(x)
        x = self.drop1(x)
        x = self.linear1Ac(x)
        x = self.linear3(x)
        x = self.linear3Ac(x)
        # x = self.drop3(x)
        y_pred = self.linear4(x)
        return y_pred
