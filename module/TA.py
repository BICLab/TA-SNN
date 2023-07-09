import torch
from torch import nn

class Tlayer(nn.Module):
    '''
    Temporal-wise Attention Layer
    '''

    def __init__(self, timeWindows, reduction=16, dimension=3):
        super(Tlayer, self).__init__()
        if dimension == 3:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        elif dimension == 4:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.temporal_excitation = nn.Sequential(nn.Linear(timeWindows, int(timeWindows // reduction)),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(int(timeWindows // reduction), timeWindows),
                                                 nn.Sigmoid()
                                                 )

    def forward(self, input):
        b = list(input.size())[0]
        t = list(input.size())[1]

        temp = self.avg_pool(input)
        y = temp.view(b, t)
        y = self.temporal_excitation(y).view(temp.size())

        y = torch.mul(input, y)

        return y

