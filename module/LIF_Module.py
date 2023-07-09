import torch
from torch import nn
from module.LIF import *
from module.TA import *


class AttLIF(nn.Module):
    def __init__(self,
                 inputSize,
                 hiddenSize,
                 spikeActFun,
                 attention='T',
                 onlyLast=False,
                 useBatchNorm=False,
                 useLayerNorm=False,
                 init_method=None,
                 scale=0.3,
                 pa_dict=None,
                 pa_train_self=False,
                 bias=True,
                 reduction=16,
                 T=60,
                 p=0,
                 track_running_stats=False,
                 mode_select='spike',
                 mem_act=torch.relu,
                 TR_model='NTR',
                 ):
        super().__init__()
        self.onlyLast = onlyLast

        self.useBatchNorm = useBatchNorm

        self.network = nn.Sequential()
        self.attention_flag = attention
        self.linear = nn.Linear(
            in_features=inputSize,
            out_features=hiddenSize,
            bias=bias,
        )

        if self.useBatchNorm:
            self.BNLayer = nn.BatchNorm1d(num_features=hiddenSize,
                                          track_running_stats=track_running_stats)

        if init_method is not None:
            paramInit(model=self.linear, method=init_method)
        if self.attention_flag == 'T':
            self.attention = Tlayer(timeWindows=T, dimension=3, reduction=reduction)
        elif self.attention_flag == 'no':
            pass

        self.network.add_module('IF',
                                IFCell(inputSize,
                                       hiddenSize,
                                       spikeActFun,
                                       bias=bias,
                                       scale=scale,
                                       pa_dict=pa_dict,
                                       pa_train_self=pa_train_self,
                                       p=p,
                                       mode_select=mode_select,
                                       mem_act=mem_act,
                                       TR_model=TR_model,
                                       ))

    def forward(self, data):

        for layer in self.network:
            layer.reset()

        b, t, _ = data.size()
        output = self.linear(data.reshape(b * t, -1))

        if self.useBatchNorm:
            output = self.BNLayer(output)

        outputsum = output.reshape(b, t, -1)

        if self.attention_flag == 'T':
            data = self.attention(outputsum)
        elif self.attention_flag == 'no':
            data = outputsum

        for step in range(list(data.size())[1]):
            out = data[:, step, :]
            for layer in self.network:
                out = layer(out)
            output = out

            if step == 0:
                temp = list(output.size())
                temp.insert(1, list(data.size())[1])
                outputsum = torch.zeros(temp)
                if outputsum.device != data.device:
                    outputsum = outputsum.to(data.device)
            outputsum[:, step, :] = output

        if self.onlyLast:
            return output
        else:
            return outputsum


class ConvAttLIF(nn.Module):
    def __init__(self,
                 inputSize,
                 hiddenSize,
                 kernel_size,
                 spikeActFun,
                 attention='T',
                 bias=True,
                 onlyLast=False,
                 padding=1,
                 useBatchNorm=False,
                 init_method=None,
                 scale=0.02,
                 pa_dict=None,
                 pa_train_self=False,
                 reduction=16,
                 T=60,
                 stride=1,
                 pooling_kernel_size=1,
                 p=0,
                 track_running_stats=False,
                 mode_select='spike',
                 mem_act=torch.relu,
                 TR_model='NTR',
                 ):
        super().__init__()

        self.onlyLast = onlyLast
        self.attention_flag = attention

        self.conv2d = nn.Conv2d(in_channels=inputSize,
                                out_channels=hiddenSize,
                                kernel_size=kernel_size,
                                bias=True,
                                padding=padding,
                                stride=stride,
                                )

        if init_method is not None:
            paramInit(model=self.conv2d, method=init_method)

        self.useBatchNorm = useBatchNorm

        if self.useBatchNorm:
            self.BNLayer = nn.BatchNorm2d(hiddenSize, track_running_stats=track_running_stats)

        self.pooling_kernel_size = pooling_kernel_size
        if self.pooling_kernel_size > 1:
            self.pooling = nn.AvgPool2d(kernel_size=pooling_kernel_size)

        if self.attention_flag == 'T':
            self.attention = Tlayer(timeWindows=T, dimension=5, reduction=reduction)
        elif self.attention_flag == 'no':
            pass
        self.network = nn.Sequential()
        self.network.add_module('ConvIF',
                                ConvIFCell(inputSize=inputSize,
                                           hiddenSize=hiddenSize,
                                           kernel_size=kernel_size,
                                           bias=bias,
                                           spikeActFun=spikeActFun,
                                           padding=padding,
                                           scale=scale,
                                           pa_dict=pa_dict,
                                           pa_train_self=pa_train_self,
                                           p=p,
                                           mode_select=mode_select,
                                           mem_act=mem_act,
                                           TR_model=TR_model,
                                           ))

    def forward(self, data):

        for layer in self.network:
            layer.reset()

        b, t, c, h, w = data.size()
        out = data.reshape(b * t, c, h, w)
        output = self.conv2d(out)

        if self.useBatchNorm:
            output = self.BNLayer(output)

        if self.pooling_kernel_size > 1:
            output = self.pooling(output)

        _, c, h, w = output.size()
        outputsum = output.reshape(b, t, c, h, w)

        if self.attention_flag == 'T':
            data = self.attention(outputsum)
        elif self.attention_flag == 'no':
            data = outputsum

        for step in range(list(data.size())[1]):
            out = data[:, step, :, :, :]
            for layer in self.network:
                out = layer(out)
            output = out

            if step == 0:
                temp = list(output.size())
                temp.insert(1, list(data.size())[1])
                outputsum = torch.zeros(temp)
                if outputsum.device != data.device:
                    outputsum = outputsum.to(data.device)

            outputsum[:, step, :, :, :] = output

        if self.onlyLast:
            return output
        else:
            return outputsum
