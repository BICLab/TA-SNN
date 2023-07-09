import torch
import torch.nn as nn
from module.LIF_Module import AttLIF
from torch import optim


def create_net(config):
    # Net
    # define approximate firing function

    class ActFun(torch.autograd.Function):
        def __init__(self):
            super(ActFun, self).__init__()

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input.ge(0.).float()

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            temp = abs(input) < config.lens
            return grad_output * temp.float() / (2 * config.lens)

    # fc layer
    cfg_fc = [700 // config.ds, 128, 128, config.target_size]

    class Net(nn.Module):

        def __init__(self, ):
            super(Net, self).__init__()

            self.FC0 = AttLIF(
                inputSize=cfg_fc[0],
                hiddenSize=cfg_fc[1],
                spikeActFun=ActFun.apply,
                useBatchNorm=True,
                T=config.T,
                pa_dict={'alpha': config.alpha, 'beta': config.beta, 'Vreset': config.Vreset, 'Vthres': config.Vthres},
                reduction=config.reduction,
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
                attention='no',
            )
            self.FC1 = AttLIF(
                inputSize=cfg_fc[1],
                hiddenSize=cfg_fc[2],
                spikeActFun=ActFun.apply,
                useBatchNorm=True,
                T=config.T,
                pa_dict={'alpha': config.alpha, 'beta': config.beta, 'Vreset': config.Vreset, 'Vthres': config.Vthres},
                reduction=config.reduction,
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
                attention='no',
            )

            self.FC2 = AttLIF(
                inputSize=cfg_fc[2],
                hiddenSize=cfg_fc[3],
                spikeActFun=ActFun.apply,
                useBatchNorm=True,
                T=config.T,
                pa_dict={'alpha': config.alpha, 'beta': config.beta, 'Vreset': config.Vreset, 'Vthres': config.Vthres},
                reduction=config.reduction,
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
                attention='no',
            )
            self.dropOut = nn.Dropout(p=0.5)

        def forward(self, input):
            b, t, _ = input.size()
            outputs = input

            outputs = self.FC0(outputs)
            outputs = self.FC1(outputs)
            outputs = self.dropOut(outputs)
            outputs = self.FC2(outputs)
            outputs = torch.sum(outputs, dim=1)
            outputs = outputs / t

            return outputs

    config.model = Net().to(config.device)

    # optimizer
    config.optimizer = optim.Adam(
        config.model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay)

    config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=config.optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    config.model = nn.DataParallel(
        config.model,
        device_ids=config.device_ids)
