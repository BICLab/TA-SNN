import torch
import torch.nn as nn
from module.LIF_Module import AttLIF, ConvAttLIF
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

    # cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
    cfg_cnn = [(2, 64, 1, 1, 3,),
               (64, 128, 1, 1, 3,),
               (128, 128, 1, 1, 3,),
               ]
    # pooling kernel_size
    cfg_pool = [1, 2, 2]
    # fc layer
    cfg_fc = [cfg_cnn[2][1] * 8 * 8, 256, config.target_size]

    class Net(nn.Module):

        def __init__(self, ):
            super(Net, self).__init__()

            in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
            pooling_kernel_size = cfg_pool[0]
            self.convAttLIF0 = ConvAttLIF(
                inputSize=in_planes,
                hiddenSize=out_planes,
                kernel_size=(kernel_size, kernel_size),
                spikeActFun=ActFun.apply,
                init_method=config.init_method,
                useBatchNorm=True,
                pooling_kernel_size=pooling_kernel_size,
                T=config.T,
                pa_dict={'alpha': config.alpha,  'beta': config.beta, 'Vreset': config.Vreset, 'Vthres': config.Vthres},
                reduction= config.reduction,
                track_running_stats = config.track_running_stats,
                attention='no',
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
            )

            in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
            pooling_kernel_size = cfg_pool[1]
            self.convAttLIF1 = ConvAttLIF(
                inputSize=in_planes,
                hiddenSize=out_planes,
                kernel_size=(kernel_size, kernel_size),
                spikeActFun=ActFun.apply,
                init_method=config.init_method,
                useBatchNorm=True,
                pooling_kernel_size=pooling_kernel_size,
                T=config.T,
                pa_dict={'alpha': config.alpha,  'beta': config.beta, 'Vreset': config.Vreset, 'Vthres': config.Vthres},
                reduction=config.reduction,
                track_running_stats=config.track_running_stats,
                attention='no',
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
            )

            in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
            pooling_kernel_size = cfg_pool[2]
            self.convAttLIF2 = ConvAttLIF(
                inputSize=in_planes,
                hiddenSize=out_planes,
                kernel_size=(kernel_size, kernel_size),
                spikeActFun=ActFun.apply,
                init_method=config.init_method,
                useBatchNorm=True,
                pooling_kernel_size=pooling_kernel_size,
                T=config.T,
                pa_dict={'alpha': config.alpha,  'beta': config.beta, 'Vreset': config.Vreset, 'Vthres': config.Vthres},
                reduction=config.reduction,
                track_running_stats=config.track_running_stats,
                attention='no',
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
            )

            self.FC0 = AttLIF(
                inputSize=cfg_fc[0],
                hiddenSize=cfg_fc[1],
                spikeActFun=ActFun.apply,
                useBatchNorm=True,
                T=config.T,
                pa_dict={'alpha': config.alpha,  'beta': config.beta, 'Vreset': config.Vreset, 'Vthres': config.Vthres},
                reduction=config.reduction,
                track_running_stats=config.track_running_stats,
                attention='no',
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
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
                attention='no',
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
            )

        def forward(self, input):
            b, t, _, _, _ = input.size()
            outputs = input

            outputs = self.convAttLIF0(outputs)
            outputs = self.convAttLIF1(outputs)
            outputs = self.convAttLIF2(outputs)

            outputs = outputs.reshape(b, t, -1)

            outputs = self.FC0(outputs)

            outputs = self.FC1(outputs)
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

