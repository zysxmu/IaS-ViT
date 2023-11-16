import torch
from quant.block_recon import LinearTempDecay
from quant.data_utils import save_grad_data, save_inp_oup_data
from quant.quant_model import set_quant_state
from quant.quantizer import lp_loss
from quant.quant_modules import *


def layer_reconstruction(model, layer, teacher_model, T_module, cali_data: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.001, opt_mode: str = 'mse',
                         asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, weight_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         multi_gpu: bool = False, last_stage:bool = False):
    """
    Block reconstruction to optimize the output from each layer.
    """
    set_quant_state(layer, input_quant=act_quant, weight_quant=weight_quant)

    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.weight.requires_grad = False
            module.bias.requires_grad = False
        if isinstance(module, UniformQuantizer):
            if hasattr(module, 'delta') and module.delta is not None:
                module.delta.requires_grad = False
            if hasattr(module, 'zero_point') and module.zero_point is not None:
                module.zero_point.requires_grad = False
        if isinstance(module, LogSqrt2Quantizer):
            if hasattr(module, 'delta') and module.delta is not None:
                module.delta.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, layer.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

    loss_mode = 'none'
    rec_loss = opt_mode

    loss_func = LossFunction(layer, round_loss=loss_mode, weight=weight,
                             max_count=iters, rec_loss=rec_loss, b_range=b_range,
                             decay_start=0, warmup=warmup, p=p)

    # Save data before optimizing the rounding
    cached_inps, cached_outs = save_inp_oup_data(model, layer, teacher_model, T_module,
                                                 cali_data, asym, act_quant, weight_quant, batch_size, last_stage=last_stage)
    cached_grads = None
    device = 'cuda'

    for i in range(iters):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        cur_grad = cached_grads[idx] if opt_mode != 'mse' else None

        optimizer.zero_grad()
        out_quant = layer(cur_inp)

        err = loss_func(out_quant, cur_out, cur_grad)
        err.backward(retain_graph=True)

        optimizer.step()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()

class LossFunction:
    def __init__(self,
                 layer,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.layer = layer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        total_loss = rec_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} \tcount={}'.format(
                  float(total_loss),self.count))
        return total_loss

