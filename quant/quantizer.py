import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

class UniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, is_act:bool = False):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.channel_wise = channel_wise
        self.is_act = is_act
        self.register_buffer('inited', torch.zeros(1))
    
    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={}, bit = {})".format(self.inited, self.channel_wise, self.n_bits)
        return s

    def forward(self, x: torch.Tensor):
        if self.inited == 0:
            delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.delta, self.zero_point = Parameter(delta).contiguous(), Parameter(zero_point).contiguous()
            self.inited.fill_(1)

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            if self.is_act:
                if len(x.shape) == 3:
                    n_channels = x_clone.shape[-1]
                elif len(x.shape) == 4: 
                    n_channels = x_clone.shape[1] 
                elif len(x.shape) == 2: 
                    n_channels = x_clone.shape[1]
                else:
                    raise NotImplementedError

                if len(x.shape) == 4: 
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=-1)[0].max(dim=-1)[0]
                elif len(x.shape) == 2:
                    x_max = x_clone.abs().max(dim=0)[0]
                elif len(x.shape) == 3: 
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
                else:
                    raise NotImplementedError

                delta = x_max.clone()
                zero_point = x_max.clone()
                for c in range(n_channels):
                    if len(x.shape) == 3:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                    elif len(x.shape) == 4:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c,...], channel_wise=False)
                    else:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False)

                if len(x.shape) == 4:
                    delta = delta.view(1, -1, 1, 1)
                    zero_point = zero_point.view(1, -1, 1, 1)
                elif len(x.shape) == 2:
                    delta = delta.view(1, -1)
                    zero_point = zero_point.view(1, -1)
                elif len(x.shape) == 3:
                    delta = delta.view(1, 1, -1)
                    zero_point = zero_point.view(1, 1, -1)
                else:
                    raise NotImplementedError
                
            else: 
                n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
                if len(x.shape) == 4:
                    x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                elif len(x.shape) == 2:
                    x_max = x_clone.abs().max(dim=-1)[0]
                elif len(x.shape) == 3:
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
                else:
                    raise NotImplementedError

                delta = x_max.clone()
                zero_point = x_max.clone()
                for c in range(n_channels):
                    if len(x.shape) == 3:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                    else:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)

                if len(x.shape) == 4:
                    delta = delta.view(-1, 1, 1, 1)
                    zero_point = zero_point.view(-1, 1, 1, 1)
                elif len(x.shape) == 2:
                    delta = delta.view(-1, 1)
                    zero_point = zero_point.view(-1, 1)
                elif len(x.shape) == 3:
                    delta = delta.view(1, 1, -1)
                    zero_point = zero_point.view(1, 1, -1)
                else:
                    raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e+10
            if self.is_act:
                search_range = [0.999, 0.9999, 0.99999]
            else:
                search_range = [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999]
            for pct in search_range:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)   
                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


class LogSqrt2Quantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, is_act: bool = False):
        super(LogSqrt2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.register_buffer('inited', torch.zeros(1))
        self.channel_wise = channel_wise
        self.is_act = is_act

    def forward(self, x: torch.Tensor):
        if self.inited == 0:
            delta = self.init_quantization_scale(x)
            self.delta = Parameter(delta).contiguous()

            self.inited.fill_(1)

        x = x + self.delta
        x = self.logk(x, self.base)
        delta = (self.maxv - self.minv) / (self.n_levels)
        x_int = round_ste((x - self.minv) / delta)
        x_int = torch.clamp(x_int, 0, self.n_levels)
        x_float_q = self.base ** round_ste((x_int * delta + self.minv))
        x_dequant = x_float_q - self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor):
        x_clone = x.clone().detach()
        cur_bias = -10
        self.base = 2
        best_score = 1e+10
        for bias in [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1.0]:
            x_q, maxv, minv = self.quantize(x_clone, bias)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            if score < best_score:
                best_score = score
                cur_bias = bias
                self.maxv = maxv
                self.minv = minv
        return torch.tensor(cur_bias)

    def logk(self, x, k):
        natural_log = torch.log(x)
        log_k = natural_log / torch.log(torch.tensor(k, dtype=x.dtype))
        return log_k

    def quantize(self, x, bias):

        x = x + bias
        x = self.logk(x, self.base)
        maxv = torch.max(x)
        minv = torch.min(x)

        delta = (maxv - minv) / (self.n_levels - 1)
        x_int = round_ste((x - minv) / delta)
        x_int = torch.clamp(x_int, 0, self.n_levels - 1)
        # x_int = torch.clamp(x_int, 0, self.n_levels)
        x_float_q = self.base ** torch.round((x_int * delta + minv))
        x_float_q = x_float_q - bias

        return x_float_q, maxv, minv