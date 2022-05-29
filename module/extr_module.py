import torch
import torch.nn as nn
from common.utils import mlp, weight_init, EnsembleLinear
import numpy as np


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}
OUT_DIM_108 = {4: 47}


class PixelExtractor(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, num_fc=1, *args):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.is_fc = (num_fc == 1)

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        if obs_shape[-1] == 64:
            out_dim = OUT_DIM_64[num_layers]
        elif obs_shape[-1] == 108:
            assert num_layers in OUT_DIM_108
            out_dim = OUT_DIM_108[num_layers]
        else:
            out_dim = OUT_DIM[num_layers]

        self.repr_dim = num_filters * out_dim * out_dim
        if self.is_fc:
            self.fc = nn.Linear(self.repr_dim, self.feature_dim)
            self.ln = nn.LayerNorm(self.feature_dim)
        self.infos = dict()
        self.apply(weight_init)

    def forward_conv(self, obs):
        conv = torch.relu(self.convs[0](obs))
        self.infos['conv1'] = conv
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.infos['conv%s' % (i + 1)] = conv
        return conv.view(obs.size(0), -1)

    def forward_fc(self, obs, tanh=False):
        if not self.is_fc:
            return obs
        z = self.fc(obs)
        self.infos['fc'] = z
        z = self.ln(z)
        self.infos['ln'] = z
        if tanh:
            z = torch.tanh(z)
            self.infos['tanh'] = z
        return z

    def forward(self, obs, conv_detach=False, tanh=False):
        assert obs.max() <= 1.0, print(obs.max())
        self.infos['obs'] = obs
        h = self.forward_conv(obs)
        if conv_detach:
            h = h.detach()
        h = self.forward_fc(h, tanh)
        return h

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers and hidden layers"""
        # tie_weights(src=source.fc, trg=self.fc)
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq, loss_type=None, history=True, name=None):
        if not log_freq or step % log_freq != 0:
            return

        name = 'train_encoder' if name is None else name
        if history:
            for k, v in self.infos.items():
                L.log_histogram('%s/%s_hist' % (name, k), v, step)
                if len(v.shape) > 2:
                    L.log_image('%s/%s_img' % (name, k), v[0], step)

        loss_type = f'-{loss_type}' if loss_type is not None else ''
        for i in range(self.num_layers):
            L.log_param('%s/conv%s%s' % (name, i + 1, loss_type), self.convs[i], step)
        if self.is_fc:
            L.log_param('%s/fc%s' % (name, loss_type), self.fc, step)
            L.log_param('%s/ln%s' % (name, loss_type), self.ln, step)


class PixelExtractor_v1(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, num_fc=1, *args):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.is_fc = (num_fc == 1)

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.BatchNorm2d(num_filters))
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.convs.append(nn.BatchNorm2d(num_filters))

        if obs_shape[-1] == 64:
            out_dim = OUT_DIM_64[num_layers]
        elif obs_shape[-1] == 108:
            assert num_layers in OUT_DIM_108
            out_dim = OUT_DIM_108[num_layers]
        else:
            out_dim = OUT_DIM[num_layers]

        self.repr_dim = num_filters * out_dim * out_dim
        if self.is_fc:
            self.fc = nn.Linear(self.repr_dim, self.feature_dim)
        self.infos = dict()
        self.apply(weight_init)

    def forward_conv(self, obs):
        conv = obs
        for i in range(self.num_layers):
            conv = self.convs[i * 2](conv)
            conv = self.convs[i * 2 + 1](conv)
            conv = torch.relu(conv)
            self.infos['conv%s' % (i + 1)] = conv
        return conv.view(obs.size(0), -1)

    def forward_fc(self, obs, tanh=False):
        if not self.is_fc:
            return obs
        z = self.fc(obs)
        self.infos['fc'] = z
        if tanh:
            z = torch.tanh(z)
            self.infos['tanh'] = z
        return z

    def forward(self, obs, conv_detach=False, tanh=False):
        assert not (obs > 1.0).any()
        self.infos['obs'] = obs
        h = self.forward_conv(obs)
        if conv_detach:
            h = h.detach()
        h = self.forward_fc(h, tanh)
        return h

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers and hidden layers"""
        # tie_weights(src=source.fc, trg=self.fc)
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq, loss_type=None, history=True, name=None):
        if not log_freq or step % log_freq != 0:
            return

        name = 'train_encoder' if name is None else name
        if history:
            for k, v in self.infos.items():
                L.log_histogram('%s/%s_hist' % (name, k), v, step)
                if len(v.shape) > 2:
                    L.log_image('%s/%s_img' % (name, k), v[0], step)

        loss_type = f'-{loss_type}' if loss_type is not None else ''
        for i in range(self.num_layers):
            L.log_param('%s/conv%s%s' % (name, i + 1, loss_type), self.convs[i*2], step)
        if self.is_fc:
            L.log_param('%s/fc%s' % (name, loss_type), self.fc, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, num_layers, num_filters, *args):
        super().__init__()

        assert len(obs_shape) == 1

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass



_AVAILABLE_ENCODERS = {'pixel': PixelExtractor, 'identity': IdentityEncoder}


def make_extr(obs_shape, extr_latent_dim, num_layers, num_filters, num_fc, **args):
    return PixelExtractor(obs_shape, extr_latent_dim, num_layers, num_filters, num_fc)
