# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, List, Optional, Tuple
from torch import Tensor

from apex import amp
import torch
import torch.nn as nn
from parts.features import FeatureFactory
import random

from helpers import construct_padding

from functools import partial

jasper_activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}

def init_weights(m, mode='xavier_uniform'):
    if isinstance(m, MaskedConv1d):
        init_weights(m.conv, mode)
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def compute_new_kernel_size(kernel_size, kernel_width):
    new_kernel_size = max(int(kernel_size * kernel_width), 1)
    # If kernel is even shape, round up to make it odd
    if new_kernel_size % 2 == 0:
        new_kernel_size += 1
    return new_kernel_size

def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        return (dilation * kernel_size) // 2 - 1
    return kernel_size // 2

class AudioPreprocessing(nn.Module):
    """GPU accelerated audio preprocessing
    """
    __constants__ = ["optim_level"]
    def __init__(self, **kwargs):
        nn.Module.__init__(self)    # For PyTorch API
        self.optim_level = kwargs.get('optimization_level', 0)
        self.featurizer = FeatureFactory.from_config(kwargs)
        self.transpose_out = kwargs.get("transpose_out", False)

    @torch.no_grad()
    def forward(self, input_signal, length):
        processed_signal = self.featurizer(input_signal, length)
        processed_length = self.featurizer.get_seq_len(length)    
        if self.transpose_out:
            processed_signal.transpose_(2,1)
            return processed_signal, processed_length
        else:
            return processed_signal, processed_length
                
class SpectrogramAugmentation(nn.Module):
    """Spectrogram augmentation
    """
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.spec_cutout_regions = SpecCutoutRegions(kwargs)
        self.spec_augment = SpecAugment(kwargs)

    @torch.no_grad()
    def forward(self, input_spec):
        augmented_spec = self.spec_cutout_regions(input_spec)
        augmented_spec = self.spec_augment(augmented_spec)
        return augmented_spec

class SpecAugment(nn.Module):
    """Spec augment. refer to https://arxiv.org/abs/1904.08779
    """
    def __init__(self, cfg):
        super(SpecAugment, self).__init__()
        self.cutout_x_regions = cfg.get('cutout_x_regions', 0)
        self.cutout_y_regions = cfg.get('cutout_y_regions', 0)

        self.cutout_x_width = cfg.get('cutout_x_width', 10)
        self.cutout_y_width = cfg.get('cutout_y_width', 10)

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        mask = torch.zeros(x.shape).byte()
        for idx in range(sh[0]):
            for _ in range(self.cutout_x_regions):
                cutout_x_left = int(random.uniform(0, sh[1] - self.cutout_x_width))

                mask[idx, cutout_x_left:cutout_x_left + self.cutout_x_width, :] = 1

            for _ in range(self.cutout_y_regions):
                cutout_y_left = int(random.uniform(0, sh[2] - self.cutout_y_width))

                mask[idx, :, cutout_y_left:cutout_y_left + self.cutout_y_width] = 1

        x = x.masked_fill(mask.to(device=x.device), 0)

        return x

class SpecCutoutRegions(nn.Module):
    """Cutout. refer to https://arxiv.org/pdf/1708.04552.pdf
    """
    def __init__(self, cfg):
        super(SpecCutoutRegions, self).__init__()

        self.cutout_rect_regions = cfg.get('cutout_rect_regions', 0)
        self.cutout_rect_time = cfg.get('cutout_rect_time', 5)
        self.cutout_rect_freq = cfg.get('cutout_rect_freq', 20)

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        mask = torch.zeros(x.shape, dtype=torch.uint8)

        for idx in range(sh[0]):
            for i in range(self.cutout_rect_regions):
                cutout_rect_x = int(random.uniform(
                        0, sh[1] - self.cutout_rect_freq))
                cutout_rect_y = int(random.uniform(
                        0, sh[2] - self.cutout_rect_time))

                mask[idx, cutout_rect_x:cutout_rect_x + self.cutout_rect_freq,
                         cutout_rect_y:cutout_rect_y + self.cutout_rect_time] = 1

        x = x.masked_fill(mask.to(device=x.device), 0)

        return x

class AdaptLayerActivations(nn.Module):
    def __init__(self,feat_in, use_bilstm=False, use_gating=True):
        super(AdaptLayerActivations,self).__init__()
        self.use_bilstm = use_bilstm
        self.use_gating = use_gating
        if self.use_bilstm:
            self.bilstm = nn.LSTM(input_size=feat_in,
                                hidden_size=feat_in,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.10,
                                bidirectional=True)
            self.fc_layer_1 = nn.Linear(2*feat_in,feat_in)
            if self.use_gating:
                self.fc_layer_2 = nn.Linear(2*feat_in,feat_in)
        else:
            self.fc_layer_1 = nn.Linear(feat_in,2*feat_in)
            self.fc_layer_2 = nn.Linear(2*feat_in,feat_in)
            ffn_layers = [self.fc_layer_1, nn.ReLU(), nn.Dropout(0.1),  self.fc_layer_2 ]
            self.ffn_layers = nn.Sequential(*ffn_layers)
            if self.use_gating:
                self.fc_layer_3 = nn.Linear(feat_in,feat_in//4)
                self.fc_layer_4 = nn.Linear(feat_in//4, feat_in)
                gate_layers = [self.fc_layer_3, nn.ReLU(), nn.Dropout(0.1), self.fc_layer_4]
                self.gate_layers = nn.Sequential(*gate_layers)


    def forward(self, x):
        signal, signal_length = x
        residual = []
        if len(signal) > 1:
            residual = signal[:-1]
        signal = signal[-1]
        signal = signal.transpose(1,2)
        #print("audio_signal_shape_in_lstm: ",audio_signal.shape)
        if self.use_bilstm:
            pack_padded_signal = nn.utils.rnn.pack_padded_sequence(signal,
                                                                 signal_length, 
                                                                 batch_first=True, 
                                                                 enforce_sorted=False)
            bilstm_out,_ = self.bilstm(pack_padded_signal)
            bilstm_out, tmp_signal_length = nn.utils.rnn.pad_packed_sequence(bilstm_out, 
                                                        batch_first=True, 
                                                        padding_value=0.0,
                                                        total_length=signal.shape[1])

            #print(tmp_signal_length,'\n',signal_length,'\n\n')
            if self.use_gating:
                mixing_probs = torch.sigmoid(self.fc_layer_2(bilstm_out))
                #print("mixing_probs_shape: ",mixing_probs.shape)
                revised_signal = (1-mixing_probs)*signal + mixing_probs*(self.fc_layer_1(bilstm_out))
            else:
                revised_signal = self.fc_layer_1(bilstm_out)
        else:
            revised_signal = self.ffn_layers(signal)
            if self.use_gating:
                mixing_probs = torch.sigmoid(self.gate_layers(signal))
                revised_signal = mixing_probs*signal + (1-mixing_probs)*revised_signal

        return residual+[revised_signal.transpose(1,2)],signal_length 

class JasperEncoder(nn.Module):
    __constants__ = ["use_conv_mask"]    
    """Jasper encoder
    """
    def __init__(self, **kwargs):
        cfg = {}
        for key, value in kwargs.items():
            cfg[key] = value

        nn.Module.__init__(self)
        self._cfg = cfg

        activation = jasper_activations[cfg['encoder']['activation']]()
        self.use_conv_mask = cfg['encoder'].get('convmask', False)
        feat_in = cfg['input']['features'] * cfg['input'].get('frame_splicing', 1)
        init_mode = cfg.get('init_mode', 'xavier_uniform')
        turn_bn_eval = cfg.get('turn_bn_eval',True)
        print('turn_bn_eval: ',turn_bn_eval)

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for layer_number,lcfg in enumerate(cfg['jasper']):
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            heads = lcfg.get('heads', -1)
            residual_mode = lcfg.get('residual_mode', 'add')
            se = lcfg.get('se', False)
            se_reduction_ratio = lcfg.get('se_reduction_ratio', 8)
            se_context_window = lcfg.get('se_context_window', -1)
            se_interpolation_mode = lcfg.get('se_interpolation_mode', 'nearest')
            kernel_size_factor = lcfg.get('kernel_size_factor', 1.0)
            stride_last = lcfg.get('stride_last', False)
            normalization_mode = lcfg.get('normalization_mode','batch')
            norm_groups = lcfg.get('norm_groups',-1)
            encoder_layers.append(
                JasperBlock(
                    feat_in,
                    lcfg['filters'],
                    repeat=lcfg['repeat'],
                    kernel_size=lcfg['kernel'],
                    stride=lcfg['stride'],
                    dilation=lcfg['dilation'],
                    dropout=lcfg['dropout'],
                    residual=lcfg['residual'],
                    groups=groups,
                    separable=separable,
                    heads=heads,
                    residual_mode=residual_mode,
                    normalization=normalization_mode,
                    norm_groups=norm_groups,
                    activation=activation,
                    residual_panes=dense_res,
                    conv_mask=self.use_conv_mask,
                    se=se,
                    se_reduction_ratio=se_reduction_ratio,
                    se_context_window=se_context_window,
                    se_interpolation_mode=se_interpolation_mode,
                    kernel_size_factor=kernel_size_factor,
                    stride_last=stride_last,
                    turn_bn_eval=turn_bn_eval
                )
            )

            feat_in = lcfg['filters']               
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if self.use_conv_mask:
            audio_signal, length = x
            return self.encoder(([audio_signal], length))
        else:
            print("Error, not yet handeled conv_mask_false for intervining layer")
            exit(1)
            return self.encoder([x])

class JasperDecoderForCTC(nn.Module):
    """Jasper decoder
    """
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self._feat_in = kwargs.get("feat_in")
        self._num_classes = kwargs.get("num_classes")
        init_mode = kwargs.get('init_mode', 'xavier_uniform')

        self.decoder_layers = nn.Sequential(
            nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True),)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, encoder_output):
        out = self.decoder_layers(encoder_output[-1]).transpose(1, 2)
        return nn.functional.log_softmax(out, dim=2)

class JasperEncoderDecoder(nn.Module):
    """Contains jasper encoder and decoder
    """
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.transpose_in=kwargs.get("transpose_in", False)
        self.jasper_encoder = JasperEncoder(**kwargs.get("jasper_model_definition"))
        self.jasper_decoder = JasperDecoderForCTC(feat_in=kwargs.get("feat_in"),
                                                  num_classes=kwargs.get("num_classes"))
        
    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    
    def forward(self, x):
        if self.jasper_encoder.use_conv_mask:
            t_encoded_t, t_encoded_len_t = self.jasper_encoder(x)
        else:
            if self.transpose_in:
                x = x.transpose(1, 2)   
            t_encoded_t = self.jasper_encoder(x)
            
        out = self.jasper_decoder(t_encoded_t)
        if self.jasper_encoder.use_conv_mask:
            return out, t_encoded_len_t
        else:
            return out

    def infer(self, x):
        if self.jasper_encoder.use_conv_mask:
            return self.forward(x)
        else:
            ret = self.forward(x[0])
            return ret, len(ret)
        
    
class Jasper(JasperEncoderDecoder):
    """Contains data preprocessing, spectrogram augmentation, jasper encoder and decoder
    """
    def __init__(self, **kwargs):
        JasperEncoderDecoder.__init__(self, **kwargs)
        feature_config = kwargs.get("feature_config")
        if self.transpose_in:
            feature_config["transpose"] = True
        self.audio_preprocessor = AudioPreprocessing(**feature_config)
        self.data_spectr_augmentation = SpectrogramAugmentation(**feature_config)


class MaskedConv1d(nn.Module):
    __constants__ = ["use_conv_mask", "real_out_channels", "heads"]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        heads=-1,
        bias=False,
        use_mask=True,
    ):
        super(MaskedConv1d, self).__init__()

        if not (heads == -1 or groups == in_channels):
            raise ValueError("Only use heads for depthwise convolutions")

        self.real_out_channels = out_channels
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.use_mask = use_mask
        self.heads = heads

    def get_seq_len(self, lens):
        return (
            lens + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1
        ) / self.conv.stride[0] + 1

    def forward(self, x, lens):
        if self.use_mask:
            lens = lens.to(dtype=torch.long)
            max_len = x.size(2)
            mask = torch.arange(max_len).to(lens.device).expand(len(lens), max_len) >= lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
            # del mask
            lens = self.get_seq_len(lens)

        sh = x.shape
        if self.heads != -1:
            x = x.view(-1, self.heads, sh[-1])

        out = self.conv(x)

        if self.heads != -1:
            out = out.view(sh[0], self.real_out_channels, -1)

        return out, lens

class GroupShuffle(nn.Module):
    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()

        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape

        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])

        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])

        return x

class SqueezeExcite(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        context_window: int = -1,
        interpolation_mode: str = 'nearest',
        activation: Optional[Callable] = None,
    ):
        """
        Squeeze-and-Excitation sub-module.

        Args:
            channels: Input number of channels.
            reduction_ratio: Reduction ratio for "squeeze" layer.
            context_window: Integer number of timesteps that the context
                should be computed over, using stride 1 average pooling.
                If value < 1, then global context is computed.
            interpolation_mode: Interpolation mode of timestep dimension.
                Used only if context window is > 1.
                The modes available for resizing are: `nearest`, `linear` (3D-only),
                `bilinear`, `area`
            activation: Intermediate activation function used. Must be a
                callable activation function.
        """
        super(SqueezeExcite, self).__init__()
        self.context_window = int(context_window)
        self.interpolation_mode = interpolation_mode

        if self.context_window <= 0:
            self.pool = nn.AdaptiveAvgPool1d(1)  # context window = T
        else:
            self.pool = nn.AvgPool1d(self.context_window, stride=1)

        if activation is None:
            activation = nn.ReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            activation,
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )

    def forward(self, x):
        batch, channels, timesteps = x.size()
        y = self.pool(x)  # [B, C, T - context_window + 1]
        y = y.transpose(1, 2)  # [B, T - context_window + 1, C]
        y = self.fc(y)  # [B, T - context_window + 1, C]
        y = y.transpose(1, 2)  # [B, C, T - context_window + 1]

        if self.context_window > 0:
            y = torch.nn.functional.interpolate(y, size=timesteps, mode=self.interpolation_mode)

        y = torch.sigmoid(y)

        return x * y

class JasperBlock(nn.Module):
    __constants__ = ["conv_mask", "separable", "residual_mode", "res", "mconv"]

    def __init__(
        self,
        inplanes,
        planes,
        repeat=3,
        kernel_size=11,
        kernel_size_factor=1,
        stride=1,
        dilation=1,
        padding='same',
        dropout=0.2,
        activation=None,
        residual=True,
        groups=1,
        separable=False,
        heads=-1,
        normalization="batch",
        norm_groups=1,
        residual_mode='add',
        residual_panes=[],
        conv_mask=False,
        se=False,
        se_reduction_ratio=16,
        se_context_window=None,
        se_interpolation_mode='nearest',
        stride_last=False,
        turn_bn_eval=True,
    ):
        super(JasperBlock, self).__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        kernel_size_factor = float(kernel_size_factor)
        if type(kernel_size) in (list, tuple):
            kernel_size = [compute_new_kernel_size(k, kernel_size_factor) for k in kernel_size]
        else:
            kernel_size = compute_new_kernel_size(kernel_size, kernel_size_factor)

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.conv_mask = conv_mask
        self.separable = separable
        self.residual_mode = residual_mode
        self.se = se

        inplanes_loop = inplanes
        conv = nn.ModuleList()

        for _ in range(repeat - 1):
            # Stride last means only the last convolution in block will have stride
            if stride_last:
                stride_val = [1]
            else:
                stride_val = stride

            conv.extend(
                self._get_conv_bn_layer(
                    inplanes_loop,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride_val,
                    dilation=dilation,
                    padding=padding_val,
                    groups=groups,
                    heads=heads,
                    separable=separable,
                    normalization=normalization,
                    norm_groups=norm_groups,
                    turn_bn_eval=turn_bn_eval
                )
            )

            conv.extend(self._get_act_dropout_layer(drop_prob=dropout, activation=activation))

            inplanes_loop = planes

        conv.extend(
            self._get_conv_bn_layer(
                inplanes_loop,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_val,
                groups=groups,
                heads=heads,
                separable=separable,
                normalization=normalization,
                norm_groups=norm_groups,
                turn_bn_eval=turn_bn_eval
            )
        )

        if se:
            conv.append(
                SqueezeExcite(
                    planes,
                    reduction_ratio=se_reduction_ratio,
                    context_window=se_context_window,
                    interpolation_mode=se_interpolation_mode,
                    activation=activation,
                )
            )

        self.mconv = conv

        res_panes = residual_panes.copy()
        self.dense_residual = residual

        if residual:
            res_list = nn.ModuleList()

            if residual_mode == 'stride_add':
                stride_val = stride
            else:
                stride_val = [1]

            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                res = nn.ModuleList(
                    self._get_conv_bn_layer(
                        ip,
                        planes,
                        kernel_size=1,
                        normalization=normalization,
                        norm_groups=norm_groups,
                        stride=stride_val,
                        turn_bn_eval=turn_bn_eval
                    )
                )

                res_list.append(res)

            self.res = res_list
        else:
            self.res = None

        self.mout = nn.Sequential(*self._get_act_dropout_layer(drop_prob=dropout, activation=activation))

    def _get_conv(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
    ):
        use_mask = self.conv_mask
        if use_mask:
            return MaskedConv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups,
                heads=heads,
                use_mask=use_mask,
            )
        else:
            return nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups,
            )

    def _get_conv_bn_layer(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
        normalization="batch",
        norm_groups=1,
        turn_bn_eval=True,
    ):
        if norm_groups == -1:
            norm_groups = out_channels

        if separable:
            layers = [
                self._get_conv(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=in_channels,
                    heads=heads,
                ),
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    padding=0,
                    bias=bias,
                    groups=groups,
                ),
            ]
        else:
            layers = [
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                )
            ]

        if normalization == "group":
            layers.append(nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels))
        elif normalization == "instance":
            layers.append(nn.GroupNorm(num_groups=out_channels, num_channels=out_channels))
        elif normalization == "layer":
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        elif normalization == "batch":
            layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.0 if turn_bn_eval else 0.1))
        else:
            raise ValueError(
                f"Normalization method ({normalization}) does not match" f" one of [batch, layer, group, instance]."
            )

        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [activation, nn.Dropout(p=drop_prob)]
        return layers

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_: Tuple[List[Tensor], Optional[Tensor]]):
        # type: (Tuple[List[Tensor], Optional[Tensor]]) -> Tuple[List[Tensor], Optional[Tensor]] # nopep8
        lens_orig = None
        xs = input_[0]
        if len(input_) == 2:
            xs, lens_orig = input_

        # compute forward convolutions
        out = xs[-1]

        lens = lens_orig
        for i, l in enumerate(self.mconv):
            # if we're doing masked convolutions, we need to pass in and
            # possibly update the sequence lengths
            # if (i % 4) == 0 and self.conv_mask:
            if isinstance(l, MaskedConv1d):
                out, lens = l(out, lens)
            else:
                out = l(out)

        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if isinstance(res_layer, MaskedConv1d):
                        res_out, _ = res_layer(res_out, lens_orig)
                    else:
                        res_out = res_layer(res_out)

                if self.residual_mode == 'add' or self.residual_mode == 'stride_add':
                    out = out + res_out
                else:
                    out = torch.max(out, res_out)

        # compute the output
        out = self.mout(out)
        if self.res is not None and self.dense_residual:
            return xs + [out], lens

        return [out], lens

class GreedyCTCDecoder(nn.Module):
    """ Greedy CTC Decoder
    """
    def __init__(self, **kwargs):
        nn.Module.__init__(self)    # For PyTorch API
    @torch.no_grad()
    def forward(self, log_probs):
            argmx = log_probs.argmax(dim=-1, keepdim=False).int()
            return argmx

class CTCLossNM:
    """ CTC loss
    """
    def __init__(self, **kwargs):
        self._blank = kwargs['num_classes'] - 1
        zero_infinity = kwargs['zero_infinity']
        self._criterion = nn.CTCLoss(blank=self._blank, reduction='none', zero_infinity=zero_infinity)
            

    def __call__(self, log_probs, targets, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        targets = targets.long()
        loss = self._criterion(log_probs.transpose(1, 0), targets, input_length,
                                                         target_length)
        return torch.mean(loss)