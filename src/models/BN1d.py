import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
from scipy.signal import hilbert
from typing import List
from einops import rearrange

import os
import sys
import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)
from models.patchTST import PatchTST


def any_requires_grad(module: nn.Module) -> bool:
    '''
    Check if any parameter in module requires grad.
    '''
    any_grad = False
    for p in module.parameters():
        if p.requires_grad:
            any_grad = True
    return any_grad

class DynamicTanh(nn.Module):
    '''
    Transformers without Normalization.
    https://arxiv.org/pdf/2503.10622
    '''
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        # Assuming we do not have timm.layers.LayerNorm2d
        module_output = DynamicTanh(module.normalized_shape, True)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output


class BlaschkeLayer1d(nn.Module):
    '''
    Blaschke Layer for 1D signals.
    The input dimension is assumed to be [B, L, C].

    In Blaschke decomposition, a function F(x) can be approximated by
    F(x) = s_1 * B_1 + s_2 * B_1 * B_2 + s_3 * B_1 * B_2 * B_3 + ...
        where s_i is a scaling factor (scalar),
        and B_i is the i-th Blaschke factor.
        B(x) = exp(i \theta(x)),
        \theta(x) = \sum_{j >= 0} \sigma ((x - \alpha_j) / \beta_j),
        \sigma(x) = \arctan(x) + \pi / 2

    Attributes:
    -----------
        param_net (torch.nn.Module): Neural network to extract Blaschke parameters.
        num_blaschke (int): Number of Blaschke components.
        eps (float): Small number for numerical stability.
        device (torch.device): Torch device.
    '''

    def __init__(self,
                 param_net: torch.nn.Module,
                 num_blaschke: int = 1,
                 eps: float = 1e-11,
                 device: str = 'cpu') -> None:

        super().__init__()

        self.param_net = param_net
        self.num_blaschke = num_blaschke
        assert self.num_blaschke == 1, 'Current implementation only supports 1 Blaschke factor.'

        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)  # To facilitate checkpointing.
        self.eps = eps

        self.to(device)

    def __repr__(self):
        return (f"BlaschkeLayer1d("
                f"num_blaschke={self.num_blaschke})")

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Custom activation function used in the BlaschkeLayer1d.
        '''
        output = torch.arctan(x) + torch.pi / 2
        return output

    def estimate_blaschke_parameters(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Estimate the Blaschke parameters.
        Using gradient checkpointing to trade time for space.
        Otherwise it is easy to OOM when using many layers.

        Args:
        -----
            x : 3D torch.float
                inputs, shape [B, C, L] (batch size, channel, signal length)
        '''

        B, C, L = x.shape                                              # [batch_size, num_channels, seq_len]
        assert len(x.shape) == 3
        x_real, x_imag = torch.real(x), torch.imag(x)                  # [batch_size, num_channels, seq_len]
        x = torch.stack((x_real, x_imag), dim=2).float()               # [batch_size, num_channels, 2, seq_len]
        x = rearrange(x, 'b c r l -> (b c) r l')                       # [batch_size * num_channels, 2, seq_len]
        assert len(x.shape) == 3

        if any_requires_grad(self.param_net):
            params = checkpoint(self.param_net, x, self.dummy_tensor)  # [batch_size * num_channels, 4]
        else:
            params = self.param_net(x, self.dummy_tensor)              # [batch_size * num_channels, 4]
        params = rearrange(params, '(b c) p -> b c p', b=B, c=C)       # [batch_size, num_channels, 4]

        alpha, log_beta, scale_real, scale_imag = params[..., 0], params[..., 1], params[..., 2], params[..., 3]
        self.alpha = alpha
        self.beta = torch.exp(log_beta + self.eps)
        self.scale = scale_real + 1j * scale_imag
        return

    def compute_blaschke_factor(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute the $B(x)$ for the input x.

        Args:
        -----
            x : 3D torch.float
                inputs, shape [B, C, L] (batch size, channel, signal length)

        Returns:
        --------
            y : 3D torch.float
                outputs, shape [B, C, L] (batch size, channel, signal length)
        '''
        frac = (x - self.alpha.unsqueeze(-1)) / self.beta.unsqueeze(-1)
        theta_x = self.activation(frac)
        # theta_x = torch.sigmoid(frac)
        blaschke_factor = torch.exp(1j * theta_x)
        return blaschke_factor

    def to(self, device: str):
        super().to(device)
        self.device = device
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
        -----
            x : 3D torch.float
                inputs, shape [B, C, L] (batch size, channel, signal length)

        Returns:
        --------
            y : 3D torch.float
                outputs, shape [B, C, L] (batch size, channel, signal length)

        Example
        -------
        >>> model = BlaschkeLayer1d(signal_len=100, signal_channel=10, num_blaschke=1)
        >>> x = torch.normal(0, 1, size=(32, 10, 100))
        >>> y = model(x)
        '''

        # [B, C, L]
        assert len(x.shape) == 3

        # Compute the Blaschke product $B(x)$. [B, C, L]
        self.estimate_blaschke_parameters(x)
        blaschke_factor = self.compute_blaschke_factor(x)
        return blaschke_factor

class Ignore2ndArg(nn.Module):
    '''
    This is a module wrapper that essentially ignores the 2nd input argument.
    The reason for using it is that sometimes we need to use checkpointing
    to trade time for space, but when we use the checkpointing on the first
    module of the neural network, it stupidly breaks the gradient, because
    none of the inputs have `requires_grad=True`. As a workaround, we can
    pass in a dummy vector that has `requires_grad=True` but ignore it
    during computation.
    '''
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.module(x)
        return x

class BlaschkeNetwork1d(nn.Module):
    '''
    Blaschke Network for 1D signals.

    A neural network composed of multiple BlaschkeLayer1d instances that leverages the Blaschke product
    for complex function representation. This network is designed for applications requiring
    iterative, layer-based transformations, such as complex function approximation or signal
    reconstruction.

    Attributes:
    -----------
        signal_len (int): Length of the signal.
        num_channels (int): Number of signal channels.
        patch_size (int): Patch size for the signal. Used for patch embedding of signal.
        layers (int): Number of Blaschke Network layers.
        detach_by_iter (bool): If true, penalize approximation of each iteration independently.
        out_classes (int): Number of output classes in the classification problem.
        num_blaschke_list (List[int]): Number of Blaschke factors in each layer.

        param_net_dim (int): Param Net (Transformer) transformer dimension.
        param_net_depth (int): Param Net (Transformer) numbet of layers.
        param_net_heads (int): Param Net (Transformer) number of heads.
        param_net_mlp_dim (int): Param Net (Transformer) MLP dimension.

        eps (float): Small number for numerical stability.
        device (torch.device): Torch device.
        seed (int): Random seed.
    '''

    def __init__(self,
                 signal_len: int,
                 num_channels: int = 1,
                 patch_size: int = 1,
                 layers: int = 1,
                 detach_by_iter: bool = False,
                 out_classes: int = 10,
                 num_blaschke_list: List[int] = None,
                 param_net_dim: int = 64,
                 param_net_depth: int = 2,
                 param_net_heads: int = 4,
                 param_net_mlp_dim: int = 256,
                 eps: float = 1e-6,
                 device: str = 'cpu',
                 seed: int = 1) -> None:

        super().__init__()
        torch.manual_seed(seed)

        self.signal_len = signal_len
        self.num_channels = num_channels
        self.out_classes = out_classes
        self.layers = layers
        self.detach_by_iter = detach_by_iter
        self.num_blaschke_list = num_blaschke_list

        if self.num_blaschke_list is None:
            # Determine Blaschke parameters if not provided
            self.num_blaschke_list = [1 for _ in range(self.layers)]

        # Initialize learnable parameters for computing Blaschke product.
        # 4 output neurons, respectively for:
        #   alpha, log_beta, scale_real, scale_imaginary
        num_blaschke_params = 4

        param_net = PatchTST(
            num_channels=2,         # (real, imaginary)
            num_classes=num_blaschke_params,
            patch_size=patch_size,
            num_patch=signal_len // patch_size,
            n_layers=param_net_depth,
            n_heads=param_net_heads,
            d_model=param_net_dim,
            shared_embedding=True,
            d_ff=param_net_mlp_dim,
            dropout=0.2,
            head_dropout=0.2,
            act='gelu',
            norm='layernorm',
            res_attention=False,
        )
        self.param_net = Ignore2ndArg(   # `Ignore2ndArg` is a wrapper to facilitate checkpointing.
            convert_ln_to_dyt(param_net) # Replace layernorm with dynamic Tanh.
        )

        # Initialize the BNLayers
        self.encoder = nn.ModuleList([])

        for layer_idx in range(self.layers):
            self.encoder.append(
                BlaschkeLayer1d(num_blaschke=self.num_blaschke_list[layer_idx],
                                param_net=self.param_net,
                                eps=eps,
                                device=device)
            )

        num_features = sum(self.num_blaschke_list) * num_channels * num_blaschke_params

        # In linear probing, only this is updated.
        # This is technically not a linear probing, but rather a two-stage training.
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, self.out_classes),
        )

        # Initialize weights
        self.initialize_weights()
        self.to(device)

    def initialize_weights(self) -> None:
        '''Initialize the weights of the BlaschkeLayer1d.'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze(self) -> None:
        '''
        Freeze all parameters.
        '''
        for p in self.parameters():
            p.requires_grad = False
        return

    def unfreeze_classifier(self) -> None:
        '''
        Unfreeze the parameter of the final linear classifier.
        '''
        for p in self.classifier.parameters():
            p.requires_grad = True
        return

    def complexify(self, x: torch.Tensor, carrier_freq: float = 0) -> torch.Tensor:
        '''
        Complexify the signal for Blaschke decomposition.
        '''
        signal = x.cpu().detach().numpy()
        signal = rearrange(signal, 'b c l -> (b c) l')  # b: batch size, c: number of channels, l: signal length.
        # Hilbert transform after removing zero-order drift.
        signal = hilbert(signal - np.mean(signal, axis=1, keepdims=True))
        # Frequency shifting by carrier frequency.
        time_indices = np.arange(x.shape[-1])
        signal = signal * np.exp(1j * 2 * np.pi * carrier_freq * time_indices)
        # Mitigate boundary effects. This is a common approach when performing Fourier analyses, spectral filtering, etc.
        signal_boundary_smoothed = np.concatenate((signal, np.fliplr(np.conj(signal))), axis=1)
        mask_nonnegative_freq = np.ones_like(signal_boundary_smoothed)
        mask_nonnegative_freq[:, mask_nonnegative_freq.shape[1] // 2:] = 0
        signal = np.fft.ifft(np.fft.fft(signal_boundary_smoothed) * mask_nonnegative_freq)
        signal = signal[:, :signal.shape[1] // 2]
        signal = rearrange(signal, '(b c) l -> b c l', b=x.shape[0], c=x.shape[1])

        # Cast to torch Tensor.
        signal = torch.from_numpy(signal).to(x.device)
        return signal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
        -----
            x : 3D torch.float
                inputs, shape [B, C, L] (batch size, channel, signal length)

        Returns:
        --------
            y_pred : 2D torch.float
                outputs, shape [B, CLS] (batch size, number of classes)
            residual_signals_sqsum : torch.float
                squared norm of signal approximation error
        '''

        # Input should be a [B, C, L] signal.
        assert len(x.shape) == 3

        # Complexify the signal using hilbert transform.
        signal_complex = self.complexify(x)

        blaschke_factors = []
        residual_signal, residual_sqnorm = signal_complex, None
        blaschke_coeffs = None

        for layer in self.encoder:
            blaschke_factors.append(layer(residual_signal))

            # Blaschke product is the cumulative product of Blaschke factors
            # B_1 * B_2 * ... * B_n.
            blaschke_product = 1
            for blaschke_factor in blaschke_factors:
                blaschke_product = blaschke_product * blaschke_factor

            # The Blaschke approximation is given by
            # s_n * B_1 * B_2 * ... * B_n.
            curr_signal_approx = layer.scale.unsqueeze(-1) * blaschke_product

            # F_{n+1} = F_n - s_n * B_1 * ... * B_n.
            residual_signal = residual_signal - curr_signal_approx

            # This helps sanity checking the residual norms at each iteration.
            if residual_sqnorm is None:
                residual_sqnorm = torch.real(residual_signal).pow(2).mean(dim=(1,2)).unsqueeze(1)
            else:
                residual_sqnorm = torch.cat((residual_sqnorm,
                                             torch.real(residual_signal).pow(2).mean(dim=(1,2)).unsqueeze(1)),
                                            dim=1)
            if self.detach_by_iter:
                # Detach the gradient so that each iteration is penalized separately.
                residual_signal = residual_signal.detach()

            # NOTE: Currently, the model is trained end-to-end, where the Blaschke parameters
            # are used for downstream classification, and the gradient for classification can be backproped
            # through the Blaschke networks (param_net).
            curr_iter_coeffs = torch.hstack((layer.alpha, layer.beta, torch.real(layer.scale), torch.imag(layer.scale)))
            if blaschke_coeffs is None:
                blaschke_coeffs = curr_iter_coeffs
            else:
                blaschke_coeffs = torch.cat((blaschke_coeffs, curr_iter_coeffs), dim=1)

        y_pred = self.classifier(blaschke_coeffs)

        return y_pred, residual_sqnorm, blaschke_coeffs


    def to(self, device: str):
        super().to(device)
        self.device = device
        return self


if __name__ == '__main__':
    model = BlaschkeNetwork1d(layers=3, signal_len=100, num_channels=10, patch_size=20)
    x = torch.normal(0, 1, size=(32, 10, 100))
    y, x_approximations = model(x)

