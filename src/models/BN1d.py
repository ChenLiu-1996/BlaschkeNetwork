import torch
import torch.nn as nn
import numpy as np
from scipy.signal import hilbert
from typing import List
from einops import rearrange

import os
import sys
import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)
from models.transformer1d import Transformer1d


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

        # Set other parameters
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

        Args:
        -----
            x : 3D torch.float
                inputs, shape [B, C, L] (number of samples, channel, signal length)
        '''
        assert len(x.shape) == 3
        x_real, x_imag = torch.real(x), torch.imag(x)
        x = torch.cat((x_real, x_imag), dim=1).float()
        assert len(x.shape) == 3

        params = self.param_net(x)
        assert len(params.shape) == 2 and params.shape[0] == x.shape[0] and params.shape[1] == 4
        alpha, log_beta, scale_real, scale_imag = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
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
                inputs, shape [B, C, L] (number of samples, channel, signal length)

        Returns:
        --------
            y : 3D torch.float
                outputs, shape [B, C, L] (number of samples, channel, signal length)
        '''
        frac = (rearrange(x, 'b c l -> b (c l)') - self.alpha.unsqueeze(-1)) / self.beta.unsqueeze(-1)
        theta_x = self.activation(frac)
        blaschke_factor = torch.exp(1j * theta_x)
        blaschke_factor = rearrange(blaschke_factor, 'b (c l) -> b c l', c=x.shape[1], l=x.shape[2])
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
                inputs, shape [B, C, L] (number of samples, channel, signal length)

        Returns:
        --------
            y : 3D torch.float
                outputs, shape [B, C, L] (number of samples, channel, signal length)

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
                 out_classes: int = 10,
                 num_blaschke_list: List[int] = None,
                 param_net_dim: int = 64,
                 param_net_depth: int = 2,
                 param_net_heads: int = 4,
                 param_net_mlp_dim: int = 64,
                 eps: float = 1e-6,
                 device: str = 'cpu',
                 seed: int = 1) -> None:

        super().__init__()
        torch.manual_seed(seed)

        self.signal_len = signal_len
        self.num_channels = num_channels
        self.out_classes = out_classes
        self.layers = layers
        self.num_blaschke_list = num_blaschke_list

        if self.num_blaschke_list is None:
            # Determine Blaschke parameters if not provided
            self.num_blaschke_list = [1 for _ in range(self.layers)]

        # Initialize learnable parameters for computing Blaschke product.
        # 4 output neurons, respectively for:
        #   alpha, log_beta, scale_real, scale_imaginary
        num_blaschke_params = 4

        self.param_net = Transformer1d(
            seq_len=signal_len,
            patch_size=patch_size,
            channels=2 * num_channels,  # (real, imaginary)
            num_classes=num_blaschke_params,
            dim=param_net_dim,
            depth=param_net_depth,
            heads=param_net_heads,
            mlp_dim=param_net_mlp_dim,
            dropout=0,
            emb_dropout=0,
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

        num_features = sum(self.num_blaschke_list) * num_blaschke_params
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
                inputs, shape [B, C, L] (number of samples, channel, signal length)

        Returns:
        --------
            y_pred : 2D torch.float
                outputs, shape [B, CLS] (number of samples, number of classes)
            residual_signals_sqsum : torch.float
                squared norm of signal approximation error
        '''

        # Input should be a [B, C, L] signal.
        assert len(x.shape) == 3

        # Complexify the signal using hilbert transform.
        signal_complex = self.complexify(x)

        blaschke_factors = []
        residual_signal, residual_sqnorm = signal_complex, 0
        parameters_for_downstream = None

        for layer in self.encoder:
            blaschke_factors.append(layer(residual_signal))

            # This is B_1 * B_2 * ... * B_n.
            blaschke_product = blaschke_factors[0]
            for blaschke_factor in blaschke_factors[1:]:
                blaschke_product = blaschke_product * blaschke_factor

            # This is s_n * B_1 * B_2 * ... * B_n.
            curr_signal_approx = layer.scale.unsqueeze(-1).unsqueeze(-1) * blaschke_product

            # F_{n+1} = F_n - s_n * B_1 * ... * B_n.
            residual_signal = residual_signal - curr_signal_approx
            residual_sqnorm = residual_sqnorm + torch.real(residual_signal).pow(2).mean()

            # Detach the gradient so that each iteration is penalized separately.
            residual_signal = residual_signal.detach()

            # NOTE: Currently, the model is trained end-to-end, where the Blaschke parameters
            # are used for downstream classification, and the gradient for classification can be backproped
            # through the Blaschke networks (param_net).
            curr_parameters = torch.stack((layer.alpha, layer.beta, torch.real(layer.scale), torch.imag(layer.scale)), dim=1)
            if parameters_for_downstream is None:
                parameters_for_downstream = curr_parameters
            else:
                parameters_for_downstream = torch.cat((parameters_for_downstream, curr_parameters), dim=1)

        y_pred = self.classifier(parameters_for_downstream)

        return y_pred, residual_sqnorm


    def to(self, device: str):
        super().to(device)
        self.device = device
        return self


if __name__ == '__main__':
    model = BlaschkeNetwork1d(layers=3, signal_len=100, num_channels=10, patch_size=20)
    x = torch.normal(0, 1, size=(32, 10, 100))
    y, x_approximations = model(x)

