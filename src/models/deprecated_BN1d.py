import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
from scipy.signal import hilbert
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
        num_roots (int): Number of Blaschke roots.
        eps (float): Small number for numerical stability.
        device (torch.device): Torch device.
    '''

    def __init__(self,
                 param_net: torch.nn.Module,
                 num_roots: int = 1,
                 eps: float = 1e-11,
                 device: str = 'cpu') -> None:

        super().__init__()

        self.param_net = param_net
        self.num_roots = num_roots

        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)  # To facilitate checkpointing.
        self.eps = eps

        self.to(device)

    def __repr__(self):
        return (f"BlaschkeLayer1d("
                f"num_roots={self.num_roots})")

    def activation(self, signal: torch.Tensor) -> torch.Tensor:
        '''
        Custom activation function used in the BlaschkeLayer1d.
        '''
        output = torch.arctan(signal) + torch.pi / 2
        return output

    def estimate_blaschke_parameters(self, signal_complex: torch.Tensor) -> torch.Tensor:
        '''
        Estimate the Blaschke parameters.
        Using gradient checkpointing to trade time for space.
        Otherwise it is easy to OOM when using many layers.

        Args:
        -----
            signal_complex : 3D torch.float
                inputs, shape [B, C, L] (batch size, channel, signal length)
        '''

        B, C, L = signal_complex.shape                                        # [batch_size, num_channels, seq_len]
        assert len(signal_complex.shape) == 3
        x_real = torch.real(signal_complex)                                   # [batch_size, num_channels, seq_len]
        x_imag = torch.imag(signal_complex)                                   # [batch_size, num_channels, seq_len]

        # signal_real_fft = torch.fft.rfft(x_real, n=2*L-1, dim=-1)             # [batch_size, num_channels, seq_len * 2]
        # x_fft_real = torch.real(signal_real_fft)                              # [batch_size, num_channels, seq_len]
        # x_fft_imag = torch.imag(signal_real_fft)                              # [batch_size, num_channels, seq_len]

        # signal = torch.stack((x_real, x_imag,
        #                       x_fft_real, x_fft_imag), dim=2).float()         # [batch_size, num_channels, 4, seq_len]

        signal = torch.stack((x_real, x_imag), dim=2).float()                 # [batch_size, num_channels, 2, seq_len]
        signal = rearrange(signal, 'b c r l -> (b c) r l')                    # [batch_size * num_channels, 2, seq_len]
        assert len(signal.shape) == 3

        if any_requires_grad(self.param_net):
            params = checkpoint(self.param_net, signal, self.dummy_tensor,
                                use_reentrant=False)                          # [batch_size * num_channels, num_roots * 3 + 1]
        else:
            params = self.param_net(signal, self.dummy_tensor)                # [batch_size * num_channels, num_roots * 3 + 1]
        params = rearrange(params, '(b c) p -> b c p', b=B, c=C)              # [batch_size, num_channels, num_roots * 3 + 1]

        self.alphas = params[..., :self.num_roots]
        self.log_betas = params[..., self.num_roots : self.num_roots * 2]
        self.logit_gammas = params[..., self.num_roots * 2 : self.num_roots * 3]
        self.scale_real = params[..., self.num_roots * 3 : self.num_roots * 3 + 1]
        self.scale_imag = params[..., self.num_roots * 3 + 1 : self.num_roots * 3 + 2]
        return

    @property
    def betas(self):
        '''
        betas have to be positive.
        '''
        return torch.nn.functional.softplus(self.log_betas)

    @property
    def gammas(self):
        '''
        gammas have to be between 0 and 1.
        '''
        return torch.sigmoid(self.logit_gammas)

    @property
    def scale(self):
        return self.scale_real + 1j * self.scale_imag

    def compute_blaschke_factor(self, signal_complex: torch.Tensor) -> torch.Tensor:
        '''
        Compute $B(x)$ for the input signal.

        B(x) = exp(i \theta(x))
        \theta(x) = \sum_j activation((x - \alpha_j) / \beta_j)
        Note that here `x` is not the signal, but rather the time indices.
        In the implementation, to avoid confusion, we use the symbol `t`.

        Args:
        -----
            signal_complex : 3D torch.float
                inputs, shape [B, C, L] (batch size, channel, signal length)

        Returns:
        --------
            blaschke_factor : 3D torch.float
                outputs, shape [B, C, L] (batch size, channel, signal length)
        '''

        signal_complex = signal_complex.unsqueeze(2)                          # [B, C, 1, L]
        t = rearrange(torch.linspace(-1, 1, signal_complex.shape[-1]),
                      'l -> 1 1 1 l').to(signal_complex.device)               # [1, 1, 1, L]
        alphas = self.alphas.unsqueeze(-1)                                    # [B, C, R, 1]
        betas = self.betas.unsqueeze(-1)                                      # [B, C, R, 1]
        gammas = self.gammas.unsqueeze(-1)                                    # [B, C, R, 1]

        activated = self.activation((t - alphas) / betas)                     # [B, C, R, L]
        phase = (gammas * activated).sum(dim=2)                               # sum over roots (R) → [B, C, L]

        blaschke_factor = torch.exp(1j * phase)
        return blaschke_factor

    def to(self, device: str):
        super().to(device)
        self.device = device
        return self

    def forward(self, signal_complex: torch.Tensor) -> torch.Tensor:
        '''
        Args:
        -----
            signal_complex : 3D torch.float
                inputs, shape [B, C, L] (batch size, channel, signal length)

        Returns:
        --------
            blaschke_factor : 3D torch.float
                outputs, shape [B, C, L] (batch size, channel, signal length)

        Example
        -------
        >>> model = BlaschkeLayer1d(signal_len=100, signal_channel=10, num_roots=1)
        >>> signal = torch.normal(0, 1, size=(32, 10, 100))
        >>> blaschke_factor = model(signal)
        '''

        # [B, C, L]
        assert len(signal_complex.shape) == 3

        # Compute the Blaschke product $B(x)$. [B, C, L]
        self.estimate_blaschke_parameters(signal_complex)
        blaschke_factor = self.compute_blaschke_factor(signal_complex)
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
        num_roots (int): Number of Blaschke roots in each layer.

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
                 num_roots: int = 8,
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
        self.detach_by_iter = detach_by_iter
        self.num_roots = num_roots

        # 3 per root (alpha, log_beta, gamma), 2 per iteration: (scale_real, scale_imag).
        num_classes = self.num_roots * 3 + 2
        param_net = PatchTST(
            num_channels=2,  # (real, imaginary)
            num_classes=num_classes,
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
        self.param_net = Ignore2ndArg(    # `Ignore2ndArg` is a wrapper to facilitate checkpointing.
            convert_ln_to_dyt(param_net)  # Replace layernorm with dynamic Tanh.
        )

        # Initialize the BNLayers
        self.encoder = nn.ModuleList([])

        for _ in range(self.layers):
            self.encoder.append(
                BlaschkeLayer1d(num_roots=self.num_roots,
                                param_net=self.param_net,
                                eps=eps,
                                device=device)
            )

        num_features = self.layers * self.num_channels * num_classes

        # In linear probing, only this is updated.
        # This is technically not a linear probing, but rather a two-stage training.
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, self.out_classes),
            nn.SiLU(),
            nn.BatchNorm1d(self.out_classes),
            nn.Linear(self.out_classes, self.out_classes),
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

    def complexify(self, signal: torch.Tensor, carrier_freq: float = 0) -> torch.Tensor:
        '''
        Complexify the signal for Blaschke decomposition.
        '''
        device = signal.device
        original_shape = signal.shape

        signal = signal.cpu().detach().numpy()
        signal = rearrange(signal, 'b c l -> (b c) l')  # b: batch size, c: number of channels, l: signal length.
        # Remove zero-order drift.
        signal = signal - np.mean(signal, axis=1, keepdims=True)
        # Hilbert transform.
        signal = hilbert(signal)
        # Frequency shifting by carrier frequency.
        time_indices = np.arange(signal.shape[-1])
        signal = signal * np.exp(1j * 2 * np.pi * carrier_freq * time_indices)
        # Mitigate boundary effects. This is a common approach when performing Fourier analyses, spectral filtering, etc.
        signal_boundary_smoothed = np.concatenate((signal, np.fliplr(np.conj(signal))), axis=1)
        mask_nonnegative_freq = np.ones_like(signal_boundary_smoothed)
        mask_nonnegative_freq[:, mask_nonnegative_freq.shape[1] // 2:] = 0
        signal = np.fft.ifft(np.fft.fft(signal_boundary_smoothed) * mask_nonnegative_freq)
        signal = signal[:, :signal.shape[1] // 2]
        signal = rearrange(signal, '(b c) l -> b c l', b=original_shape[0], c=original_shape[1])

        # Cast to torch Tensor.
        signal = torch.from_numpy(signal).to(device)
        return signal

    @torch.no_grad()
    def test_approximate(self, signal: torch.Tensor) -> torch.Tensor:
        # Input should be a [B, C, L] signal.
        assert len(signal.shape) == 3

        # Complexify the signal using hilbert transform.
        signal_complex = self.complexify(signal)

        blaschke_factors = []
        s_arr, B_prod_arr = None, None

        residual_signal = signal_complex
        for layer in self.encoder:
            blaschke_factors.append(layer(residual_signal))

            # Blaschke product is the cumulative product of Blaschke factors
            # B_1 * B_2 * ... * B_n.
            blaschke_product = 1
            for blaschke_factor in blaschke_factors:
                blaschke_product = blaschke_product * blaschke_factor

            # The Blaschke approximation is given by
            # s_n * B_1 * B_2 * ... * B_n.
            curr_signal_approx = layer.scale * blaschke_product

            # F_{n+1} = F_n - s_n * B_1 * ... * B_n.
            residual_signal = residual_signal - curr_signal_approx

            if s_arr is None:
                s_arr = layer.scale.cpu().unsqueeze(-1)
                B_prod_arr = blaschke_product.cpu().unsqueeze(-1)
            else:
                s_arr = torch.cat((s_arr, layer.scale.cpu().unsqueeze(-1)), dim=-1)
                B_prod_arr = torch.cat((B_prod_arr, blaschke_product.cpu().unsqueeze(-1)), dim=-1)

        return signal_complex, s_arr, B_prod_arr

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        '''
        Args:
        -----
            signal : 3D torch.float
                inputs, shape [B, C, L] (batch size, channel, signal length)

        Returns:
        --------
            y_pred : 2D torch.float
                outputs, shape [B, CLS] (batch size, number of classes)
            residual_sqnorm : torch.float
                squared norm of signal approximation error
            weighting_coeffs: torch.float
                model-predicted weighting coefficient for each Blaschke root
        '''

        # Input should be a [B, C, L] signal.
        assert len(signal.shape) == 3

        # Complexify the signal using hilbert transform.
        signal_complex = self.complexify(signal)

        blaschke_factors = []
        residual_signal, residual_sqnorm_withgrad, gamma_deviation_withgrad = signal_complex, None, None
        blaschke_coeffs, scales_withgrad = None, None
        mean_scale, active_roots_ratio = None, None

        for layer in self.encoder:
            blaschke_factors.append(layer(residual_signal))

            # Blaschke product is the cumulative product of Blaschke factors
            # B_1 * B_2 * ... * B_n.
            blaschke_product = blaschke_factors[-1]
            for blaschke_factor in blaschke_factors[:-1]:
                if self.detach_by_iter:
                    # Detach the gradient so that each iteration is penalized separately.
                    blaschke_product = blaschke_product * blaschke_factor.detach()
                else:
                    blaschke_product = blaschke_product * blaschke_factor

            # The Blaschke approximation is given by
            # s_n * B_1 * B_2 * ... * B_n.
            curr_signal_approx = layer.scale * blaschke_product

            # F_{n+1} = F_n - s_n * B_1 * ... * B_n.
            if self.detach_by_iter:
                # Detach the gradient so that each iteration is penalized separately.
                residual_signal = residual_signal.detach() - curr_signal_approx
            else:
                residual_signal = residual_signal - curr_signal_approx

            # This helps sanity checking the residual norms at each iteration.
            curr_sqnorm = torch.abs(residual_signal).pow(2).mean().unsqueeze(0)
            curr_deviation = layer.gammas * (1 - layer.gammas)
            if residual_sqnorm_withgrad is None:
                residual_sqnorm_withgrad = curr_sqnorm
                gamma_deviation_withgrad = curr_deviation
            else:
                residual_sqnorm_withgrad = torch.cat((residual_sqnorm_withgrad, curr_sqnorm), dim=0)
                gamma_deviation_withgrad = torch.cat((gamma_deviation_withgrad, curr_deviation), dim=0)

            # Track the mean scale per layer.
            if mean_scale is None:
                mean_scale = layer.scale.detach().cpu().mean().unsqueeze(0).abs()
            else:
                mean_scale = torch.cat((mean_scale, layer.scale.detach().cpu().mean().unsqueeze(0).abs()), dim=0)

            # Track the ratio of active roots per layer.
            active_roots = (layer.logit_gammas > 0).sum().cpu().detach()
            total_roots = layer.logit_gammas.numel()
            if active_roots_ratio is None:
                active_roots_ratio = (active_roots / total_roots).unsqueeze(0)
            else:
                active_roots_ratio = torch.cat((active_roots_ratio, (active_roots / total_roots).unsqueeze(0)), dim=0)

            # NOTE: Currently, the model is trained end-to-end, where the Blaschke parameters
            # are used for downstream classification, and the gradient for classification can be backproped
            # through the Blaschke networks (param_net).
            curr_iter_coeffs = torch.cat((rearrange(layer.alphas, 'b c r -> b (c r)'),
                                          rearrange(layer.betas, 'b c r -> b (c r)'),
                                          rearrange(layer.gammas, 'b c r -> b (c r)'),
                                          rearrange(layer.scale_real, 'b c r -> b (c r)'),
                                          rearrange(layer.scale_imag, 'b c r -> b (c r)')
                                          ), dim=1).detach()

            if blaschke_coeffs is None:
                blaschke_coeffs = curr_iter_coeffs
            else:
                blaschke_coeffs = torch.cat((blaschke_coeffs, curr_iter_coeffs), dim=1)

            if scales_withgrad is None:
                scales_withgrad = layer.scale
            else:
                scales_withgrad = torch.cat((scales_withgrad, layer.scale), dim=2)

        y_pred = self.classifier(blaschke_coeffs)

        return y_pred, residual_sqnorm_withgrad, gamma_deviation_withgrad, scales_withgrad, mean_scale.numpy(), active_roots_ratio.numpy()


    def to(self, device: str):
        super().to(device)
        self.device = device
        return self


if __name__ == '__main__':
    model = BlaschkeNetwork1d(layers=3, num_roots=4, signal_len=100, num_channels=10, patch_size=20)
    signal = torch.normal(0, 1, size=(32, 10, 100))
    y, residual_sqnorm, gamma_deviation, mean_scale, active_roots_ratio = model(signal)
    signal_complex, s_arr, B_prod_arr = model.test_approximate(signal[:1, :1, :])
