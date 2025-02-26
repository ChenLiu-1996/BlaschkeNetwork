import torch
import torch.nn as nn
from scipy.signal import hilbert
from typing import List, Tuple
from einops import rearrange

import os
import sys
import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)
from models.transformer1d import Transformer1d


class BlaschkeLayer2d(nn.Module):
    '''
    Blaschke Layer for 2D signals.
    The input dimension is assumed to be [B, C, H, W].

    In Blaschke decomposition, a function F(x) can be approximated by
    F(x) = s_1 * B_1 + s_2 * B_1 * B_2 + s_3 * B_1 * B_2 * B_3 + ...
        where s_i is a scaling factor (scalar),
        and B_i is the i-th Blaschke product.
        B(x) = exp(i \theta(x)),
        \theta(x) = \sum_{j >= 0} \sigma ((x - \alpha_j) / \beta_j),
        \sigma(x) = \arctan(x) + \pi / 2

    Args:
    -----------
        param_net_h (torch.nn.Module): Neural network to extract Blaschke parameters along height axis.
        param_net_w (torch.nn.Module): Neural network to extract Blaschke parameters along width axis.
        num_blaschke (int): Number of Blaschke components.
        eps (float): Small number for numerical stability.
        device (torch.device): Torch device.
    '''

    def __init__(self,
                 param_net_h: torch.nn.Module,
                 param_net_w: torch.nn.Module,
                 num_blaschke: int = 1,
                 eps: float = 1e-11,
                 device: str = 'cpu') -> None:

        super().__init__()

        self.param_net_h = param_net_h
        self.param_net_w = param_net_w
        self.num_blaschke = num_blaschke
        assert self.num_blaschke == 1, 'Current implementation only supports 1 Blaschke factor.'

        # Set other parameters
        self.eps = eps
        self.to(device)

    def __repr__(self):
        return (f"BlaschkeLayer2d("
                f"num_blaschke={self.num_blaschke})")

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Custom activation function used in the BlaschkeLayer2d.
        '''
        output = torch.arctan(x) + torch.pi / 2
        return output

    def estimate_blaschke_parameters(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Estimate the Blaschke parameters.

        Args:
        -----
            x : 4D torch.float
                inputs, shape [B, C, H, W] (number of samples, channel, height, width)
        '''
        assert len(x.shape) == 4
        x_real, x_imag = torch.real(x), torch.imag(x)
        x = torch.cat((x_real, x_imag), dim=1).float()
        assert len(x.shape) == 4

        params_h = self.param_net_h(rearrange(x, 'b c h w -> b (c w) h'))
        params_w = self.param_net_w(rearrange(x, 'b c h w -> b (c h) w'))
        assert len(params_h.shape) == 2 and params_h.shape[0] == x.shape[0] and params_h.shape[1] == 4
        assert len(params_w.shape) == 2 and params_w.shape[0] == x.shape[0] and params_w.shape[1] == 4
        alpha_h, log_beta_h, scale_real_h, scale_imag_h = params_h[:, 0], params_h[:, 1], params_h[:, 2], params_h[:, 3]
        alpha_w, log_beta_w, scale_real_w, scale_imag_w = params_w[:, 0], params_w[:, 1], params_w[:, 2], params_w[:, 3]
        self.alpha_h = alpha_h
        self.alpha_w = alpha_w
        self.beta_h = torch.exp(log_beta_h + self.eps)
        self.beta_w = torch.exp(log_beta_w + self.eps)
        self.scale_h = scale_real_h + 1j * scale_imag_h
        self.scale_w = scale_real_w + 1j * scale_imag_w
        return

    def compute_blaschke_product(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute the $B(x)$ for the input x.

        Args:
        -----
            x : 4D torch.float
                inputs, shape [B, C, H, W] (number of samples, channel, height, width)

        Returns:
        --------
            y : 4D torch.float
                outputs, shape [B, C, H, W] (number of samples, channel, height, width)
        '''
        frac_h = (rearrange(x, 'b c h w -> b (c h w)') - self.alpha_h.unsqueeze(-1)) / self.beta_h.unsqueeze(-1)
        frac_w = (rearrange(x, 'b c h w -> b (c h w)') - self.alpha_w.unsqueeze(-1)) / self.beta_w.unsqueeze(-1)
        theta_x_h = self.activation(frac_h)
        theta_x_w = self.activation(frac_w)
        blaschke_product_h = torch.exp(1j * theta_x_h)
        blaschke_product_w = torch.exp(1j * theta_x_w)
        blaschke_product = blaschke_product_h * blaschke_product_w
        blaschke_product = rearrange(blaschke_product, 'b (c h w) -> b c h w', c=x.shape[1], h=x.shape[2], w=x.shape[3])
        return blaschke_product

    def to(self, device: str):
        super().to(device)
        self.device = device
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
        -----
            x : 4D torch.float
                inputs, shape [B, C, H, W] (number of samples, channel, height, width)

        Returns:
        --------
            y : 4D torch.float
                outputs, shape [B, C, H, W] (number of samples, channel, height, width)

        Example
        -------
        >>> model = BlaschkeLayer2d(image_dim=(224, 224), num_channels=3, num_blaschke=1)
        >>> x = torch.normal(0, 1, size=(32, 3, 224, 224))
        >>> y = model(x)
        '''

        # Input should be a [B, C, H, W] signal.
        assert len(x.shape) == 4

        # Compute the Blaschke product $B(x)$. [B, L, C_out]
        self.estimate_blaschke_parameters(x)
        blaschke_product = self.compute_blaschke_product(x)
        return blaschke_product


class BlaschkeNetwork2d(nn.Module):
    '''
    Blaschke Network for 2D images.

    A neural network composed of multiple BlaschkeLayer2d instances that leverages the Blaschke product
    for complex function representation. This network is designed for applications requiring
    iterative, layer-based transformations, such as complex function approximation or signal
    reconstruction.

    Args:
    -----------
        image_dim (Tuple[int]): Dimension of the image in (height, width).
        num_channels (int): Number of image channels. Usually 3 or 1.
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
                 image_dim: Tuple[int],
                 num_channels: int = 3,
                 patch_size: int = 1,
                 layers: int = 5,
                 out_classes: int = 10,
                 num_blaschke_list: List[int] = None,
                 param_net_dim: int = 256,
                 param_net_depth: int = 2,
                 param_net_heads: int = 8,
                 param_net_mlp_dim: int = 256,
                 eps: float = 1e-6,
                 device: str = 'cpu',
                 seed: int = 1) -> None:

        super().__init__()
        torch.manual_seed(seed)

        self.image_dim = image_dim
        self.num_channels = num_channels
        self.out_classes = out_classes
        self.layers = layers
        self.num_blaschke_list = num_blaschke_list

        if self.num_blaschke_list is None:
            # Determine Blaschke parameters if not provided
            self.num_blaschke_list = [1 for _ in range(self.layers)]

        # Initialize learnable parameters for computing Blaschke product.
        # 4 output neuros, respectively for:
        #   alpha, log_beta, scale_real, scale_imaginary
        num_blaschke_params = 4

        self.param_net_h = Transformer1d(
            seq_len=image_dim[0],
            patch_size=patch_size,
            channels=2 * num_channels * image_dim[1],  # (real, imaginary) for each channel
            num_classes=num_blaschke_params,
            dim=param_net_dim,
            depth=param_net_depth,
            heads=param_net_heads,
            mlp_dim=param_net_mlp_dim,
            dropout=0,
            emb_dropout=0,
        )

        self.param_net_w = Transformer1d(
            seq_len=image_dim[1],
            patch_size=patch_size,
            channels=2 * num_channels * image_dim[0],  # (real, imaginary)
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
                BlaschkeLayer2d(num_blaschke=self.num_blaschke_list[layer_idx],
                                param_net_h=self.param_net_h,
                                param_net_w=self.param_net_w,
                                eps=eps,
                                device=device)
            )

        num_features = sum(self.num_blaschke_list) * num_blaschke_params * 2
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
        '''Initialize the weights of the BlaschkeLayer2d.'''
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
        -----
            x : 4D torch.float
                inputs, shape [B, C, H, W] (number of samples, channel, height, width)

        Returns:
        --------
            y_pred : 2D torch.float
                outputs, shape [B, CLS] (number of samples, number of classes)
            residual_signals_sqsum : torch.float
                squared norm of signal approximation error
        '''

        # Input should be a [B, C, H, W] signal.
        assert len(x.shape) == 4

        # Complexify the signal using hilbert transform.
        # Real part is the same. Imaginary part is the Hilbert transform of the real part.
        signal_complex = torch.from_numpy(hilbert(x.cpu().detach().numpy())).to(x.device)

        blaschke_products = []
        residual_signal, residual_signals_sqsum = signal_complex, 0
        parameters_for_downstream = None

        for layer in self.encoder:
            blaschke_products.append(layer(residual_signal))

            # This is B_1 * B_2 * ... * B_n.
            cumulative_product = blaschke_products[0]
            for blaschke_product in blaschke_products[1:]:
                cumulative_product = cumulative_product * blaschke_product

            # This is s_n * B_1 * B_2 * ... * B_n.
            scale = layer.scale_h * layer.scale_w
            curr_signal_approx = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * cumulative_product

            residual_signal = residual_signal - curr_signal_approx
            residual_signals_sqsum = residual_signals_sqsum + torch.real(residual_signal).pow(2).mean()

            # NOTE: Currently, the model is trained end-to-end, where the Blaschke parameters
            # are used for downstream classification, and the gradient for classification can be backproped
            # through the Blaschke networks (param_net_h, param_net_w).
            curr_parameters = torch.stack((layer.alpha_h, layer.alpha_w,
                                           layer.beta_h, layer.beta_w,
                                           torch.real(layer.scale_h), torch.imag(layer.scale_h),
                                           torch.real(layer.scale_w), torch.imag(layer.scale_w)), dim=1)
            if parameters_for_downstream is None:
                parameters_for_downstream = curr_parameters
            else:
                parameters_for_downstream = torch.cat((parameters_for_downstream, curr_parameters), dim=1)

        parameters_for_downstream = parameters_for_downstream.detach()
        y_pred = self.classifier(parameters_for_downstream)

        return y_pred, residual_signals_sqsum


    def to(self, device: str):
        super().to(device)
        self.device = device
        return self


if __name__ == '__main__':
    model = BlaschkeNetwork2d(layers=3, image_dim=(224, 224), num_channels=3, patch_size=16)
    x = torch.normal(0, 1, size=(32, 3, 224, 224))
    y, x_approximations = model(x)

