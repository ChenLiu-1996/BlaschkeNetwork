import torch
import torch.nn as nn
import scipy.signal as signal
from typing import List

from models.transformer1d import Transformer1d


class BlaschkeLayer1d(nn.Module):
    '''
    Blaschke Layer for 1d signal.
    The input dimension is assumed to be [B, L, C].

    In Blaschke decomposition, a function F(x) can be approximated by
    F(x) = s_1 * B_1 + s_2 * B_1 * B_2 + s_3 * B_1 * B_2 * B_3 + ...
        where s_i is a scaling factor (scalar),
        and B_i is the i-th Blaschke product.
        B(x) = exp(i \theta(x)),
        \theta(x) = \sum_{j >= 0} \sigma ((x - \alpha_j) / \beta_j),
        \sigma(x) = \arctan(x) + \pi / 2

    Attributes:
    -----------
        eps (float): Small constant to prevent division by zero
        device (str): Device to run the computations ('cpu' or 'cuda')
        proj_learnable (bool, optional): If True, the projection matrix in each layer is learnable;
            otherwise, it remains fixed. Default is False.
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
        '''
        assert len(x.shape) == 2
        x_real, x_imag = torch.real(x), torch.imag(x)
        x = torch.stack((x_real, x_imag), dim=1).float()
        assert len(x.shape) == 3

        params = self.param_net(x)
        assert len(params.shape) == 2 and params.shape[0] == x.shape[0] and params.shape[1] == 4
        alpha, log_beta, scale_real, scale_imag = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        self.alpha = alpha
        self.beta = torch.exp(log_beta + self.eps)
        self.scale = scale_real + 1j * scale_imag
        return

    def compute_blaschke_product(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute the $B(x)$ for the input x.
        x shape: [B, L]
        '''
        frac = (x - self.alpha.unsqueeze(-1)) / self.beta.unsqueeze(-1)
        theta_x = self.activation(frac)
        blaschke_product = torch.exp(1j * theta_x)
        return blaschke_product

    def to(self, device: str):
        super().to(device)
        self.device = device
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
        -----
            x : 2D torch.float
                inputs, shape [B, C_in, L] (number of samples, signal length)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape [B, L] (number of samples, signal length)

        Example
        -------
        >>> model = BlaschkeLayer1d(signal_dim=100, num_blaschke=1)
        >>> x = torch.normal(0, 1, size=(32, 100))
        >>> y = model(x)
        '''

        # [B, L]
        assert len(x.shape) == 2

        # Compute the Blaschke product $B(x)$. [B, L, C_out]
        self.estimate_blaschke_parameters(x)
        blaschke_product = self.compute_blaschke_product(x)
        return blaschke_product


class BlaschkeNetwork1d(nn.Module):
    '''
    Blaschke Network for 1d signal.

    A neural network composed of multiple BlaschkeLayer1d instances that leverages the Blaschke product
    for complex function representation. This network is designed for applications requiring
    iterative, layer-based transformations, such as complex function approximation or signal
    reconstruction.

    Attributes:
    -----------
        layers_hidden (List[int]): Number of layers in the network.
        Blaschke_param (List[int], optional): List specifying the number of Blaschke parameters for each layer.
        p_mul (int, optional): Multiplier factor for constructing the BlaschkeLayer1d. Controls the scaling of the Blaschke parameters per layer. Default is None.
        start_p (int, optional): Initial Blaschke parameter for the first layer. Used to incrementally build the Blaschke parameters if `Blaschke_param` is not provided. Default is 1.
        eps (float, optional): A small constant to prevent division by zero in computations. Default is 1e-6.

        device (str, optional): Device for running computations, either 'cpu' or 'cuda'. Default is 'cpu'.
        seed (int, optional): Random seed for initialization to ensure reproducibility. Default is 2024.
        proj_learnable (bool, optional): If True, the projection matrix in each layer is learnable; otherwise, it remains fixed. Default is False.
    '''

    def __init__(self,
                 signal_dim: int,
                 patch_size: int,
                 layers: int = 3,
                 out_classes: int = 10,
                 num_blaschke_list: List[int] = None,
                 eps: float = 1e-6,
                 device: str = 'cpu',
                 seed: int = 1) -> None:

        super().__init__()
        torch.manual_seed(seed)

        self.signal_dim = signal_dim
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

        # scaling_factor = signal_dim / num_blaschke_params
        # self.param_net = nn.Sequential(
        #     nn.Linear(signal_dim * 2, int(num_blaschke_params * scaling_factor**0.7)),
        #     nn.BatchNorm1d(int(num_blaschke_params * scaling_factor**0.7)),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(int(num_blaschke_params * scaling_factor**0.7), int(num_blaschke_params * scaling_factor**0.5)),
        #     nn.BatchNorm1d(int(num_blaschke_params * scaling_factor**0.5)),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(int(num_blaschke_params * scaling_factor**0.5), num_blaschke_params),
        # )

        self.param_net = Transformer1d(
            seq_len=signal_dim,
            patch_size=patch_size,
            channels=2,  # (real, imaginary)
            num_classes=num_blaschke_params,
            dim=256,
            depth=2,
            heads=8,
            mlp_dim=128,
            dropout=0.1,
            emb_dropout=0.1
        )

        # Initialize weights
        self.initialize_weights()

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
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, self.out_classes),
            nn.BatchNorm1d(self.out_classes),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_classes, self.out_classes),
        )

        self.to(device)

    def initialize_weights(self) -> None:
        '''Initialize the weights of the BlaschkeLayer1d.'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
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
            x : 2D torch.float
                inputs, shape [B, L] (number of samples, signal length)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape [B, L] (number of samples, signal length)
        '''

        # Input should be a [B, L] signal.
        assert len(x.shape) == 2

        # Compute the Z-transform using CZT (Chirp Z-transform).
        signal_complex = torch.from_numpy(signal.czt(x)).to(x.device)

        blaschke_products = []
        residual_signal, residual_signals_sqsum = signal_complex, 0
        parameters_for_downstream = None

        for layer in self.encoder:
            blaschke_products.append(layer(residual_signal))
            cumulative_product = blaschke_products[0]
            for blaschke_product in blaschke_products[1:]:
                cumulative_product = cumulative_product * blaschke_product

            # Take the manigude of complex number.
            curr_signal_approx = layer.scale.unsqueeze(-1) * cumulative_product
            residual_signal = residual_signal - curr_signal_approx
            residual_signals_sqsum = residual_signals_sqsum + torch.real(residual_signal).pow(2).mean()
            # Stop the gradient from passing to the next recurrence layer.
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

        return y_pred, residual_signals_sqsum


    def to(self, device: str):
        super().to(device)
        self.device = device
        return self



if __name__ == '__main__':
    model = BlaschkeNetwork1d(layers=3, signal_dim=100, patch_size=20)
    x = torch.normal(0, 1, size=(32, 100))
    y, x_approximations = model(x)

