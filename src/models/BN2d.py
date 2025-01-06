import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict



# class BDN(nn.Module):
#     def __init__(
#         self,
#         layers_hidden: List[int],
#         base_activation = F.silu,
#     ) -> None:
#         super().__init__()
#         self.layers = nn.ModuleList([
#             BDNLayer(
#                 in_dim,
#                 out_dim,
#                 base_activation=base_activation,
#             ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
#         ])

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x


# class BDNLayer(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         use_base_update: bool = True,
#         use_layernorm: bool = True,
#         base_activation = F.silu,
#     ) -> None:
#         super().__init__()

#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.layernorm = None
#         if use_layernorm:
#             assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
#             self.layernorm = nn.LayerNorm(input_dim)

#         if use_base_update:
#             self.base_activation = base_activation
#             self.base_linear = nn.Linear(input_dim, output_dim)

#     def forward(self, x, use_layernorm=True):
#         if self.layernorm is not None and use_layernorm:
#             spline_basis = self.rbf(self.layernorm(x))
#         else:
#             spline_basis = self.rbf(x)

#         ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))

#         if self.use_base_update:
#             base = self.base_linear(self.base_activation(x))
#             ret = ret + base

#         return ret


class BDLayerImage(nn.Module):
    """
    BDLayer Class (for image input).
    The input dimension is assumed to be [B, C, H, W].

    Attributes:
    -----------
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        nb_blaschke (int): Number of Blaschke factors
        eps (float): Small constant to prevent division by zero
        device (str): Device to run the computations ('cpu' or 'cuda')
        att_learnable (bool, optional): If True, the Attention matrix in each layer is learnable; otherwise, it remains fixed. Default is False.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 nb_blaschke: int,
                 eps: float = 1e-11,
                 device: str = 'cpu',
                 att_learnable: bool = False) -> None:

        super().__init__()

        # Initialize learnable parameters for computing Blaschke product
        self.att_learnable = att_learnable
        self.alpha = nn.Parameter(torch.empty(1, 1, 1, in_dim, nb_blaschke, dtype=torch.float32))
        self.beta = nn.Parameter(torch.empty(1, 1, 1, in_dim, nb_blaschke, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.empty(1, 1, 1, in_dim, dtype=torch.float32))

        # Projection parameter: fixed or learnable
        self.attention = nn.Parameter(torch.empty(in_dim, out_dim, dtype=torch.float32), requires_grad=att_learnable)

        # Initialize weights
        self.initialize_weights()

        # Set other parameters
        self.eps = eps
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nb_blaschke = nb_blaschke
        self.to(device)

    def __repr__(self):
        return (f"BDLayerImage("
                f"in_dim={self.in_dim}, "
                f"out_dim={self.out_dim})")

    def initialize_weights(self) -> None:
        """Initialize the weights of the BDLayerImage."""
        with torch.no_grad():
            self.alpha.uniform_(-1, 1)
            self.beta.uniform_(0, 1)
            self.gamma.fill_(0.01)
            if self.att_learnable:
                self.attention.uniform_(0, 1)
            else:
                self.attention.fill_(1)

    def _sigma_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Custom activation function used in the BDLayerImage.
        """
        output = torch.arctan(x) + torch.pi / 2
        return output

    def _compute_blaschke_factors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Blaschke factors ($\theta(x)$) for the input x.
        x shape: [B, H, W, C, 1]
        """
        frac = (x - self.alpha) / (self.beta + self.eps)
        return torch.sum(self._sigma_function(frac), dim=-1)

    def to(self, device: str):
        super().to(device)
        self.device = device
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        -----
            x : 4D torch.float
                inputs, shape [B, C_in, H, W] (number of samples, input dimension, height, width)

        Returns:
        --------
            y : 4D torch.float
                outputs, shape [B, C_out, H, W] (number of samples, output dimension, height, width)

        Example
        -------
        >>> model = BDLayerImage(in_dim=3, out_dim=5, nb_blaschke=2)
        >>> x = torch.normal(0, 1, size=(100, 3, 16, 16))
        >>> y = model(x)
        """

        assert len(x.shape) == 4

        # Channel first to channel last. [B, H, W, C_in]
        x = x.transpose(1, 3)
        # Add dimension for broadcasting the Blaschke factors. [B, H, W, C_in, 1]
        x = x.unsqueeze(-1)
        # Compute the Blaschke factors $\theta(x)$. [B, H, W, C_in]
        blaschke_factors = self._compute_blaschke_factors(x)
        # Apply the phase shift. [B, H, W, C_in]
        theta_shifted = blaschke_factors #+ self.gamma
        # Compute the projection. [B, H, W, C_out]
        attention_sum = (1 / torch.sum(self.attention, dim=0)) * self.attention
        theta_scaled = theta_shifted @ attention_sum
        # Transform into the complex domain. [B, H, W, C_out]
        out = torch.exp(2j * theta_scaled)

        # Channel last to channel first. [B, C_out, H, W]
        out = out.transpose(1, 3)
        return out


class Complex2Real(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.real(x)


class BDNetworkImage(nn.Module):
    """
    BDNetwork Class (for image input).

    A neural network composed of multiple BDLayerImage instances that leverages the Blaschke product
    for complex function representation. This network is designed for applications requiring
    iterative, layer-based transformations, such as complex function approximation or signal
    reconstruction.

    Attributes:
    -----------
        layers_hidden (List[int]): Number of layers in the network.
        Blaschke_param (List[int], optional): List specifying the number of Blaschke parameters for each layer.
        p_mul (int, optional): Multiplier factor for constructing the BDLayerImage. Controls the scaling of the Blaschke parameters per layer. Default is None.
        start_p (int, optional): Initial Blaschke parameter for the first layer. Used to incrementally build the Blaschke parameters if `Blaschke_param` is not provided. Default is 1.
        eps (float, optional): A small constant to prevent division by zero in computations. Default is 1e-6.

        device (str, optional): Device for running computations, either 'cpu' or 'cuda'. Default is 'cpu'.
        seed (int, optional): Random seed for initialization to ensure reproducibility. Default is 2024.
        att_learnable (bool, optional): If True, the Attention matrix in each layer is learnable; otherwise, it remains fixed. Default is False.
    """

    def __init__(self,
                 layers_hidden: List[int],
                 Blaschke_param: List[int] = None,
                 p_mul: int = 1,
                 start_p: int = 1,
                 eps: float = 1e-6,
                 device: str = 'cpu',
                 seed: int = 1,
                 att_learnable: bool = False) -> None:

        super().__init__()
        torch.manual_seed(seed)
        self.layers_hidden = layers_hidden

        if Blaschke_param is None:
            # Determine Blaschke parameters if not provided
            Blaschke_param = [i * p_mul + start_p for i in range(len(self.layers_hidden)-2)]

        # Initialize the BNLayers
        self.encoder = nn.ModuleList([])

        for (in_dim, out_dim, param) in zip(layers_hidden[:-1], layers_hidden[1:], Blaschke_param):
            self.encoder.append(nn.Sequential(
                BDLayerImage(in_dim=in_dim,
                             out_dim=out_dim,
                             nb_blaschke=param,
                             eps=eps,
                             device=device,
                             att_learnable=att_learnable),
                Complex2Real(),
                nn.BatchNorm2d(out_dim),
            ))

        self.classifier = nn.Linear(layers_hidden[-2], layers_hidden[-1])

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        -----
            x : 4D torch.float
                inputs, shape [B, C_in, H, W] (number of samples, input dimension, height, width)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape [B, C_out] (number of samples, output dimension)
        """

        # Encode image as [B, L, C_in] signal.


        for layer in self.encoder:
            x = layer(x)
            x = torch.real(x)
            # Downsample.
            x = torch.nn.functional.interpolate(x,
                                                scale_factor=0.5,
                                                mode='bilinear',
                                                align_corners=True)

        # Global average pooling.
        x = x.mean(dim=(2, 3))
        x = self.classifier(x)
        return x


    def to(self, device: str):
        super().to(device)
        self.device = device
        return self



if __name__ == '__main__':
    model = BDNetworkImage(layers_hidden=[3, 16, 64, 10])
    x = torch.normal(0, 1, size=(100, 3, 32, 32))
    y = model(x)