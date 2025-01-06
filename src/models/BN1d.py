import torch
import torch.nn as nn
from typing import List


def count_parameters(model, trainable_only: bool = False):
    if not trainable_only:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BlaschkeLayer1d(nn.Module):
    """
    Blaschke Layer for 1d signal.
    The input dimension is assumed to be [B, L, C].

    Attributes:
    -----------
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        nb_blaschke (int): Number of Blaschke factors
        eps (float): Small constant to prevent division by zero
        device (str): Device to run the computations ('cpu' or 'cuda')
        proj_learnable (bool, optional): If True, the projection matrix in each layer is learnable; otherwise, it remains fixed. Default is False.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 nb_blaschke: int,
                 eps: float = 1e-11,
                 device: str = 'cpu',
                 proj_learnable: bool = False) -> None:

        super().__init__()

        # Initialize learnable parameters for computing Blaschke product
        self.proj_learnable = proj_learnable
        self.alpha = nn.Parameter(torch.empty(1, 1, out_dim, nb_blaschke, dtype=torch.float32))
        self.log_beta = nn.Parameter(torch.empty(1, 1, out_dim, nb_blaschke, dtype=torch.float32))

        # Projection parameter: fixed or learnable
        self.projection = nn.Parameter(torch.empty(in_dim, out_dim, dtype=torch.float32), requires_grad=proj_learnable)

        # Initialize weights
        self.initialize_weights()

        # Set other parameters
        self.eps = eps
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nb_blaschke = nb_blaschke
        self.to(device)

    def __repr__(self):
        return (f"BlaschkeLayer1d("
                f"in_dim={self.in_dim}, "
                f"out_dim={self.out_dim})")

    def initialize_weights(self) -> None:
        """Initialize the weights of the BlaschkeLayer1d."""
        with torch.no_grad():
            self.alpha.uniform_(-1, 1)
            self.log_beta.uniform_(0, 1)
            # self.gamma.fill_(0.01)
            if self.proj_learnable:
                self.projection.uniform_(0, 1)
            else:
                self.projection.fill_(1)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Custom activation function used in the BlaschkeLayer1d.
        """
        output = torch.arctan(x) + torch.pi / 2
        return output

    def _compute_blaschke_factors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Blaschke factors ($\theta(x)$) for the input x.
        x shape: [B, L, C, 1]
        """
        frac = (x - self.alpha) / (torch.exp(self.log_beta) + self.eps)
        return torch.sum(self.activation(frac), dim=-1)

    def to(self, device: str):
        super().to(device)
        self.device = device
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        -----
            x : 3D torch.float
                inputs, shape [B, C_in, L] (number of samples, input dimension, length)

        Returns:
        --------
            y : 3D torch.float
                outputs, shape [B, C_out, L] (number of samples, output dimension, length)

        Example
        -------
        >>> model = BlaschkeLayer1d(in_dim=3, out_dim=5, nb_blaschke=2)
        >>> x = torch.normal(0, 1, size=(32, 3, 100))
        >>> y = model(x)
        """

        # [B, C_in, L]
        assert len(x.shape) == 3

        # Channel first to channel last. [B, L, C_in]
        x = x.transpose(1, 2)

        # Compute the projection. [B, L, C_out]
        x = x @ (self.projection / torch.sum(self.projection, dim=0))

        # Add a dimension for number of Blaschke components.
        x = x.unsqueeze(-1)

        # Compute the Blaschke factors $\theta(x)$. [B, L, C_out]
        blaschke_factors = self._compute_blaschke_factors(x)
        out = blaschke_factors

        # # Transform into the complex domain. [B, L, C_out]
        # out = torch.exp(1j * out)

        # Channel last to channel first. [B, C_out, L]
        out = out.transpose(1, 2)
        return out


# class Complex2Real(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return torch.real(x)


class BlaschkeNetwork1d(nn.Module):
    """
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
    """

    def __init__(self,
                 layers_hidden: List[int],
                 Blaschke_param: List[int] = None,
                 p_mul: int = 1,
                 start_p: int = 1,
                 eps: float = 1e-6,
                 device: str = 'cpu',
                 seed: int = 1,
                 proj_learnable: bool = False) -> None:

        super().__init__()
        torch.manual_seed(seed)
        self.layers_hidden = layers_hidden

        if Blaschke_param is None:
            # Determine Blaschke parameters if not provided
            self.Blaschke_param = [i * p_mul + start_p for i in range(len(self.layers_hidden)-2)]
        else:
            self.Blaschke_param = Blaschke_param

        # Initialize the BNLayers
        self.encoder = nn.ModuleList([])

        for (in_dim, out_dim, param) in zip(layers_hidden[:-1], layers_hidden[1:], self.Blaschke_param):
            self.encoder.append(nn.Sequential(
                BlaschkeLayer1d(in_dim=in_dim,
                                out_dim=out_dim,
                                nb_blaschke=param,
                                eps=eps,
                                device=device,
                                proj_learnable=proj_learnable),
                # Complex2Real(),
                nn.BatchNorm1d(out_dim),
            ))

        self.classifier = nn.Linear(layers_hidden[-2], layers_hidden[-1])

        self.to(device)

        print('Number of total parameters: ', count_parameters(self))
        print('Number of trainable parameters: ', count_parameters(self, trainable_only=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        -----
            x : 3D torch.float
                inputs, shape [B, C_in, L] (number of samples, input dimension, signal length)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape [B, C_out] (number of samples, output dimension)
        """

        # Input should be a [B, C_in, L] signal.
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        for layer in self.encoder:
            x = layer(x)

        # Global average pooling.
        x = x.mean(dim=(2))
        x = self.classifier(x)
        return x


    def to(self, device: str):
        super().to(device)
        self.device = device
        return self



if __name__ == '__main__':
    model = BlaschkeNetwork1d(layers_hidden=[3, 16, 64, 10])
    x = torch.normal(0, 1, size=(32, 3, 100))
    y = model(x)

