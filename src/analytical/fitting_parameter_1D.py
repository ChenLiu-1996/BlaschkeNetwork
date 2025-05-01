import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm import tqdm

import os
import sys
import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.insert(0, import_dir)
from analytical_decomposition import display_blaschke_product, complexify_signal


def reshape_and_complexify(signal: np.ndarray) -> np.ndarray:
    original_shape = signal.shape
    if len(original_shape) == 1:
        signal = rearrange(signal, '(c l) -> c l', c=1)
    elif len(original_shape) == 2:
        pass
    elif len(original_shape) == 3:
        signal = rearrange(signal, 'b c l -> (b c) l')
    else:
        raise ValueError(f'Currently only supporting 1D/2D/3D signals, but got shape {original_shape}.')

    signal_complex = complexify_signal(signal)
    return signal_complex


class BlaschkeParams(torch.nn.Module):
    '''
    This is a torch module that stores all Blaschke parameters.
    All Blaschke parameters are treated as learnable scalars.
    '''
    def __init__(self,
                 num_roots: int = 1,
                 seed: int = 1):
        super().__init__()
        torch.manual_seed(seed)

        self.num_roots = num_roots

        self.alphas = torch.nn.Parameter(torch.empty(self.num_roots))
        self.log_betas = torch.nn.Parameter(torch.empty(self.num_roots))
        self.logit_gammas = torch.nn.Parameter(torch.empty(self.num_roots))
        self.scale_real = torch.nn.Parameter(torch.empty(1))
        self.scale_imag = torch.nn.Parameter(torch.empty(1))

        torch.nn.init.normal_(self.alphas, mean=0.0, std=1e-2)
        torch.nn.init.normal_(self.log_betas, mean=0.0, std=1e-2)
        torch.nn.init.normal_(self.logit_gammas, mean=0.0, std=1e-2)
        torch.nn.init.normal_(self.scale_real, mean=0.0, std=1e-2)
        torch.nn.init.normal_(self.scale_imag, mean=0.0, std=1e-2)

    @property
    def betas(self):
        '''
        betas have to be positive.
        '''
        return torch.exp(self.log_betas)

    @property
    def gammas(self):
        '''
        gammas have to be between 0 and 1.
        '''
        return torch.sigmoid(self.logit_gammas)

    @property
    def scale(self):
        return self.scale_real + 1j * self.scale_imag

    def activation(self, signal: torch.Tensor) -> torch.Tensor:
        '''
        Custom activation function used in the BlaschkeLayer1d.
        '''
        output = torch.arctan(signal) + torch.pi / 2
        return output

    def compute_blaschke_factor(self,
                                signal_complex: torch.Tensor) -> torch.Tensor:

        # NOTE: Assuming B = 1, C = 1.

        signal_complex = signal_complex.unsqueeze(2)                           # [B, C, 1, L]
        t = rearrange(torch.linspace(-1, 1, signal_complex.shape[-1]),
                      'l -> 1 1 1 l').to(signal_complex.device)                # [1, 1, 1, L]
        alphas = rearrange(self.alphas, 'r -> 1 1 r 1')                        # [B, C, R, 1]
        betas = rearrange(self.betas, 'r -> 1 1 r 1')                          # [B, C, R, 1]
        gammas = rearrange(self.gammas, 'r -> 1 1 r 1')                        # [B, C, R, 1]

        activated = self.activation((t - alphas) / betas)                      # [B, C, R, L]
        phase = (gammas * activated).sum(dim=2)                                # sum over roots (R) → [B, C, L]

        blaschke_factor = torch.exp(1j * phase)
        return blaschke_factor

    def forward(self,
                signal_complex: torch.Tensor) -> torch.Tensor:
        # [B, C, L]
        assert len(signal_complex.shape) == 3

        # Compute the Blaschke product $B(x)$. [B, C, L]
        blaschke_factor = self.compute_blaschke_factor(signal_complex)
        return blaschke_factor


def plot_signal_approx(signal_complex: np.ndarray, scale: np.ndarray, blaschke_product: np.ndarray, blaschke_order: int = 6) -> None:
    fig, ax = plt.subplots(blaschke_order, blaschke_order + 2, figsize = (26, 16))
    if blaschke_order == 1:
        ax = ax[np.newaxis, :]

    time_arr = np.arange(signal_complex.shape[-1])
    signal_complex = signal_complex.squeeze(0)
    blaschke_product = blaschke_product.squeeze(1)

    for total_order in range(blaschke_order):
        ax[total_order, 0].plot(time_arr, signal_complex.real, label = 'original signal', color='firebrick', alpha=0.8)
        ax[total_order, 0].legend(loc='lower left')
        ax[total_order, 0].spines['top'].set_visible(False)
        ax[total_order, 0].spines['right'].set_visible(False)

    for total_order in range(1, blaschke_order + 1):
        for curr_order in range(1, total_order + 1):
            ax[total_order - 1, curr_order].hlines(np.abs(scale[curr_order-1]), xmin=time_arr.min(), xmax=time_arr.max(), label = f'$s_{curr_order}$', color='darkblue', linestyle='--')
            ax[total_order - 1, curr_order].plot(time_arr, (blaschke_product[curr_order-1] * scale[curr_order-1]).real, label = f'$s_{curr_order}$ * ${display_blaschke_product(curr_order)}$', color='darkgreen', alpha=0.6)
            ax[total_order - 1, curr_order].legend(loc='lower left')
            ax[total_order - 1, curr_order].spines['top'].set_visible(False)
            ax[total_order - 1, curr_order].spines['right'].set_visible(False)

    final = 0
    for curr_order in range(1, blaschke_order + 1):
        final += (blaschke_product[curr_order-1] * scale[curr_order-1]).real
        ax[curr_order - 1, blaschke_order + 1].plot(time_arr, signal_complex.real, label = 'original signal', color='firebrick', alpha=0.8)
        ax[curr_order - 1, blaschke_order + 1].plot(time_arr, final, label = 'reconstruction', color='skyblue', alpha=0.9)
        ax[curr_order - 1, blaschke_order + 1].plot(time_arr, final - signal_complex.real, label = 'residual', color='gray', alpha=1.0)
        ax[curr_order - 1, blaschke_order + 1].set_title(f'Reconstruction Error: {np.power(np.abs(final - signal_complex.real), 2).mean():.4f}')
        ax[curr_order - 1, blaschke_order + 1].legend(loc='lower left')
        ax[curr_order - 1, blaschke_order + 1].spines['top'].set_visible(False)
        ax[curr_order - 1, blaschke_order + 1].spines['right'].set_visible(False)

    # Remove axes from unused subplots
    for i in range(blaschke_order):
        for j in range(blaschke_order + 2):
            if (j > i + 1) and (j != blaschke_order + 1):
                ax[i, j].axis('off')

    fig.tight_layout(pad=2)
    fig.savefig('demo_fitting_parameter_1D.png')
    return

def run_sample(model_input: torch.Tensor, model: torch.nn.Module, blaschke_order: int, detach_by_iter:bool) -> torch.Tensor:
    blaschke_factors = []
    residual_signal, residual_sqnorm, gamma_deviation = model_input, None, None
    mean_scale, active_roots_ratio = None, None
    B_prod_arr = None

    for iter_idx in range(blaschke_order):
        blaschke_factors.append(model[iter_idx](residual_signal))

        # Blaschke product is the cumulative product of Blaschke factors
        # B_1 * B_2 * ... * B_n.
        blaschke_product = blaschke_factors[-1]
        for blaschke_factor in blaschke_factors[:-1]:
            if detach_by_iter:
                # Detach the gradient so that each iteration is penalized separately.
                blaschke_product = blaschke_product * blaschke_factor.detach()
            else:
                blaschke_product = blaschke_product * blaschke_factor

        # The Blaschke approximation is given by
        # s_n * B_1 * B_2 * ... * B_n.
        scale = model[iter_idx].scale
        curr_signal_approx = scale * blaschke_product

        # F_{n+1} = F_n - s_n * B_1 * ... * B_n.
        if detach_by_iter:
            # Detach the gradient so that each iteration is penalized separately.
            residual_signal = residual_signal.detach() - curr_signal_approx
        else:
            residual_signal = residual_signal - curr_signal_approx

        # This helps sanity checking the residual norms at each iteration.
        curr_sqnorm = torch.abs(residual_signal).pow(2).mean().unsqueeze(0)
        curr_deviation = model[iter_idx].gammas * (1 - model[iter_idx].gammas)
        if residual_sqnorm is None:
            residual_sqnorm = curr_sqnorm
            gamma_deviation = curr_deviation
        else:
            residual_sqnorm = torch.cat((residual_sqnorm, curr_sqnorm), dim=0)
            gamma_deviation = torch.cat((gamma_deviation, curr_deviation), dim=0)

        if B_prod_arr is None:
            B_prod_arr = blaschke_product.detach().cpu()
        else:
            B_prod_arr = torch.cat((B_prod_arr, blaschke_product.detach().cpu()), dim=0)

        # Track the mean scale per layer.
        if mean_scale is None:
            mean_scale = scale.mean().cpu().detach().unsqueeze(0)
        else:
            mean_scale = torch.cat((mean_scale, scale.mean().cpu().detach().unsqueeze(0)), dim=0)

        # Track the ratio of active roots per layer.
        active_roots = (model[iter_idx].logit_gammas > 0).sum().cpu().detach()
        total_roots = model[iter_idx].logit_gammas.numel()
        if active_roots_ratio is None:
            active_roots_ratio = (active_roots / total_roots).unsqueeze(0)
        else:
            active_roots_ratio = torch.cat((active_roots_ratio, (active_roots / total_roots).unsqueeze(0)), dim=0)

    return residual_sqnorm, gamma_deviation, B_prod_arr.numpy(), mean_scale.numpy(), active_roots_ratio.numpy()


if __name__ == '__main__':
    # Load the signal.
    test_signal_file = '../../data/gravitational_wave_HanfordH1.txt'
    signal_df = pd.read_csv(test_signal_file, sep=' ', header=None)
    time_arr = np.array(signal_df[0]) - np.min(signal_df[0])
    signal_arr = np.array(signal_df[1])

    # Parameters for Blaschke decomposition.
    blaschke_order = 6                       # Using 6 for best visualization.
    num_roots = 32
    num_epochs = 3000
    learning_rate = 1e-2
    detach_by_iter = False
    coeff_binary = 10                        # Encourages root selectors (gammas) to be binary.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    signal_complex = reshape_and_complexify(signal=signal_arr)

    model = torch.nn.ModuleList()
    for _ in range(blaschke_order):
        model.append(BlaschkeParams(num_roots=num_roots))
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model_input = torch.from_numpy(rearrange(signal_complex, '(b c) l -> b c l', b=1)).to(device)

    with tqdm(range(num_epochs)) as pbar:
        for epoch_idx in pbar:

            residual_sqnorm, gamma_deviation, blaschke_product, mean_scale, active_roots_ratio = \
                run_sample(model_input=model_input, model=model, blaschke_order=blaschke_order, detach_by_iter=detach_by_iter)
            loss_recon = residual_sqnorm.mean()
            loss_binary = gamma_deviation.mean()
            loss = loss_recon + loss_binary * coeff_binary

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss_recon=f'{loss_recon:.5f}',
                             loss_binary_scaled=f'{loss_binary * coeff_binary:.5f}',
                             active_roots_ratio=f'{active_roots_ratio}')

    plot_signal_approx(signal_complex=signal_complex,
                       scale=[model[iter_idx].scale.cpu().detach().numpy() for iter_idx in range(blaschke_order)],
                       blaschke_product=blaschke_product,
                       blaschke_order=blaschke_order)
