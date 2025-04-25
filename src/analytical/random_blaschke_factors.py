import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from einops import rearrange
from matplotlib import pyplot as plt

from analytical_decomposition import display_blaschke_product, complexify_signal


def plot_random_blaschke(signal_complex, s_arr, B_prod_arr):
    assert len(signal_complex.shape) == 3
    assert len(s_arr.shape) == len(B_prod_arr.shape) == 4
    assert signal_complex.shape[0] == 1
    assert signal_complex.shape[1] == B_prod_arr.shape[1] == 1
    assert signal_complex.shape[2] == B_prod_arr.shape[2]
    blaschke_order = B_prod_arr.shape[-1]

    signal_complex = rearrange(signal_complex, 'b c l -> (b c) l').squeeze(0)
    s_arr = rearrange(s_arr, 'b c l i -> (b c l) i').squeeze(0)
    B_prod_arr = rearrange(B_prod_arr, 'b c l i -> (b c) l i').squeeze(0)

    fig, ax = plt.subplots(blaschke_order, blaschke_order + 2, figsize = (4 * blaschke_order + 8, 4 * blaschke_order))
    if blaschke_order == 1:
        ax = ax[np.newaxis, :]
    time_arr = np.arange(signal_complex.shape[-1])
    for total_order in range(blaschke_order):
        ax[total_order, 0].plot(time_arr, signal_complex.real, label = 'original signal', color='firebrick', alpha=0.8)
        ax[total_order, 0].legend(loc='lower left')
        ax[total_order, 0].spines['top'].set_visible(False)
        ax[total_order, 0].spines['right'].set_visible(False)

    for total_order in range(1, blaschke_order + 1):
        for curr_order in range(1, total_order + 1):
            ax[total_order - 1, curr_order].hlines(s_arr[curr_order-1].real, xmin=time_arr.min(), xmax=time_arr.max(), label = f'$s_{curr_order}$', color='darkblue', linestyle='--')
            ax[total_order - 1, curr_order].plot(time_arr, (B_prod_arr[:, curr_order-1] * s_arr[curr_order-1]).real, label = f'$s_{curr_order}$ * ${display_blaschke_product(curr_order)}$', color='darkgreen', alpha=0.6)
            ax[total_order - 1, curr_order].legend(loc='lower left')
            ax[total_order - 1, curr_order].spines['top'].set_visible(False)
            ax[total_order - 1, curr_order].spines['right'].set_visible(False)

    final = 0
    for curr_order in range(1, blaschke_order + 1):
        final += (B_prod_arr[:, curr_order-1] * s_arr[curr_order-1]).real
        # ax[curr_order - 1, blaschke_order + 1].plot(time_arr, signal_complex.real, label = 'original signal', color='firebrick', alpha=0.8)
        ax[curr_order - 1, blaschke_order + 1].plot(time_arr, final, label = 'reconstruction', color='skyblue', alpha=0.9)
        # ax[curr_order - 1, blaschke_order + 1].plot(time_arr, final - signal_complex.real, label = 'residual', color='gray', alpha=1.0)
        ax[curr_order - 1, blaschke_order + 1].legend(loc='lower left')
        ax[curr_order - 1, blaschke_order + 1].spines['top'].set_visible(False)
        ax[curr_order - 1, blaschke_order + 1].spines['right'].set_visible(False)

    # Remove axes from unused subplots
    for i in range(blaschke_order):
        for j in range(blaschke_order + 2):
            if (j > i + 1) and (j != blaschke_order + 1):
                ax[i, j].axis('off')

    save_path = os.path.join('demo_random_blaschke_factors.png')
    fig.tight_layout(pad=2)
    fig.savefig(save_path)
    plt.close(fig)
    return


def activation(signal):
    return np.arctan(signal) + np.pi / 2


if __name__ == '__main__':
    test_signal_file = '../../data/gravitational_wave_HanfordH1.txt'
    signal_df = pd.read_csv(test_signal_file, sep=' ', header=None)
    signal_real = np.array(signal_df[1])
    signal_real = rearrange(signal_real, '(c l) -> c l', c=1)
    signal_complex = complexify_signal(signal_real)
    signal_complex = rearrange(signal_complex, '(b c) l -> b c l', b=1)

    blaschke_order = 4
    num_roots = 256

    random_seed = 1
    np.random.seed(random_seed)
    alpha = np.random.randn(*signal_complex.shape[:2], num_roots, blaschke_order)
    beta = np.random.randn(*signal_complex.shape[:2], num_roots, blaschke_order)

    s_arr = np.random.randn(*signal_complex.shape[:2], 1, blaschke_order) \
            + 1j * np.random.randn(*signal_complex.shape[:2], 1, blaschke_order)

    B_factors = []
    B_prod_arr = None
    for i in range(blaschke_order):
        t = rearrange(np.linspace(0, 1, signal_complex.shape[-1]), 'l -> 1 1 1 l')
        acts = activation((t - alpha[..., i][..., None]) / beta[..., i][..., None]) # [B, C, R, L]
        phase = acts.sum(axis=2)                                                    # sum over roots (R) → [B, C, L]
        B = np.exp(1j * phase)
        B_factors.append(B)

        blaschke_product = 1
        for blaschke_factor in B_factors:
            blaschke_product = blaschke_product * blaschke_factor

        if B_prod_arr is None:
            B_prod_arr = blaschke_product[..., None]
        else:
            B_prod_arr = np.concatenate((B_prod_arr, blaschke_product[..., None]), axis=-1)

    plot_random_blaschke(signal_complex=signal_complex, s_arr=s_arr, B_prod_arr=B_prod_arr)