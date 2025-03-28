import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from einops import rearrange


def decompose_blaschke_factors(signal: np.ndarray,
                               fourier_poly_order: int,
                               oversampling_rate: int,
                               eps: float):
    '''
    signal: Assume shape (signal_len,)
    fourier_poly_order: the degree of Fourier polynomial. Used to truncated the functions B & G.
    oversampling_rate: oversampling rate
    eps: value of the threshold to cut the signal
    '''

    num_channels, signal_len = signal.shape
    half_signal_len = signal_len // 2
    time_idx = np.arange(1, oversampling_rate * signal_len + 1)

    # Oversampling.
    signal_fft = np.fft.fft(signal)
    signal_fft_oversampled = np.zeros((num_channels, signal_len * oversampling_rate), dtype=complex)
    signal_fft_oversampled[:, 0:half_signal_len] = signal_fft[:, 0 :half_signal_len] * oversampling_rate
    signal_oversampled = np.fft.ifft(signal_fft_oversampled)

    # Compute G and B.
    signal_amplitude = np.abs(signal_oversampled)
    if np.min(signal_amplitude) < eps * np.max(signal_amplitude):
        # Avoid division by zero
        eps2 = (eps * np.max(signal_amplitude))**2
        log_abs_signal = 0.5 * np.log(signal_amplitude**2 + eps2)
    else:
        log_abs_signal = np.log(signal_amplitude)

    mean_signal = np.mean(log_abs_signal)
    log_abs_high_freq = mean_signal + 2 * np.fft.ifft(np.fft.fft(log_abs_signal - mean_signal) * (time_idx < (fourier_poly_order + 1)))

    G_oversampled = np.fft.ifft(np.fft.fft(np.exp(log_abs_high_freq)) * (time_idx < (fourier_poly_order + 1)))
    blaschke_factor_oversampled = signal_oversampled / G_oversampled

    G = np.zeros((num_channels, signal_len), dtype = complex)
    blaschke_factor = np.zeros((num_channels, signal_len), dtype = complex)
    G[:, 0 : signal_len] = G_oversampled[:, 0 : signal_len * oversampling_rate :oversampling_rate]
    blaschke_factor[:, 0 : signal_len] = blaschke_factor_oversampled[:, 0 : signal_len * oversampling_rate :oversampling_rate]

    return blaschke_factor, G


def decompose_by_frequency(signal: np.ndarray,
                           fourier_poly_order: int,
                           oversampling_rate: int,
                           lowpass_order: int):
    '''
    Isolate the high and low frequency components.

    signal: the input signal. Assume shape (signal_len,)
    fourier_poly_order: the degree of Fourier polynomial. Used to truncated the functions B & G.
    oversampling_rate: oversampling rate
    lowpass_order: the order of the low pass filter
    '''

    num_channels, signal_len = signal.shape
    half_signal_len = signal_len // 2
    time_idx = np.arange(oversampling_rate * signal_len) + 1

    # Isolation of high and low frequency components.
    mask_high_freq = (time_idx > lowpass_order) * (time_idx < (fourier_poly_order + 1))
    mask_low_freq = (time_idx < (lowpass_order + 1))

    signal_fft = np.fft.fft(signal, axis=1)
    signal_fft_oversampled = np.zeros((num_channels, signal_len * oversampling_rate), dtype=complex)
    signal_fft_oversampled[:, :half_signal_len] = signal_fft[:, :half_signal_len] * oversampling_rate

    high_freq_component_oversampled = np.fft.ifft(signal_fft_oversampled * mask_high_freq, axis=1)
    low_freq_component_oversampled = np.fft.ifft(signal_fft_oversampled * mask_low_freq, axis=1)

    high_freq_component = np.zeros((num_channels, signal_len),dtype=complex)
    low_freq_component = np.zeros((num_channels, signal_len), dtype=complex)
    high_freq_component[:, :signal_len] = high_freq_component_oversampled[:, :signal_len*oversampling_rate:oversampling_rate]
    low_freq_component[:, :signal_len] = low_freq_component_oversampled[:, :signal_len*oversampling_rate:oversampling_rate]

    return high_freq_component, low_freq_component


def complexify_signal(signal: np.ndarray, carrier_freq: float = 0) -> np.ndarray:
    '''
    Complexify the signal with Hilbert transform after removing zero-order drift
    '''
    assert len(signal.shape) == 2

    signal = hilbert(signal - np.mean(signal, axis=1, keepdims=True))
    # Frequency shifting by carrier frequency.
    time_indices = np.arange(signal.shape[-1])
    signal = signal * np.exp(1j * 2 * np.pi * carrier_freq * time_indices)
    # Mitigate boundary effects. This is a common approach when performing Fourier analyses, spectral filtering, etc.
    signal_boundary_smoothed = np.concatenate((signal, np.fliplr(np.conj(signal))), axis=1)
    mask_nonnegative_freq = np.ones_like(signal_boundary_smoothed)
    mask_nonnegative_freq[:, mask_nonnegative_freq.shape[1] // 2:] = 0
    signal = np.fft.ifft(np.fft.fft(signal_boundary_smoothed) * mask_nonnegative_freq)
    signal = signal[:, :signal.shape[1] // 2]
    return signal


def blaschke_decomposition(signal: np.ndarray,
                           num_blaschke_iters: int,
                           fourier_poly_order: int,
                           oversampling_rate: int = 2,
                           lowpass_order: int = 1,
                           carrier_freq: float = 0,
                           eps: float = 1e-4):
    '''
    Blaschke decomposition.

    F(z) = B_1(z) G_1(z) = B_1(z) (G_1(0) + G_1(z) - G_1(0))
    * Let B_2(z) G_2(z) = G_1(z) - G_1(0)
    F(z) = B_1(z) (G_1(0) + B_2(z) G_2(z))
         = G_1(0) B_1(z) + G_2(z) B_1(z) B_2(z)
         = G_1(0) B_1(z) + G_2(0) B_1(z) B_2(z) + G_3(z) B_1(z) B_2(z) B_3(z)
         = ...
    * Let s_1 = G_1(0), s_2 = G_2(0), ...
    F(z) = s_1 B_1(z) + s_2 B_1(z) B_2(z) + ...

    NOTE:
    `blaschke_factor` at each iteration corresponds to `B_i(z)`.
    `curr_G` at each iteration corresponds to `G_i(z)`.
    When lowpass_order = 1, `low_freq_component` at each iteration will correspond to `s_i`.

    signal: the input signal.
        Assume shape (signal_len,) or (num_channels, signal_len) or (batch_size, num_channels, signal_len).
    num_blaschke_iters: number of iterations of Blaschke decomposition. This gives (num_blaschke_iters - 1) Blaschke components.
    fourier_poly_order: the degree of Fourier polynomial. Used to truncated the functions B & G.
    oversampling_rate: oversampling rate
    lowpass_order: the order of the low pass filter
    carrier_freq: carrier frequency
    eps: value of the threshold to cut the signal
    '''

    original_shape = signal.shape
    if len(original_shape) == 1:
        batch_size_times_num_channel = 1
        signal = rearrange(signal, '(c l) -> c l', c=1)
    elif len(original_shape) == 2:
        batch_size_times_num_channel = original_shape[0]
        pass
    elif len(original_shape) == 3:
        batch_size_times_num_channel = original_shape[0] * original_shape[1]
        signal = rearrange(signal, 'b c l -> (b c) l')
    else:
        raise ValueError(f'Currently only supporting 1D/2D/3D signals, but got shape {original_shape}.')

    signal = complexify_signal(signal=signal, carrier_freq=carrier_freq)

    # First iteration of decomposition.
    curr_high_freq_component, curr_low_freq_component = decompose_by_frequency(signal, fourier_poly_order, oversampling_rate, lowpass_order)
    curr_blaschke_factor, curr_G = decompose_blaschke_factors(curr_high_freq_component, fourier_poly_order, oversampling_rate, eps)
    low_freq_component = curr_low_freq_component[None, ...]
    blaschke_factor = curr_blaschke_factor[None, ...]
    blaschke_product = curr_blaschke_factor[None, ...]

    # All remaining iterations.
    for _ in range(1, num_blaschke_iters):
        curr_high_freq_component, curr_low_freq_component = decompose_by_frequency(curr_G, fourier_poly_order, oversampling_rate, lowpass_order)
        curr_blaschke_factor, curr_G = decompose_blaschke_factors(curr_high_freq_component, fourier_poly_order, oversampling_rate, eps)

        low_freq_component = np.concatenate((low_freq_component, curr_low_freq_component[None, ...]), axis=0)
        blaschke_factor = np.concatenate((blaschke_factor, curr_blaschke_factor[None, ...]), axis=0)
        blaschke_product = np.concatenate((blaschke_product, blaschke_product[-1, :, :][None, ...] * curr_blaschke_factor[None, ...]), axis=0)

    time_indices = np.arange(signal.shape[-1])
    blaschke_product = blaschke_product * np.tile(np.exp(-1j * 2 * np.pi * carrier_freq * time_indices), (num_blaschke_iters, batch_size_times_num_channel, 1))

    # NOTE: Now, the shapes of `low_freq_component`, `blaschke_factor`, `blaschke_product` are the same:
    # [num_blaschke_iters, batch_size * num_channel, signal_len].

    if len(original_shape) == 1:
        low_freq_component = rearrange(low_freq_component, 'i 1 l -> i l')
        blaschke_factor = rearrange(blaschke_factor, 'i 1 l -> i l')
        blaschke_product = rearrange(blaschke_product, 'i 1 l -> i l')
    elif len(original_shape) == 2:
        pass
    elif len(original_shape) == 3:
        low_freq_component = rearrange(low_freq_component, 'i (b c) l -> i b c l', b=original_shape[0], c=original_shape[1])
        blaschke_factor = rearrange(blaschke_factor, 'i (b c) l -> i b c l', b=original_shape[0], c=original_shape[1])
        blaschke_product = rearrange(blaschke_product, 'i (b c) l -> i b c l', b=original_shape[0], c=original_shape[1])

    return low_freq_component, blaschke_factor, blaschke_product

def display_blaschke_product(order: int):
    '''
    A helper function to print the blaschke product in cumulative product.
    '''
    blaschke_product_str = ''
    if order == 1:
        blaschke_product_str += 'B_1'
    elif order == 2:
        blaschke_product_str += '(B_1 * B_2)'
    elif order == 3:
        blaschke_product_str += '(B_1 * B_2 * B_3)'
    else:
        blaschke_product_str += f'(B_1 * B_2 *...* B_{order})'
    return blaschke_product_str


if __name__ == '__main__':
    # Load the signal.
    test_signal_file = '../../data/gravitational_wave_HanfordH1.txt'
    signal_df = pd.read_csv(test_signal_file, sep=' ', header=None)
    time_arr = np.array(signal_df[0]) - np.min(signal_df[0])
    signal_arr = np.array(signal_df[1])

    # Parameters for Blaschke decomposition.
    num_blaschke_iters = 7                   # Using 7 for best visualization.
    oversampling_rate = 2                    # At least 2, otherwise will have undersampling issue.
    lowpass_order = 1                        # Has to be 1 to parameterize `s_i`. Higher gives better approximation.
    blaschke_order = num_blaschke_iters - 1  # By definition. Do not change.
    carrier_freq = 0                         # Seems unnecessary.

    # Blaschke decomposition.
    low_freq_component, _, blaschke_product = blaschke_decomposition(
        signal=signal_arr,
        num_blaschke_iters=num_blaschke_iters,
        fourier_poly_order=signal_arr.shape[-1],
        oversampling_rate=oversampling_rate,
        lowpass_order=lowpass_order,
        carrier_freq=carrier_freq)

    fig, ax = plt.subplots(blaschke_order, blaschke_order + 2, figsize = (26, 16))
    signal_arr = signal_arr.reshape(-1)  # For plotting purposes.
    for total_order in range(blaschke_order):
        ax[total_order, 0].plot(time_arr, signal_arr, label = 'original signal', color='firebrick', alpha=0.8)
        ax[total_order, 0].legend(loc='lower left')
        ax[total_order, 0].spines['top'].set_visible(False)
        ax[total_order, 0].spines['right'].set_visible(False)

    for total_order in range(1, blaschke_order + 1):
        for curr_order in range(1, total_order + 1):
            ax[total_order - 1, curr_order].plot(time_arr, low_freq_component[curr_order].real, label = f'$G_{curr_order}$', color='darkblue', linestyle='--')
            ax[total_order - 1, curr_order].plot(time_arr, (blaschke_product[curr_order-1] * low_freq_component[curr_order]).real, label = f'$G_{curr_order}$ * ${display_blaschke_product(curr_order)}$', color='darkgreen', alpha=0.6)
            ax[total_order - 1, curr_order].legend(loc='lower left')
            ax[total_order - 1, curr_order].spines['top'].set_visible(False)
            ax[total_order - 1, curr_order].spines['right'].set_visible(False)

    final = 0
    for curr_order in range(1, blaschke_order + 1):
        final += (blaschke_product[curr_order-1] * low_freq_component[curr_order]).real
        ax[curr_order - 1, blaschke_order + 1].plot(time_arr, signal_arr, label = 'original signal', color='firebrick', alpha=0.8)
        ax[curr_order - 1, blaschke_order + 1].plot(time_arr, final, label = 'reconstruction', color='skyblue', alpha=0.9)
        ax[curr_order - 1, blaschke_order + 1].plot(time_arr, final - signal_arr, label = 'residual', color='gray', alpha=1.0)
        ax[curr_order - 1, blaschke_order + 1].legend(loc='lower left')
        ax[curr_order - 1, blaschke_order + 1].spines['top'].set_visible(False)
        ax[curr_order - 1, blaschke_order + 1].spines['right'].set_visible(False)

    # Remove axes from unused subplots
    for i in range(blaschke_order):
        for j in range(blaschke_order + 2):
            if (j > i + 1) and (j != blaschke_order + 1):
                ax[i, j].axis('off')

    fig.tight_layout(pad=2)
    fig.savefig('demo_blaschke_decomposition.png')
