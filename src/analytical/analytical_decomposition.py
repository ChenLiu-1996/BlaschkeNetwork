import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def decompose_blaschke_factors(high_freq_component: np.ndarray,
                                fourier_poly_order: int,
                                oversampling_rate: int,
                                eps: float):
    '''
    high_freq_component: the high-frequency component of the signal. Assume shape (signal_len,)
    fourier_poly_order: the degree of Fourier polynomial. Used to truncated the functions B & G.
    oversampling_rate: oversampling rate
    eps: value of the threshold to cut the signal
    '''

    num_channels, signal_len = high_freq_component.shape
    half_signal_len = signal_len // 2
    time_idx = np.arange(1, oversampling_rate * signal_len + 1)

    # Oversampling.
    high_freq_fft = np.fft.fft(high_freq_component)
    high_freq_fft_oversampled = np.zeros((num_channels, signal_len * oversampling_rate), dtype=complex)
    high_freq_fft_oversampled[:, 0:half_signal_len] = high_freq_fft[:, 0 :half_signal_len] * oversampling_rate
    high_freq_oversampled = np.fft.ifft(high_freq_fft_oversampled)

    # Step 2: evaluate G
    # compute log(abs(Hi)) of the analytic function Hi
    signal_amplitude = np.abs(high_freq_oversampled)
    if np.min(signal_amplitude) < eps * np.max(signal_amplitude):
        # Avoid division by zero
        eps2 = (eps * np.max(signal_amplitude))**2
        log_abs_high_freq = 0.5 * np.log(signal_amplitude**2 + eps2)
    else:
        log_abs_high_freq = np.log(signal_amplitude)

    # Ana_log_abs_Hi = P_+(ln|Hi|)
    mean_signal = np.mean(log_abs_high_freq)
    Ana_log_abs_F = mean_signal + 2 * np.fft.ifft(np.fft.fft(log_abs_high_freq - mean_signal) * (time_idx < (fourier_poly_order + 1)))

    G_oversampled = np.fft.ifft(np.fft.fft(np.exp(Ana_log_abs_F)) * (time_idx < (fourier_poly_order + 1)))
    blaschke_factor_oversampled = high_freq_oversampled / G_oversampled

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

def blaschke_decomposition(signal: np.ndarray,
                           time: np.ndarray,
                           num_blaschke_iters: int,
                           fourier_poly_order: int,
                           oversampling_rate: int,
                           lowpass_order: int,
                           carrier_freq: float,
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
    F(z) = s_1 B_1 + s_2 B_1 B_2 + ...

    NOTE:
    `blaschke_factor` at each iteration corresponds to `B_i`.
    `curr_G` at each iteration corresponds to `G_i`.
    When lowpass_order = 1, `low_freq_component` at each iteration will correspond to `s_i`.

    signal: the input signal. Assume shape (signal_len,). CUrrently only support 1-channel signals.
    time: the input time. Assume shape (signal_len,)
    num_blaschke_iters: number of iterations of Blaschke decomposition. This gives (num_blaschke_iters - 1) Blaschke components.
    fourier_poly_order: the degree of Fourier polynomial. Used to truncated the functions B & G.
    oversampling_rate: oversampling rate
    lowpass_order: the order of the low pass filter
    carrier_freq: carrier frequency
    eps: value of the threshold to cut the signal
    '''

    # Complexify the signal with Hilbert transform after removing zero-order drift.
    signal = hilbert(signal - np.mean(signal))
    # Frequency shifting by carrier frequency.
    signal = signal * np.exp(1j * 2 * np.pi * carrier_freq * time)
    # Only keep the non-negative frequencies.
    signal_conjugate_symmetric = np.concatenate((signal, np.fliplr(np.conj(signal).reshape(1, -1)).squeeze()))
    mask_nonnegative_freq = np.ones(len(signal_conjugate_symmetric))
    mask_nonnegative_freq[int(len(signal_conjugate_symmetric)/2):] = 0
    signal = np.fft.ifft(np.fft.fft(signal_conjugate_symmetric) * mask_nonnegative_freq)
    signal = signal.reshape(1, -1)

    num_channels, signal_len = signal.shape

    if signal_len == 1:
        raise ValueError('The signal must be saved as a row')
    if min(num_channels, signal_len) > 1:
        raise ValueError('The code only supports one channel signal right now')

    # First iteration of decomposition.
    high_freq_component, low_freq_component = decompose_by_frequency(signal, fourier_poly_order, oversampling_rate, lowpass_order)
    curr_blaschke_factor, curr_G = decompose_blaschke_factors(high_freq_component, fourier_poly_order, oversampling_rate, eps)
    blaschke_product = curr_blaschke_factor
    blaschke_factor = curr_blaschke_factor

    # All remaining iterations.
    for _ in range(1, num_blaschke_iters):
        curr_high_freq_component, curr_low_freq_component = decompose_by_frequency(curr_G.reshape(1, -1), fourier_poly_order, oversampling_rate, lowpass_order)
        curr_blaschke_factor, curr_G = decompose_blaschke_factors(curr_high_freq_component, fourier_poly_order, oversampling_rate, eps)

        low_freq_component = np.concatenate((low_freq_component, curr_low_freq_component), axis=0)
        blaschke_factor = np.concatenate((blaschke_factor, curr_blaschke_factor), axis=0)
        blaschke_product = np.concatenate((blaschke_product, blaschke_product[-1] * curr_blaschke_factor), axis=0)

    low_freq_component = low_freq_component[:, :int(low_freq_component.shape[1]/2)]
    blaschke_product = blaschke_product[:, :int(blaschke_product.shape[1]/2)] * np.tile(np.exp(-1j * 2 * np.pi * carrier_freq * time_arr), (num_blaschke_iters, 1))

    return low_freq_component, blaschke_factor, blaschke_product


if __name__ == '__main__':
    # Load the signal.
    test_signal_file = '../../data/gravitational_wave_HanfordH1.txt'
    signal_df = pd.read_csv(test_signal_file, sep=' ', header=None)
    time_arr = np.array(signal_df[0])
    signal_arr = np.array(signal_df[1])

    # Parameters for Blaschke decomposition.
    num_blaschke_iters = 7
    oversampling_rate = 16
    lowpass_order = 1
    blaschke_order = num_blaschke_iters - 1
    carrier_freq = 10

    # Blaschke decomposition.
    low_freq_component, _, blaschke_product = blaschke_decomposition(
        signal=signal_arr,
        time=time_arr,
        num_blaschke_iters=num_blaschke_iters,
        fourier_poly_order=len(signal_arr),
        oversampling_rate=oversampling_rate,
        lowpass_order=lowpass_order,
        carrier_freq=carrier_freq)

    fig, ax = plt.subplots(blaschke_order, blaschke_order + 2, figsize = (24, 16))
    for total_order in range(blaschke_order):
        ax[total_order, 0].plot(time_arr, signal_arr, label = 'original signal', color='firebrick', alpha=0.8)
        ax[total_order, 0].legend(loc='lower left')

    for total_order in range(1, blaschke_order + 1):
        for curr_order in range(1, total_order + 1):
            ax[total_order - 1, curr_order].plot(time_arr, low_freq_component[curr_order].real, label = f'$G_{curr_order}$', color='black')
            ax[total_order - 1, curr_order].plot(time_arr, (blaschke_product[curr_order-1] * low_freq_component[curr_order]).real, label = f'$G_{curr_order}$ * prod(..., $B_{curr_order}$)', color='green', alpha=0.6)
            ax[total_order - 1, curr_order].legend(loc='lower left')

    final = 0
    for curr_order in range(1, blaschke_order + 1):
        final += (blaschke_product[curr_order-1] * low_freq_component[curr_order]).real
        ax[curr_order - 1, blaschke_order + 1].plot(time_arr, signal_arr, label = 'original signal', color='firebrick', alpha=0.8)
        ax[curr_order - 1, blaschke_order + 1].plot(time_arr, final, label = 'reconstruction', color='skyblue', alpha=0.9)
        ax[curr_order - 1, blaschke_order + 1].plot(time_arr, final - signal_arr, label = 'residual', color='gray', alpha=1.0)
        ax[curr_order - 1, blaschke_order + 1].legend(loc='lower left')

    # Remove axes from unused subplots
    for i in range(blaschke_order):
        for j in range(blaschke_order + 2):
            if (j > i + 1) and (j != blaschke_order + 1):
                ax[i, j].axis('off')

    fig.tight_layout(pad=2)
    fig.savefig('demo_blaschke_decomposition.png')
