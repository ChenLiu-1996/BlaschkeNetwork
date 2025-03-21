import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def getBG(signal: np.ndarray,
          fourier_poly_order, oversampling_rate, eps):
    '''
    signal: the input signal. Assume shape (signal_len,)
    fourier_poly_order: the degree of Fourier polynomial. Used to truncated the functions B & G.
    oversampling_rate: oversampling rate
    eps: value of the threshold to cut the signal
    '''
    # get the dimensions of the input signal
    (num_channels, signal_len) = signal.shape

    # get the index of the halfway point of the signal
    half_signal_len = signal_len // 2

    # create arrays for indexing
    unit_m = np.arange(1, oversampling_rate * signal_len + 1)

    # Plus oversampling by over
    mask_N = (unit_m < (fourier_poly_order + 1))

    signal_fft = np.fft.fft(signal)
    f_fft_m = np.zeros((num_channels, signal_len * oversampling_rate), dtype=complex)
    f_fft_m[:, 0:half_signal_len] = signal_fft[:, 0 :half_signal_len] * oversampling_rate

    f_ana_m = np.fft.ifft(f_fft_m)

    # Step 2: evaluate G
    # compute log(abs(Hi)) of the analytic function Hi
    f_abs_m = np.abs(f_ana_m)
    if np.min(f_abs_m) < eps * np.max(f_abs_m):
        # add eps to avoid division by zero
        eps2 = (eps * np.max(f_abs_m))**2
        log_abs_f = 0.5 * np.log(f_abs_m**2 + eps2)
    else:
        eps2 = 0

        log_abs_f = np.log(f_abs_m)

    # Ana_log_abs_Hi = P_+(ln|Hi|)
    m = np.mean(log_abs_f)
    Ana_log_abs_F = m + 2 * np.fft.ifft(np.fft.fft(log_abs_f - m) * mask_N)

    # filter the exp_plus
    G_ana_m = np.fft.ifft(np.fft.fft(np.exp(Ana_log_abs_F)) * mask_N)

    # Step 3: evaluate B
    B_ana_m = f_ana_m / G_ana_m

    # Step 5: from signal oversample creates signal with original size
    B_ana = np.zeros((num_channels, signal_len), dtype = complex)
    G_ana = np.zeros((num_channels, signal_len), dtype = complex)

    B_ana[:, 0 : signal_len] = B_ana_m[:, 0 : signal_len * oversampling_rate :oversampling_rate]
    G_ana[:, 0 : signal_len] = G_ana_m[:, 0 : signal_len * oversampling_rate :oversampling_rate]

    return B_ana, G_ana


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

    l, ll = signal.shape
    ll2 = ll // 2
    unit_m = np.arange(oversampling_rate * ll) + 1

    # Step 1: mask for hilbert transform with degre D, for low frequency part.
    # we suppose that f is already analytic (support freq pos)

    # design the filter

    mask_N = (unit_m < (fourier_poly_order + 1))
    maskLow = (unit_m < (lowpass_order + 1))
    maskHigh = (unit_m > lowpass_order) * mask_N

    # Filter in the fourier space with the 2 masks
    # Plus oversampling by over
    f_fft = np.fft.fft(signal, axis=1)
    f_fft_m = np.zeros((l, ll * oversampling_rate), dtype=complex)
    f_fft_m[:, :ll2] = f_fft[:, :ll2] * oversampling_rate

    # Low_ana is low freq
    Low_ana_m = np.fft.ifft(f_fft_m * maskLow, axis=1)
    # High_ana is high freq
    High_ana_m = np.fft.ifft(f_fft_m * maskHigh, axis=1)

    High_ana = np.zeros((l, ll),dtype=complex)
    Low_ana = np.zeros((l, ll), dtype=complex)

    High_ana[:, :ll] = High_ana_m[:, :ll*oversampling_rate:oversampling_rate]
    Low_ana[:, :ll] = Low_ana_m[:, :ll*oversampling_rate:oversampling_rate]

    return High_ana, Low_ana

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

    F(z) = B(z) G(z) = B(z) (G(0) + G(z) - G(0))
    * Let B_1(z) G_1(z) = G(z) - G(0)
    F(z) = B(z) (G(0) + B_1(z) G_1(z)) = G(0) B(z) + G_1(z) B(z) B_1(z)
    * Let s_1 = G(0), s_2 = G_1(z), ...
    F(z) = s_1 B_1 + s_2 B_1 B_2 + ...

    NOTE: When lowpass_order = 1, `low_freq_component` at each iteration will correspond to `s_i`.
    `blaschke_product` at each iteration corresponds to `B_i`.

    signal: the input signal. Assume shape (signal_len,)
    time: the input time. Assume shape (signal_len,)
    num_blaschke_iters: number of iterations of Blaschke decomposition. This gives (num_blaschke_iters - 1) Blaschke components.
    fourier_poly_order: the degree of Fourier polynomial. Used to truncated the functions B & G.
    oversampling_rate: oversampling rate
    lowpass_order: the order of the low pass filter
    carrier_freq: carrier frequency
    eps: value of the threshold to cut the signal
    '''

    signal = hilbert(signal - np.mean(signal))
    signal = signal * np.exp(1j * 2 * np.pi * carrier_freq * time)
    signal_conjugate_symmetric = np.concatenate((signal, np.fliplr(np.conj(signal).reshape(1, -1)).squeeze()))
    mask = np.ones(len(signal_conjugate_symmetric))
    mask[int(len(signal_conjugate_symmetric)/2):] = 0
    signal = np.fft.ifft(np.fft.fft(signal_conjugate_symmetric) * mask)
    signal = signal.reshape(1, -1)

    num_channels, signal_len = signal.shape

    if signal_len == 1:
        raise ValueError('The signal must be saved as a row')
    if min(num_channels, signal_len) > 1:
        raise ValueError('The code only supports one channel signal right now')

    # First iteration of decomposition.
    high_freq_component, low_freq_component = decompose_by_frequency(signal, fourier_poly_order, oversampling_rate, lowpass_order)
    blaschke_product, G_ana = getBG(high_freq_component, fourier_poly_order, oversampling_rate, eps)
    cumulative_blaschke_product = blaschke_product

    # All remaining iterations.
    for _ in range(1, num_blaschke_iters):
        curr_high_freq_component, curr_low_freq_component = decompose_by_frequency(G_ana.reshape(1, -1), fourier_poly_order, oversampling_rate, lowpass_order)
        curr_blaschke_product, G_ana = getBG(curr_high_freq_component, fourier_poly_order, oversampling_rate, eps)

        low_freq_component = np.concatenate((low_freq_component, curr_low_freq_component), axis=0)
        blaschke_product = np.concatenate((blaschke_product, curr_blaschke_product), axis=0)
        cumulative_blaschke_product = np.concatenate((cumulative_blaschke_product, cumulative_blaschke_product[-1] * curr_blaschke_product), axis=0)

    low_freq_component = low_freq_component[:, :int(low_freq_component.shape[1]/2)]
    cumulative_blaschke_product = cumulative_blaschke_product[:, :int(cumulative_blaschke_product.shape[1]/2)] * np.tile(np.exp(-1j * 2 * np.pi * carrier_freq * time_arr), (num_blaschke_iters, 1))

    return low_freq_component, blaschke_product, cumulative_blaschke_product


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
    low_freq_component, blaschke_product, cumulative_blaschke_product = blaschke_decomposition(
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
            ax[total_order - 1, curr_order].plot(time_arr, (cumulative_blaschke_product[curr_order-1] * low_freq_component[curr_order]).real, label = f'$G_{curr_order}$ * prod(..., $B_{curr_order}$)', color='green', alpha=0.6)
            ax[total_order - 1, curr_order].legend(loc='lower left')

    final = 0
    for curr_order in range(1, blaschke_order + 1):
        final += (cumulative_blaschke_product[curr_order-1] * low_freq_component[curr_order]).real
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
