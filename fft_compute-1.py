"""
fft_compute.py
--------------
Computes the single-sided, energy-normalised FFT magnitude spectrum from a
time-domain signal (e.g. acceleration in m/s²).

Normalisation convention
------------------------
The output magnitude array uses **energy normalisation**:

    magnitude[k] = sqrt( mag_sq_onesided[k] ) / sqrt( N * sum(w²) )

where w is the window array and N is the block size. Under this convention:

    RMS_band = sqrt( sum( magnitude[k]²  for k in band ) )

exactly recovers the time-domain RMS of the signal components within that
band — this is Parseval's theorem applied correctly with windowing.

A sinusoid of peak amplitude A contributes A/sqrt(2) to the band RMS, which
is the expected RMS of a sine wave.

References
----------
- Parseval / DFT energy : Oppenheim & Schafer, "Discrete-Time Signal Processing"
- Window normalisation   : Harris (1978), IEEE Proc. 66(1)
- Band RMS from spectrum : Bendat & Piersol, "Random Data", 4th ed., Sec. 11.4
"""

import numpy as np
from scipy.signal import get_window


def compute_fft(
    signal: np.ndarray,
    fs: float,
    block_size: int = 1024,
    window: str = "hann",
    overlap: float = 0.5,
) -> dict:
    """
    Compute the single-sided, energy-normalised, block-averaged FFT magnitude.

    Parameters
    ----------
    signal     : 1-D NumPy array  time-domain signal (e.g. acceleration m/s²)
    fs         : float            sampling frequency (Hz)
    block_size : int              FFT block length N (samples); use powers of 2
                                  e.g. 512, 1024, 2048, 4096
    window     : str              SciPy window name — 'hann' (default),
                                  'hamming', 'blackman', 'flattop', 'boxcar'
    overlap    : float            fractional overlap between blocks [0, 1)
                                  0.5 (50 %) is standard with Hann window

    Returns
    -------
    dict with keys
        'freqs'      — ndarray (Hz), frequency axis, length N//2 + 1
        'magnitude'  — ndarray (same unit as input), energy-normalised
                       one-sided magnitude, averaged across blocks.
                       Band RMS = sqrt(sum(magnitude[k]²)) over the band.
        'fs'         — float, sampling frequency
        'block_size' — int, N used
        'window'     — str, window name
        'overlap'    — float, overlap fraction
        'n_blocks'   — int, number of blocks averaged
        'df'         — float, frequency resolution = fs / N  (Hz)
    """
    N   = int(block_size)
    hop = int(N * (1.0 - overlap))
    if hop < 1:
        hop = 1

    win_array = get_window(window, N)

    # ------------------------------------------------------------------
    # Energy normalisation factor
    #
    # Parseval for windowed DFT (unnormalised rfft):
    #   sum_n |x[n]|² * sum_n w[n]²  ≈  (1/N) * sum_k |X_w[k]|²
    #
    # Rearranging, the energy-normalised one-sided magnitude satisfies:
    #   RMS²_band = sum_band magnitude[k]²
    #
    # The denominator that achieves this is: sqrt(N * sum(w²))
    # ------------------------------------------------------------------
    energy_norm = np.sqrt(N * np.sum(win_array ** 2))

    freqs  = np.fft.rfftfreq(N, d=1.0 / fs)    # length N//2 + 1
    n_bins = len(freqs)

    # Accumulate squared magnitudes across blocks (power/RMS averaging)
    mag_sq_accum = np.zeros(n_bins, dtype=np.float64)
    n_blocks = 0

    start = 0
    while start + N <= len(signal):
        block = signal[start : start + N] * win_array
        X     = np.fft.rfft(block)

        mag_sq_accum += np.abs(X) ** 2
        n_blocks     += 1
        start        += hop

    if n_blocks == 0:
        raise ValueError(
            f"Signal length ({len(signal)} samples) < block_size ({N}). "
            "Reduce block_size or provide a longer signal."
        )

    # Average across blocks
    mag_sq_avg = mag_sq_accum / n_blocks

    # ------------------------------------------------------------------
    # One-sided correction
    # rfft returns bins 0 … N//2.  Bins 1 … N//2-1 each represent a pair
    # of conjugate frequencies; doubling their squared magnitude folds the
    # energy from the negative-frequency side onto the positive side.
    # DC (k=0) and Nyquist (k=N//2) have no conjugate partner.
    # ------------------------------------------------------------------
    mag_sq_onesided          = mag_sq_avg.copy()
    mag_sq_onesided[1:-1]   *= 2.0

    magnitude = np.sqrt(mag_sq_onesided) / energy_norm

    return {
        "freqs"      : freqs,
        "magnitude"  : magnitude,
        "fs"         : fs,
        "block_size" : N,
        "window"     : window,
        "overlap"    : overlap,
        "n_blocks"   : n_blocks,
        "df"         : float(freqs[1] - freqs[0]),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs     = 1000.0
    t      = np.arange(0, 10.0, 1.0 / fs)
    # Known signal: 10 Hz (A=1.0) + 30 Hz (A=0.5) + low-level noise
    signal = (1.0 * np.sin(2 * np.pi * 10 * t)
            + 0.5 * np.sin(2 * np.pi * 30 * t)
            + 0.02 * np.random.randn(len(t)))

    result = compute_fft(signal, fs, block_size=2048, window="hann", overlap=0.5)

    rms_spectral = np.sqrt(np.sum(result["magnitude"] ** 2))
    rms_time     = np.sqrt(np.mean(signal ** 2))

    print(f"Blocks averaged     : {result['n_blocks']}")
    print(f"Freq resolution     : {result['df']:.4f} Hz")
    print(f"RMS (spectral sum)  : {rms_spectral:.5f} m/s²")
    print(f"RMS (time domain)   : {rms_time:.5f} m/s²")

    plt.figure(figsize=(10, 4))
    plt.plot(result["freqs"], result["magnitude"], linewidth=0.9, color="#1f77b4")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude — energy norm. (m/s²)")
    plt.title("Single-sided FFT Magnitude Spectrum  |  block=2048, Hann, 50 % overlap")
    plt.xlim(0, 100)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/fft_spectrum.png", dpi=150)
    print("Plot saved → fft_spectrum.png")
