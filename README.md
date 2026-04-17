import numpy as np
from scipy.signal import get_window


def amplitude_spectrum_rms(
    signal: np.ndarray,
    fs: float,
    block_size: int = 1024,
    window: str = "hann",
    overlap: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the RMS amplitude spectrum of a signal using linear averaging
    (equivalent to nCode FFT Spectrum block: Hann window, 50% overlap,
    linear average, RMS amplitude scaling).

    Parameters
    ----------
    signal     : 1D array of filtered time-domain samples
    fs         : Sampling frequency (Hz)
    block_size : FFT block size (number of samples per segment)
    window     : Window type (default 'hann')
    overlap    : Fractional overlap between blocks (default 0.50)

    Returns
    -------
    f     : Frequency axis (Hz), length = block_size // 2 + 1
    A_rms : RMS amplitude spectrum (same units as input signal)
    """

    # ------------------------------------------------------------------ #
    # 1. Segment the signal into overlapping blocks
    # ------------------------------------------------------------------ #
    hop        = int(block_size * (1.0 - overlap))   # step between block starts
    n_samples  = len(signal)
    starts     = np.arange(0, n_samples - block_size + 1, hop)
    n_blocks   = len(starts)

    if n_blocks == 0:
        raise ValueError(
            f"Signal length ({n_samples}) is shorter than block_size ({block_size}). "
            "Use a smaller block_size."
        )

    # ------------------------------------------------------------------ #
    # 2. Build window and compute coherent power gain (CPG)
    # ------------------------------------------------------------------ #
    win     = get_window(window, block_size)          # shape: (block_size,)
    cpg     = np.sum(win)                             # coherent power gain
    # RMS normalisation factor per bin
    # For a real FFT:  A_rms = |X(f)| * sqrt(2) / CPG   (two-sided → one-sided)
    # DC and Nyquist bins are NOT doubled (single-sided unique lines)

    # ------------------------------------------------------------------ #
    # 3. Accumulate linear-averaged single-sided amplitude spectrum
    # ------------------------------------------------------------------ #
    n_lines     = block_size // 2 + 1                 # number of unique freq lines
    sum_A       = np.zeros(n_lines)                   # accumulator for linear average

    for start in starts:
        block   = signal[start : start + block_size]
        X       = np.fft.rfft(block * win)            # complex spectrum, length n_lines
        A_block = np.abs(X)                           # single-sided magnitude

        # Scale to RMS amplitude (one-sided)
        A_scaled                = A_block * np.sqrt(2) / cpg
        A_scaled[0]             = A_block[0] / cpg   # DC   — no sqrt(2) factor
        if block_size % 2 == 0:
            A_scaled[-1]        = A_block[-1] / cpg  # Nyquist — no sqrt(2) factor

        sum_A += A_scaled

    # Linear average across blocks
    A_rms = sum_A / n_blocks

    # ------------------------------------------------------------------ #
    # 4. Frequency axis
    # ------------------------------------------------------------------ #
    f = np.fft.rfftfreq(block_size, d=1.0 / fs)      # Hz

    return f, A_rms
