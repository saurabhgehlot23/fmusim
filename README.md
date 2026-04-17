import numpy as np
from scipy.signal import butter, filtfilt, get_window


# -----------------------------------------------------
# 1. Bandpass Filter (0.5 – 6 Hz, 5th order Butterworth)
# -----------------------------------------------------
def bandpass_filter(signal, fs):

    f_low = 0.5
    f_high = 6.0
    order = 5

    nyq = fs / 2.0
    b, a = butter(order, [f_low/nyq, f_high/nyq], btype='band')

    return filtfilt(b, a, signal)


# -----------------------------------------------------
# 2. RMS-Amplitude Spectrum (nCode equivalent)
# -----------------------------------------------------
def compute_rms_amplitude_spectrum(signal, fs,
                                   block_size=8192,
                                   overlap=0.5):

    N = block_size
    hop = int(N * (1 - overlap))

    window = get_window('hann', N)

    freqs = np.fft.rfftfreq(N, 1/fs)
    n_bins = len(freqs)

    mag_sq_accum = np.zeros(n_bins)
    n_blocks = 0

    for start in range(0, len(signal) - N + 1, hop):

        block = signal[start:start+N]

        # Remove DC (VERY IMPORTANT)
        block = block - np.mean(block)

        # Apply window
        block = block * window

        # FFT
        X = np.fft.rfft(block)

        mag_sq = np.abs(X)**2

        # One-sided correction (IMPORTANT)
        mag_sq[1:-1] *= 2.0

        mag_sq_accum += mag_sq
        n_blocks += 1

    if n_blocks == 0:
        raise ValueError("Signal too short for given block size")

    mag_sq_avg = mag_sq_accum / n_blocks

    # -------------------------------------------------
    # Correct normalization → RMS amplitude spectrum
    # -------------------------------------------------
    window_energy = np.sum(window**2)

    magnitude = np.sqrt(mag_sq_avg / window_energy) / N

    return freqs, magnitude


# -----------------------------------------------------
# 3. Band RMS (NO df needed)
# -----------------------------------------------------
def band_rms(freqs, magnitude, f1=0.5, f2=6.0):

    band = (freqs >= f1) & (freqs <= f2)

    return np.sqrt(np.sum(magnitude[band]**2))


# -----------------------------------------------------
# 4. Full Seat RMS Pipeline
# -----------------------------------------------------
def compute_seat_rms(ax, ay, az, fs):

    # Step 1: Filter
    x = bandpass_filter(ax, fs)
    y = bandpass_filter(ay, fs)
    z = bandpass_filter(az, fs)

    # Step 2: Spectrum
    fx, mx = compute_rms_amplitude_spectrum(x, fs)
    fy, my = compute_rms_amplitude_spectrum(y, fs)
    fz, mz = compute_rms_amplitude_spectrum(z, fs)

    # Step 3: Band RMS
    rms_x = band_rms(fx, mx)
    rms_y = band_rms(fy, my)
    rms_z = band_rms(fz, mz)

    # Step 4: Resultant RMS
    rms_xyz = np.sqrt(rms_x**2 + rms_y**2 + rms_z**2)

    return rms_x, rms_y, rms_z, rms_xyz


# -----------------------------------------------------
# 5. SANITY CHECK (CRITICAL)
# -----------------------------------------------------
def sanity_check(signal, fs):

    freqs, mag = compute_rms_amplitude_spectrum(signal, fs)

    rms_time = np.sqrt(np.mean(signal**2))
    rms_fft  = np.sqrt(np.sum(mag**2))

    print("\n--- SANITY CHECK ---")
    print("Time-domain RMS :", rms_time)
    print("FFT RMS         :", rms_fft)
    print("Error (%)       :", 100*(rms_fft - rms_time)/rms_time)
    block_size = 8192
    overlap = int(block_size * 0.5)

    freqs, psd = welch(
        signal,
        fs=fs,
        window='hann',
        nperseg=block_size,
        noverlap=overlap,
        scaling='density'
    )

    return freqs, psd


# -----------------------------------------------------
# 3. Band RMS calculation
# -----------------------------------------------------

def band_rms(freqs, psd, f1=0.5, f2=6.0):

    df = freqs[1] - freqs[0]

    band = (freqs >= f1) & (freqs <= f2)

    rms = np.sqrt(np.sum(psd[band] * df))

    return rms


# -----------------------------------------------------
# 4. Complete pipeline
# -----------------------------------------------------

def seat_rms(signal_x, signal_y, signal_z, fs):

    # Filter signals
    x = bandpass_filter(signal_x, fs)
    y = bandpass_filter(signal_y, fs)
    z = bandpass_filter(signal_z, fs)

    # PSD
    fx, psdx = compute_psd(x, fs)
    fy, psdy = compute_psd(y, fs)
    fz, psdz = compute_psd(z, fs)

    # RMS per axis
    rms_x = band_rms(fx, psdx)
    rms_y = band_rms(fy, psdy)
    rms_z = band_rms(fz, psdz)

    # Resultant RMS
    rms_xyz = np.sqrt(rms_x**2 + rms_y**2 + rms_z**2)

    return rms_x, rms_y, rms_z, rms_xyz
