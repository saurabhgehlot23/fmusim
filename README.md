import numpy as np
from scipy.signal import butter, filtfilt, welch

# -----------------------------------------------------
# 1. Butterworth bandpass filter (0.5 – 6 Hz)
# -----------------------------------------------------

def bandpass_filter(signal, fs):

    f_low = 0.5
    f_high = 6.0
    order = 5

    nyq = fs / 2

    b, a = butter(order, [f_low/nyq, f_high/nyq], btype='band')

    filtered = filtfilt(b, a, signal)

    return filtered


# -----------------------------------------------------
# 2. PSD computation using Welch method
# -----------------------------------------------------

def compute_psd(signal, fs):

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
