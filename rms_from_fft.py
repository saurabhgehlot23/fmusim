"""
rms_from_fft.py
---------------
Computes the RMS value of a signal within a specified frequency band,
using the energy-normalised FFT magnitude from fft_compute.compute_fft().

Method
------
Given the energy-normalised magnitude array from Script 1, band RMS follows
directly from Parseval's theorem:

    RMS_band = sqrt( sum( magnitude[k]²  for k in [f_low, f_high] ) )

This is exact (within spectral resolution) because the energy normalisation
in fft_compute.py ensures each magnitude[k]² equals the mean-square
contribution of the frequency component at bin k.

For a single sinusoid A·sin(2π·f₀·t) in the band: RMS = A / sqrt(2).
For multiple components: RMS = sqrt( sum of individual RMS² ).

References
----------
- Parseval : Oppenheim & Schafer, "Discrete-Time Signal Processing", Ch. 8
- Band RMS : Bendat & Piersol, "Random Data", 4th ed., Sec. 11.4
"""

import numpy as np
from fft_compute import compute_fft


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def rms_in_band(
    fft_result: dict,
    f_low: float,
    f_high: float,
) -> dict:
    """
    Compute the RMS of a signal within the frequency band [f_low, f_high].

    Parameters
    ----------
    fft_result : dict   output of fft_compute.compute_fft()
    f_low      : float  lower frequency limit (Hz), inclusive
    f_high     : float  upper frequency limit (Hz), inclusive

    Returns
    -------
    dict with keys
        'rms'        — float, band-limited RMS (same unit as original signal)
        'f_low_act'  — float, actual lower bin frequency used (Hz)
        'f_high_act' — float, actual upper bin frequency used (Hz)
        'n_bins'     — int,   number of FFT bins in the band
        'band_freqs' — ndarray, frequency values of bins in band (Hz)
        'band_mag'   — ndarray, magnitude values of bins in band
    """
    freqs     = fft_result["freqs"]
    magnitude = fft_result["magnitude"]
    nyquist   = freqs[-1]

    # --- Input validation ---
    if f_low < 0:
        raise ValueError(f"f_low must be >= 0, got {f_low} Hz.")
    if f_high > nyquist:
        raise ValueError(
            f"f_high ({f_high} Hz) exceeds Nyquist ({nyquist:.1f} Hz). "
            "Increase fs or reduce f_high."
        )
    if f_low >= f_high:
        raise ValueError(
            f"f_low ({f_low} Hz) must be less than f_high ({f_high} Hz)."
        )

    # Nearest FFT bin indices
    idx_low  = int(np.argmin(np.abs(freqs - f_low)))
    idx_high = int(np.argmin(np.abs(freqs - f_high)))

    band_freqs = freqs[idx_low : idx_high + 1]
    band_mag   = magnitude[idx_low : idx_high + 1]

    # Parseval band RMS (energy-normalised magnitude — see fft_compute.py)
    rms_value = float(np.sqrt(np.sum(band_mag ** 2)))

    return {
        "rms"        : rms_value,
        "f_low_act"  : float(freqs[idx_low]),
        "f_high_act" : float(freqs[idx_high]),
        "n_bins"     : len(band_freqs),
        "band_freqs" : band_freqs,
        "band_mag"   : band_mag,
    }


# ---------------------------------------------------------------------------
# Convenience one-shot wrapper
# ---------------------------------------------------------------------------

def signal_to_band_rms(
    signal: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
    block_size: int = 1024,
    window: str = "hann",
    overlap: float = 0.5,
) -> dict:
    """
    One-shot: time-domain signal → band-limited RMS.

    Internally calls compute_fft() then rms_in_band().
    Returns a merged dict containing both FFT metadata and RMS result.
    """
    fft_result = compute_fft(signal, fs, block_size, window, overlap)
    rms_result = rms_in_band(fft_result, f_low, f_high)
    return {**fft_result, **rms_result}


# ---------------------------------------------------------------------------
# Self-test and demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # -----------------------------------------------------------------------
    # Test signal:
    #   10 Hz  A = 1.0 m/s²  → expected RMS = 1/√2  ≈ 0.7071
    #   30 Hz  A = 0.5 m/s²  → expected RMS = 0.5/√2 ≈ 0.3536
    #   50 Hz  A = 0.3 m/s²  → expected RMS = 0.3/√2 ≈ 0.2121
    #   noise  σ = 0.02 m/s² (negligible)
    # -----------------------------------------------------------------------
    fs     = 1000.0
    t      = np.arange(0, 10.0, 1.0 / fs)
    signal = (1.0 * np.sin(2 * np.pi * 10 * t)
            + 0.5 * np.sin(2 * np.pi * 30 * t)
            + 0.3 * np.sin(2 * np.pi * 50 * t)
            + 0.02 * np.random.randn(len(t)))

    # --- Step 1: Compute FFT ---
    fft_result = compute_fft(signal, fs, block_size=2048, window="hann", overlap=0.5)

    print("=" * 60)
    print("STEP 1 — FFT")
    print(f"  Blocks averaged : {fft_result['n_blocks']}")
    print(f"  Freq resolution : {fft_result['df']:.4f} Hz")
    print(f"  Nyquist         : {fft_result['freqs'][-1]:.1f} Hz")

    # --- Step 2: Band RMS ---
    test_cases = [
        ( 5.0,  20.0, "10 Hz only",          1.0 / np.sqrt(2)),
        (20.0,  40.0, "30 Hz only",          0.5 / np.sqrt(2)),
        (40.0,  60.0, "50 Hz only",          0.3 / np.sqrt(2)),
        ( 5.0,  60.0, "all three components",
          np.sqrt((1.0**2 + 0.5**2 + 0.3**2) / 2)),
    ]

    print("\n" + "=" * 60)
    print("STEP 2 — BAND RMS")
    print(f"{'Band':>12}  {'Label':<24}  {'Calc':>9}  {'Expected':>9}  {'Err%':>6}")
    print("-" * 60)

    for f_lo, f_hi, label, expected in test_cases:
        r   = rms_in_band(fft_result, f_lo, f_hi)
        err = 100.0 * abs(r["rms"] - expected) / expected
        print(f"  {f_lo:4.0f}–{f_hi:4.0f} Hz  {label:<24}  "
              f"{r['rms']:>9.5f}  {expected:>9.5f}  {err:>5.2f}%")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    colors = ["#e64c2e", "#f7a800", "#2e7d32"]
    bands_plot = [
        ( 5.0, 20.0, colors[0], "5–20 Hz"),
        (20.0, 40.0, colors[1], "20–40 Hz"),
        (40.0, 60.0, colors[2], "40–60 Hz"),
    ]

    # Top: spectrum with shaded bands
    ax = axes[0]
    ax.plot(fft_result["freqs"], fft_result["magnitude"],
            color="#1f77b4", linewidth=0.9, label="FFT magnitude", zorder=3)
    for fl, fh, c, lbl in bands_plot:
        ax.axvspan(fl, fh, alpha=0.18, color=c, label=lbl)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (m/s²)")
    ax.set_title("Single-sided FFT Magnitude Spectrum with Analysis Bands")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom: bar chart of band RMS values
    ax2 = axes[1]
    band_labels = [f"{fl:.0f}–{fh:.0f} Hz" for fl, fh, _, _ in bands_plot]
    calc_rms    = [rms_in_band(fft_result, fl, fh)["rms"] for fl, fh, _, _ in bands_plot]
    exp_rms     = [1.0/np.sqrt(2), 0.5/np.sqrt(2), 0.3/np.sqrt(2)]
    x = np.arange(len(band_labels))
    w = 0.35

    bars1 = ax2.bar(x - w/2, calc_rms, w,
                    color=[c for _, _, c, _ in bands_plot], alpha=0.85,
                    label="Calculated RMS")
    bars2 = ax2.bar(x + w/2, exp_rms, w,
                    color="none", edgecolor="black", linewidth=1.5,
                    linestyle="--", label="Expected RMS")
    ax2.set_xticks(x)
    ax2.set_xticklabels(band_labels)
    ax2.set_ylabel("RMS (m/s²)")
    ax2.set_title("Band RMS: Calculated vs Expected")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars1, calc_rms):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/rms_band_result.png", dpi=150)
    print("\nPlot saved → rms_band_result.png")
