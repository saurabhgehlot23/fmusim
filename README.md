import numpy as np
from scipy.interpolate import interp1d

def resample_timeseries(time, data, fs_in, fs_out):
    """
    Resample a time series using linear interpolation.

    Parameters
    ----------
    time : array_like
        Original time vector.
    data : array_like
        Signal values corresponding to the time vector.
    fs_in : float
        Original sampling frequency (Hz).
    fs_out : float
        Desired output sampling frequency (Hz).

    Returns
    -------
    time_out : ndarray
        Resampled time vector.
    data_out : ndarray
        Resampled signal.
    """

    # Ensure numpy arrays
    time = np.asarray(time)
    data = np.asarray(data)

    # Total signal duration
    duration = time[-1] - time[0]

    # New time vector
    dt_out = 1.0 / fs_out
    time_out = np.arange(time[0], time[-1], dt_out)

    # Linear interpolation function
    interp_func = interp1d(
        time,
        data,
        kind='linear',
        fill_value="extrapolate"
    )

    # Resampled data
    data_out = interp_func(time_out)

    return time_out, data_out
