"""UTC calendar features from legacy cache timestamps, before float32 conversion."""
import numpy as np

CYCLICAL_TIME_FEATURES = ['utc_hour_sin', 'utc_hour_cos', 'year_phase_sin', 'year_phase_cos']


def cyclical_time(midnight_seconds, utc_hour):
    """Encode time of day and elapsed fraction of the actual calendar year."""
    midnight, hour = np.broadcast_arrays(
        np.asarray(midnight_seconds, dtype=np.float64), np.asarray(utc_hour, dtype=np.float64))
    if (not np.isfinite(midnight).all() or not np.isfinite(hour).all()
            or np.any((hour < 0) | (hour >= 24))):
        raise ValueError('Invalid UTC timestamp or hour')
    dates = midnight.astype('datetime64[s]')
    years = dates.astype('datetime64[Y]')
    beginning = years.astype('datetime64[s]').astype(np.int64)
    following = (years + np.timedelta64(1, 'Y')).astype('datetime64[s]').astype(np.int64)
    seconds = midnight + hour * 3600.
    daily = 2 * np.pi * hour / 24.
    annual = 2 * np.pi * (seconds - beginning) / (following - beginning)
    return np.stack((np.sin(daily), np.cos(daily), np.sin(annual), np.cos(annual)), axis=-1)
