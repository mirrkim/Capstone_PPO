import numpy as np


def kalman_filter(measurements, Q=0.5, R=None):
    """
    1D Kalman filter for RSSI smoothing.
    """
    if R is None:
        R = float(np.var(measurements)) if len(measurements) > 1 else 6.0

    x = float(measurements[0])
    P = float(np.var(measurements)) if len(measurements) > 1 else 1.0

    for z in measurements[1:]:
        P_pred = P + Q
        K_gain = P_pred / (P_pred + R)
        x = x + K_gain * (float(z) - x)
        P = (1 - K_gain) * P_pred

    return x
