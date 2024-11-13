import numpy as np


# Given an array of Xs and Ys values for a sinusoidal function or
# signal, uses the Fast Fourier Transform to determine the period
# of oscillation of the signal. Return the period or None if the
# period cannot be determined.
def get_period_fft(xs, ys):
    # Ensure that xs and ys have the same length and there are at least two points
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    t = xs[1] - xs[0]  # sample spacing, assumes uniform time spacing
    fft_values = np.fft.fft(ys)  # fft
    frequencies = np.fft.fftfreq(len(ys), d=t)  # get the frequencies
    magnitudes = np.abs(fft_values)

    # exclude the zero frequency and find the dominant frequency
    magnitudes[0] = 0
    dominant_frequency = frequencies[np.argmax(magnitudes)]

    # if the dominant frequency is zero, the period is undetermined
    if dominant_frequency == 0:
        return None

    # calculate the period as the inverse of the dominant frequency
    period = 1 / abs(dominant_frequency)
    return period
