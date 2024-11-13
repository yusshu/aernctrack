import numpy


# Method to algorithmically obtain the period of a signal,
# given the time and value arrays of the signal, which should
# be sinusoidal. The method returns the period of the signal.
# If the period cannot be determined, returns None
def get_period_alg(xs, ys):
    # Find the peaks
    peaks_times = [xs[i] for i in range(1, len(ys) - 1) if ys[i - 1] < ys[i] > ys[i + 1]]
    if len(peaks_times) > 1:
        period = numpy.mean([peaks_times[i + 1] - peaks_times[i] for i in range(len(peaks_times) - 1)])
    else:
        period = None
    return period
