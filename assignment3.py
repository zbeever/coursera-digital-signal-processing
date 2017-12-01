import numpy as np

def subband_filtering(x, h):
    """ ASSIGNMENT 3

        Write a routine to implement the efficient version of the subband filter
        as specified by the MP3 standard

        Arguments:
        x:  a new 512-point data buffer, in time-reversed order [x[n],x[n-1],...,x[n-511]].
        h:  The prototype filter of the filter bank you found in the previous assignment

        Returns:
        s: 32 new output samples
    """
    r = np.zeros(512)
    c = np.zeros(64)
    s = np.zeros(32)

    for k in range(512):
        r[k] = h[k] * x[k]

    for q in range(64):
        c[q] = 0
        for p in range(8):
            c[q] += ((-1) ** p) * r[q + 64 * p]

    for i in range(32):
        s[i] = 0
        for q in range(64):
            s[i] += np.cos(np.pi / 64 * (2 * i + 1) * (q - 16)) * c[q]

    return s
