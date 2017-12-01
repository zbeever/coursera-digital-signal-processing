import numpy as np

def scaled_fft_db(x):
    """ ASSIGNMENT 1:
        a) Compute a 512-point Hann window and use it to weigh the input data.
        b) Compute the DFT of the weighed input, take the magnitude in dBs and
        normalize so that the maximum value is 96dB.
        c) Return the first 257 values of the normalized spectrum

        Arguments:
        x: 512-point input buffer.

        Returns:
        first 257 points of the normalized spectrum, in dBs
    """
    N = len(x)
    window = get_window(N)

    X = np.absolute(np.fft.fft(x * window) / N)
    X = X[:N/2+1]

    for sample in np.where(X < 1e-5):
        X[sample] = 1e-5

    X = 20 * np.log10(X)
    X -= np.max(X) - 96

    return X


def get_window(M):
    window = 1 - np.cos(2 * np.pi * np.arange(0, M) / (M - 1))
    c = np.sqrt((M - 1) / np.sum(window ** 2))
    window *= c

    return window

if __name__ == '__main__':
    ca.check_assignment1()
