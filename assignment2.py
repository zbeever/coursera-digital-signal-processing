import numpy as np
from scipy import signal
import check_assignment as ca

def prototype_filter():
    """ ASSIGNMENT 2

        Compute the prototype filter used in subband coding. The filter
        is a 512-point lowpass FIR h[n] with bandwidth pi/64 and stopband
        starting at pi/32

        You should use the remez routine (signal.remez()). See
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.remez.html
    """

    # Your code goes here

    return signal.remez(512, [0.0, 0.00390625, 0.015625, 0.5], [2, 0])

if __name__ == '__main__':
    ca.check_assignment2()
