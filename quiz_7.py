import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    f0 = 0
    f1 = 10
    t1 = 2
    fs = 8000
    ts = float(1) / fs
    t = np.arange(0, t1, ts)
    a = (f1 - f0) / t1

    signal = np.cos(2 * np.pi * f0 * t + a * np.pi * t ** 2)

    plt.plot(signal)
    plt.plot(np.zeros(len(t)))
    plt.show()
