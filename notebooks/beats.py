
# coding: utf-8

# # Exploring Beat Frequencies
# 
# This simple notebook will let you play with close frequencies and hear the beatings created by intermodulation. It's also a cute example of the interactivity you can achieve with notebooks.

# In[13]:

# standard bookkeeping
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display
# interactivity here:
from ipywidgets import interactive, fixed

plt.rcParams["figure.figsize"] = (14,4)


# Let's define a simple fuction that generates, plots and plays two sinusoids at the given frequencies:

# In[14]:

def beat_freq(f1=220.0, f2=224.0):
    # the clock of the system
    LEN = 3 # seconds
    Fs = 8000.0
    n = np.arange(0, int(LEN * Fs))
    s = np.sin(2*np.pi * f1/Fs * n) + np.sin(2*np.pi * f2/Fs * n)
    # play the sound
    display(Audio(data=s, rate=Fs))
    # display one second of audio
    plt.plot(s[:int(Fs)])


# In[15]:

v = interactive(beat_freq, f1=(200.0,300.0), f2=(200.0,300.0))
display(v)


# In[ ]:








# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



