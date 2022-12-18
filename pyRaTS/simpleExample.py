import pyRaTS as ts
import numpy as np
    
# Defining the series by pseudo-random generator...
T  = 10
fs = 1024
dt = 1/fs
t  = np.arange(T*fs)/fs
x  = np.random.randn(T*fs)

# Init series and some basic plots...
sig = ts.timeseries(x,t,name = 'sample timeseries')
sig.plot()
sig.plot_prob()
sig.plot_psd()

# derive response series and some further basic plots...
respsig,_ = sig.der_sdofResponse(fD = 50)
respsig.plot_psd()
respsig.plot_ls()