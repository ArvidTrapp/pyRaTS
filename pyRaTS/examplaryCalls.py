# -*- coding: utf-8 -*-
"""
@author: Arvid Trapp, Munich University of Applied Sciences
arvid.trapp@hm.edu
"""
import numpy as np
import matplotlib.pyplot as plt
import pyRaTS as ts
import FLife as fl
import scipy as sp
import pyFRF

#%% signal definition
inp  = []                   # empty list to be filled with signal specifications
fpsd = np.arange(0,100+1)   # frequency vector for corresponding PSD

# stationary part 1
psd  = np.zeros(fpsd.shape)
psd[(fpsd >= 10) & (fpsd < 60)] = 1
inp.append({'fpsd': fpsd, 'psd': psd, 'T': 100, 'var': 'x_1'})

# stationary part 2
psd2 = np.zeros(fpsd.shape)
psd2[(fpsd >= 50) & (fpsd < 100)] = 1
inp.append({'fpsd': fpsd, 'psd': psd2, 'T': 100, 'var': 'x_2'})

#%% generate timeseries

# check genGaussianSeries - generate realisation
# dictionary containing frequency vector 'fpsd', psd vector 'psd', time duration 'T' in seconds
x,t = ts.get_GaussianSeries(inp[0])

# generate realisation as object (direct)
sig = ts.timeseries(inp[0])

# derive SDOF response as timeseries object
respsig, _ = sig.der_sdofResponse(fD = 50, D = 0.03,func = 'acc2dis')
respsig.plot()

# generate quasi-stationary realisation via list of multiple signal specfications
qssig = ts.qstimeseries(inp)
qssig.plot()

# derive substitute load assumptions via FDS and response load spectra approximation
# inverse FDS (iFDS)
iFDS = qssig.der_iFDS(res = 1, m = 5, Nf = 30, D = 0.03, func='acc2dis',plot = False,maxit = 200)
iFDS.plot()
fds_qs   = qssig.est_fds(Nf = 30, D = 0.03, func='acc2dis')
fds_iFDS = iFDS.est_fds(Nf = 30, D = 0.03, func='acc2dis')
ts.plot_fds([fds_qs,fds_iFDS])  

#  response load spectra approximation
# sig.get_ls(Nf = 30, func='acc2dis', D = 0.03,p_interp = 30, plot = True)
# iRLS = sig.der_lsEquivalent(R = 5, Nf = 30, res = 1, D = 0.03, func='acc2dis',optix = None,maxit = 100) 
# iRLS.plot()   
# fds_iRLS = iRLS.est_fds(Nf = 30, D = 0.03, func='acc2dis')
# ts.plot_fds([fds_qs,fds_iFDS,fds_iRLS])  
#%% Plots
# Plots genGaussianSeries
plt.figure()
plt.plot(t,x,'r')
plt.show()

# plots realisation as timeseries - object
sig.plot()         # realisation
sig.plot_prob()    # PDF
sig.plot_psd()     # PSD
sig.plot_X()       # Fourier coefficients
sig.plot_ls()      # load spectra

# plots realisation as qstimeseries - object
qssig.plot()
qssig.plot_prob()
qssig.plot_psd()
qssig.plot_X()
qssig.plot_ls()

#%% calling some 'supporting functions' with output
SPM     = ts.est_specMoms(sig.psd,sig.fpsd)
D       = ts.est_DirlikD(sig.psd,sig.fpsd)
tf1     = ts.get_tfSDOF(sig.fpsd)
tf2     = ts.get_tfSDOF(sig.fpsd,fD = 20, D = 0.05, func = 'acc2acc')
tf3     = ts.get_tf(sig.fpsd,tf1,sig.fper) # get tf for given frequency vector

# SDOF linear transfer function
tfsig = sig.der_sdofTransfer(fD = 50, D = 0.03,func = 'acc2dis') # derive transfer function as timeseries object
tfsig.plot()
tfsig.plot_tf()
tfsig.plot_tf(plot = 'aphase')
tfsig.plot_tf(plot = 'real')

#%% calling some methods
sig.est_psd(df = 1)      # estimate PSD via specification of frequency resolution
sig.est_stats()          # estimate some basic statistical properties
sig.est_ls(si_X = 50)    # estimate load spectrum
sig.est_dirlik()         # estimate Dirlik load spectrum and damage

#%% calling some fatigue related methods
# load spectra (via FLife)
sig.plot_ls(method = None)                                  # RFC and Dirlik
sig.plot_ls(method = [fl.TovoBenasciutti,fl.Narrowband])    # Choice of FLife methods with load spectra support
# FDS (via FLife)
fdscalc1 = sig.est_fds(m = 5,Nf = np.arange(5,50,5), D = 0.02,method = 'DK')          # Dirlik: 'DK' / Rainflow C.: RFC / FLife methods
fdscalc2 = sig.est_fds(m = 5,Nf = np.arange(5,50,5), D = 0.02,method = fl.Narrowband) # Choice of FLife damage estimator methods
ts.plot_fds([fdscalc1,fdscalc2])        # Combined FDS Plot via FDS output dictionary
sig.plot_fds()                          # Calling FDS on object

#%% estimate non-stationarity matrix via Filtering-Averaging
''' Nf: frequency resolution without overlap
    olap: integer for overlapping factor
    wfunc: choice of windowing function
'''
sig.est_nonstat(Nf = 10,olap = 2,wfunc = sp.signal.windows.boxcar)
sig.plot_nonstat(func = 'Mxx')
'''
sig.plot_nonstat(func = 'Rxx')
sig.plot_nonstat(func = 'Rxx_sym') 
sig.plot_nonstat(func = 'Mxx_sym')
sig.plot_nonstat(func = 'Axx')
sig.plot_nonstat(func = 'Axx_sym')
sig.plot_nonstat(func = 'Cxx')
sig.plot_nonstat(func = 'Kxx') 
sig.plot_nonstat(func = 'rhoxx')
sig.plot_nonstat(func = 'rhoxx_sym')
'''

#%% (higher-order) linear structural dynamics
f       = sig.fpsd
H       = ts.get_tfSDOF(f,func = 'acc2acc')

m2,m4   = sig.der_statResponse(f,H) # second-order PSD / fourth-order non-stationarity matrix
ts.plot_nonstat(m4,func = 'Mxx')

# arbitrary linear transfer function
respsig1 = sig.der_response(f,H)

# SDOF linear transfer function
tfsig = sig.der_sdofTransfer(fD = 50, D = 0.03,func = 'acc2dis')
tfsig.plot()

respsig2,H = sig.der_sdofResponse(fD = 50, D = 0.03,func = 'acc2dis')
respsig2.plot()

# filter
filtsig = sig.der_highpass(20)
filtsig.plot()

# up-sampling
upsig = sig.der_zeropadded(zpfactor = 3)
upsig.plot()

#%% supporting functions: analytical signal                         
asig = sig.get_analyticalSignal(30,40)
plt.figure()
plt.plot(t,abs(asig),'r')
plt.show()  
asig = sig.get_posTaperedAnalyticalSignal(30,40)
plt.figure()
plt.plot(t,abs(asig),'r')
plt.show()   

sig
