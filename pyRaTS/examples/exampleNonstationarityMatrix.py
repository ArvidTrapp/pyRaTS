# -*- coding: utf-8 -*-
"""
demonstration of non-stationary matrix using...
PYthon module for the processing of RAndom vibration TimeSeries - pyRaTS
...
@author: Arvid Trapp, Munich University of Applied Sciences
arvid.trapp@hm.edu
"""
#%%
import numpy as np
import scipy as sp
import pyExSi as ex
import matplotlib.pyplot as plt
import pyRaTS as ts
import FLife

''' aims of this example:
Three synthetic time series are generated; one is stationary (sig_stat) and two are 
non-stationary employing a simple amplitude-modulation. However, these two non-stationary series differentiate by
the (carrier) frequencies that are modulated. One has low-frequent [0 200] Hz 
carrier frequencies (sig_ns0to200), the other one high-frequent [200 400] Hz
carrier frequencies (sig_ns200to400). 

These three series will be compared in the following using the pyRaTS timeseries class. They will be evaluated by their 
non-stationarity matrix (representation of spectral non-stationarity property - in the frequency domain), the 
Fatigue Damage Spectrum (FDS) and their response series, when transfered through a 
synthetic linear 1DOF system. 

- The non-stationary 'property' influences the fatigue damage potential. This
cannot be identified when using popular frequency-domain estimators (e.g. Dirlik FDS), 
since the example is designed so that all input (excitation) and thus all 
outputs (responses) have the same (average) PSD.
- Consequently it requires techniques such as the FDS (RFC of synthetic 
responses) or the non-stationarity matrix, which can be used to identify the 
frequencies having a non-stationary characteristic. The non-stationarity matrix  
can be processed via linear systems theory (here 1DOF) to estimate how this 
characteristic will transfer into the response. 
- The non-stationarity matrix lets one see that due to the resonant
frequency of the 1DOF the low-freq modulation (sig_ns0to200) will lead to 
non-stationary response, while high-freq non-stat. response (sig_ns200to400) 
loses characteristic in response
'''
    
''' signal definition 
three signals with frequency contributions in [0 400]:
    {stationary, nonstationary in [0, 200], nonstationary in [200, 400]}    
'''
T   = 100               # number of data points of time series
fs  = 2048              # sampling frequency [Hz]
N   = T*fs              # duration [s]
t   = np.arange(N)/fs   # time vector             
df  = 1/T               # frequency resolution [Hz]

''' define frequency vector and one-sided flat-shaped PSD '''
M    = N//2 + 1            # number of data points of frequency vector
freq = np.arange(M) * df   # frequency vector

''' non-stationarity porperty using amplitude-modulation: 
    sine modulation with 1/4 Hz being normed by mean squared '''
fmod = 1/4  # modulation frequency 
sine_modulation = 1 - 1/7 + 1/np.sqrt(2)*np.sin(2*np.pi*fmod*t)

''' prepare frequencies [0, 200] '''
PSD_0to200          = ex.get_psd(freq, freq_lower = df, freq_upper = 200)
stat_gauss_0to200   = ex.random_gaussian(N, PSD_0to200, fs)
nonstat_0to200      = sine_modulation*stat_gauss_0to200

''' prepare frequencies [200, 400] '''
PSD_200to400        = ex.get_psd(freq, freq_lower = 200+df, freq_upper = 400)
stat_gauss_200to400 = ex.random_gaussian(N, PSD_200to400, fs)
nonstat_200to400    = sine_modulation*stat_gauss_200to400

''' random noise '''
noise = np.random.randn(len(stat_gauss_0to200)) * stat_gauss_0to200.std()/20

''' define time series: {stationary, nonstationary in [0, 200], nonstationary in [200, 400]} '''
sig_stat       = {'x':stat_gauss_0to200 + stat_gauss_200to400 + noise,'var': "x", 'name': 'stationary time series'}
sig_ns0to200   = {'x':nonstat_0to200    + stat_gauss_200to400 + noise,'var': "x_{ns,lowf}", 'name': 'time series non-stationary in [0, 200] Hz'}
sig_ns200to400 = {'x':stat_gauss_0to200 + nonstat_200to400    + noise,'var': "x_{ns,highf}", 'name': 'time series non-stationary in [200, 400] Hz'}

# combine sigs as list to make them iterable
sigs   = ts.timeseries([[sig_stat], [sig_ns0to200], [sig_ns200to400]],fs=fs)
FDS_RFC, FDS_DK, respsigs = [],[],[]
resps_gyy,resps_nonstat = [],[]

# plot of timeseries
sigs.plot()
# plot of PSD
sigs.plot_psd()
# plot of load spectra 
sigs.plot_ls(method = [FLife.TovoBenasciutti])
''' estimation of non-stationarity matrix '''
sigs.est_nonstat(Nf = 50,olap=2,fu = 500,wfunc = sp.signal.windows.boxcar)
sigs.plot_nonstat(func=['Mxx','Axx'])
print('excess kurtosis (timeseries):',sigs.stat['kurtosis'])
print('excess kurtosis (statistical):',[chan.nonstat['kurtosis'] for chan in sigs.chan]) 
    
# calculation of Fatigue Damage Spectrum (FDS)
FDS_RFC = sigs.est_fds(func = 'acc2acc',Nf =  60, m =  5)
FDS_DK  = sigs.est_fds(func = 'acc2acc',Nf =  60, m =  5, method = FLife.Dirlik)
sigs.plot_fds() # overview of all FDS

''' Comparison of the FDS inbetween signals '''  
# time-domain FDS identifies larger 'damage potential' due to non-stationarity  
ts.plot_fds(FDS_RFC)
# frequency-domain FDS does NOT identidy larger 'damage potential' being based on the average PSD
ts.plot_fds(FDS_DK)
''' Comparison of FDS for different methods ''' 
for chan in sigs.chan:
    chan.plot_fds()    

''' response analysis '''
# derive response signals for fD = 50
f = sigs.fpsd
H = ts.get_tfSDOF(f,fD = 50, D = 0.05, func = 'acc2acc')
# realization
respsigs  = sigs.der_response(f,H)
respsigs.plot()
respsigs.plot_psd()
# non-stationarity matrix
respsigs.est_nonstat(Nf = 50,olap=2,fu = 500,wfunc = sp.signal.windows.boxcar)
respsigs.plot_nonstat(func=['Mxx'])

# Comparison of response kurtosis for sampling-based / statistical calculation 
resp_nonstat = sigs.der_statResponse(f,H)
for nonstat in resp_nonstat:
    ts.plot_nonstat(nonstat[1],func=['Mxx']) # nonstat[0]: Response-PSD, nonstat[1]: Response-NSM 
print('excess kurtosis (timeseries): ', respsigs.stat['kurtosis'])
print('excess kurtosis (statistical): ', [chan.nonstat['kurtosis'] for chan in respsigs.chan])   
    
# %%
