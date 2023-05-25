# %% Arvid Trapp
import numpy as np
import pyExSi as ex
import matplotlib.pyplot as plt
import pyRaTS as ts
import FLife

''' Defining the synthetic example'''
fs    = 600             # sampling frequency [Hz]            
dfpsd = 0.05            # frequency resolution
fpsd  = np.arange(0,fs/2+dfpsd,dfpsd)

# %%
'''PSD-Shapes'''
psd1  = np.zeros(fpsd.shape)
psd1[(fpsd >= 10) & (fpsd < 30)] = 0.5
psd1[(fpsd >= 30) & (fpsd < 50)] = 0.25
psd1[(fpsd >= 50) & (fpsd < 80)]  = 0.1
psd2  = np.zeros(fpsd.shape)
psd2[(fpsd >= 30) & (fpsd < 50)] = 0.0625
psd2[(fpsd >= 50) & (fpsd < 80)]  = 0.1
psd2[(fpsd >= 80) & (fpsd < 90)] = 0.125

p = [0.2, 0.8]
avpsd = p[0]*psd1+p[1]*psd2

''' Generate Realization'''
T = 100
defswitch = []
defswitch.append({'psd': psd1,'fpsd':fpsd,'T':p[0]*T,'var': 'x_{ns,p=1}'})
defswitch.append({'psd': psd2,'fpsd':fpsd,'T':p[1]*T,'var': 'x_{ns,p=2}'})
defstat = []
defstat.append({'psd': avpsd,'fpsd':fpsd,'T':p[0]*T,'var': 'x_{stat,p=1}'})
defstat.append({'psd': avpsd,'fpsd':fpsd,'T':p[1]*T,'var': 'x_{stat,p=2}'})
sig    = ts.timeseries([defswitch,defstat])

# Optional: naming of channels
sig.chan[0].var = 'x_{ns}'
sig.chan[0].name = '(non-stationary)'
sig.chan[1].var = 'x_{stat}'
sig.chan[1].name = '(stationary)'

# PSDs
sig.plot()
sig.plot_psd(access='proc')
sig.plot_psd(access='comp')

# responses
respsig20 = sig.der_sdofResponse(fD = 20,func = 'acc2dis4mm')
respsig20.plot()
respsig20.plot_psd(access='comp')
respsig20.plot_ls(access = 'comp')

# non-stationarity matrix
NSMs = sig.est_nonstat(fl = 10,fu = 90,Nf = 40,olap = 2)
for NSM in NSMs:
    ts.plot_nonstat(NSM,func = 'Mxx',lim=[0,3])
    ts.plot_nonstat(NSM,func = 'Axx',lim=[0,6])
    ts.plot_nonstat(NSM,func = 'Cxx',lim=[-0.5,2.5])

# bi-modal response
tf20     = ts.get_tfSDOF(fpsd,fD = 20, D = 0.03, func = 'acc2acc')
tf65     = ts.get_tfSDOF(fpsd,fD = 65, D = 0.03, func = 'acc2acc')
tf20_65  = tf20+tf65
respsig20_65 = sig.der_response(fpsd,tf20_65)
respsig20_65.chan[0].var = 'y_{ns,fD: 20, 65}'
respsig20_65.chan[0].name = '(non-stationary response)'
respsig20_65.chan[1].var = 'y_{stat,fD: 20, 65}'
respsig20_65.chan[1].name = '(stationary response)'

respsig20_65.plot()
respsig20_65.plot_psd()
respsig20_65.plot_ls(access = 'comp')

# analysis of fourth-order transfer behavior
tfobj20_65 = sig.arr[0][0].der_tfObj(fpsd,tf20_65)
Mxx20_65 = tfobj20_65.est_nonstat(fl = 10,fu = 90,Nf = 30,olap = 2)
figs_MxxTF20_65   = ts.plot_nonstat(Mxx20_65,func = 'Mxx')

# FDS analysis
fds_sig  = sig.est_fds(Nf = np.arange(10,100,2))
sig.plot_fds()

# statistical response analysis
out20_65 = sig.der_statResponse(fpsd,tf20_65)
print('reponse kurtosis (statistical)',[NSM[1]['kurtosis'] for NSM in out20_65])
print('reponse kurtosis (from realization)',respsig20_65.stat['kurtosis'])
