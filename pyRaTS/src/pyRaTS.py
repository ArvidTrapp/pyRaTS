# -*- coding: utf-8 -*-

"""
PYthon module for the processing of RAndom vibration TimeSeries - pyRaTS
Version: 0.2
@author: Arvid Trapp, Munich University of Applied Sciences
arvid.trapp@hm.edu
"""

from cmath import log
from subprocess import call
import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
import scipy as sp
import rainflow as rf
import FLife
import warnings
from tqdm import tqdm
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from datetime import datetime

###############################################################################
'''
supporting functions / 'static methods' (no need of self object)
- est_specMoms:             calculate spectral moments for given PSD
- est_dirlikD:              estimate fatigue damage via Dirlik method

- get_tfSDOF:               obtain basic SDOF-system's transfer function
- get_twoSidedFourierCoeff: mirror one-sided Fourier coefficients
- get_tf:                   prepare linear transfer function
- get_nonstat:              postprocess calculation of non-stationarity matrix
- get_GaussianSeries:       generate stationary Gaussian time series
- get_weightFunc:           get weighting function (optimization)
- get_lsQsDirlik:           get load spectra for quasi-stationary loading using Dirlik

- soe_lsEquivalent:         SOE for load spectral equivalence
- soe_FDSEquivalent:        SOE for FDS equivalence

- plot_fds:                 plot FDS of (multiple) signals
- plot_nonstat:             plot non-stationarity matrix
- plot_lsEquivalent:        plot SOE: load spectral equivalence 
- plot_FDSEquivalent:       plot SOE: FDS equivalence 

- txt_lsEquivalent:         text output SOE: load spectral equivalence 
'''
optiparam = 0

def est_specMoms(psd,fpsd):
    '''
    calculate spectral moments for given PSD
    INPUTS:
        :param psd:     array_like
            input-PSD
        :param fpsd:    array_like
            input frequency vector corresponding to PSD
    OUTPUTS:
        out: dictionary with keys {specm0,specm1,specm2,specm4,ny_0,ny_p,alpha1,alpha2}
    '''
    psd,fpsd = np.array(psd),np.array(fpsd)
    dfpsd    = np.mean(np.diff(fpsd))
    out = {'specm0': np.sum(psd*dfpsd), 
           'specm1': np.sum(psd*fpsd*dfpsd),
           'specm2': np.sum(psd*fpsd**2*dfpsd),
           'specm4': np.sum(psd*fpsd**4*dfpsd)}
    out['ny_0']   = np.sqrt(out['specm2']/out['specm0'])
    out['ny_p']   = np.sqrt(out['specm4']/out['specm2'])
    out['alpha1'] = out['specm1']/ np.sqrt(out['specm0']*out['specm2'])
    out['alpha2'] = out['specm2']/ np.sqrt(out['specm0']*out['specm4']) 
    return out     

def est_dirlikD(psd,fpsd,m=1): 
    '''
    estimate fatigue damage via Dirlik method
    INPUTS:
        :param psd:     array_like
            input-PSD
        :param fpsd:    array_like
            input frequency vector corresponding to PSD
        :param m:       scalar
            SN-curve exponent
    OUTPUTS:
        out: dictionary with keys {specm0,specm1,specm2,specm4,ny_0,ny_p,alpha1,alpha2}
    SOURCES:
        - T. Dirlik, Application of computers in fatigue analysis. PhD Thesis, University of Warwick, 1985
    '''    
    SPM = est_specMoms(psd,fpsd)
    Xm  = SPM['specm1']/SPM['specm0']*np.sqrt(SPM['specm2']/SPM['specm4'])
    D1  = 2*(Xm-SPM['alpha2']**2)/(1+SPM['alpha2']**2)
    R   = (SPM['alpha2']-Xm-D1**2)/(1-SPM['alpha2']-D1+D1**2)
    D2  = (1-SPM['alpha2']-D1+D1**2)/(1-R)
    D3  = 1-D1-D2
    Q   = 1.25*(SPM['alpha2']-D3-D2*R)/D1 
    D   = SPM['ny_p']*np.sqrt(SPM['specm0'])**m*(D1*Q**m*sp.special.gamma(1+m)+np.sqrt(2)**m*sp.special.gamma(1+m/2)*(D2*abs(R)**m+D3))
    return D

def get_tfSDOF(fvec, fD = 50, D = 0.03,func = 'acc2dis'):
    '''
    obtain single-degree-of-freedom (SDOF)-system transfer function for given frequency vector input
    INPUTS:
        :param fvec:    array_like
            input frequency vector
        :param fD:      scalar
            desired resonant frequency
        :param D:       scalar
            damping coefficient
        :param func:    string
            choice of input-output relation 
    OUTPUTS:
        H: transfer function for SDOF-system
    SOURCES:
        P. Wolfsteiner, A. Trapp, Fatigue life due to non-Gaussian excitation – An analysis of the fatigue damage spectrum using higher order spectra, International Journal of Fatigue 127 (2019) 203–216
    '''      
    Omega   = 2*np.pi*fvec
    omega_0 = fD*2*np.pi/np.sqrt(1-D**2)
    if func == 'acc2dis':
        H = 1/(-Omega**2+2*D*omega_0*1j*Omega+omega_0**2)
    elif func == 'acc2dis4mm':
        H = 1000/(-Omega**2+2*D*omega_0*1j*Omega+omega_0**2)
    elif func == 'acc2vel':
        H = 1j*Omega/(-Omega**2+2*D*omega_0*1j*Omega+omega_0**2)
    elif func == 'acc2acc':    
        H = -Omega**2/(-Omega**2+2*D*omega_0*1j*Omega+omega_0**2)
    return H  

def get_analyticalSignal(Xts,ffper,fl,fu):
    ''' get analytical signal
    INPUTS:
    :param Xts:   vector
        two-sided Fourier coefficients
    :param ffper: vector
        two-sided frequency vector
    :param fl:    scalar
        lower frequency
    :param fu:    scalar
        upper frequency 
    OUTPUTS:  
        array-like
            tapered analytical signal 
    SOURCE:
        A. Trapp, P. Wolfsteiner, Estimating higher-order spectra via filtering-averaging, Mechanical Systems and Signal Processing 150 (2021)
    '''
    N      = len(Xts)
    if N%2:
        fs = 2*-ffper[0]+ffper[2]-ffper[1]
    else:
        fs = 2*-ffper[0]
    Xfilt  = np.zeros(N,dtype='cdouble')
    fmask  = (ffper >= fl) & (ffper < fu)
    Xfilt[fmask] = 2*Xts[fmask]
    return np.fft.ifft(np.fft.ifftshift(Xfilt*fs))   

def get_posTaperedAnalyticalSignal(Xts,ffper,fl,fu,wfunc=sp.signal.windows.boxcar):
    ''' get positive(!) tapered analytical signal
    INPUTS:
    :param Xts:   vector
        two-sided Fourier coefficients
    :param ffper: vector
        two-sided frequency vector
    :param fl:    scalar
        lower frequency
    :param fu:    scalar
        upper frequency 
    :param wfunc: function handle
        window function
    OUTPUTS:  
        array-like
            tapered analytical signal 
    '''
    N      = len(Xts)
    if N%2:
        fs = 2*-ffper[0]+ffper[2]-ffper[1]
    else:
        fs = 2*-ffper[0]
    Xfilt  = np.zeros(N,dtype='cdouble')
    fmask  = (ffper >= fl) & (ffper < fu)
    win    = wfunc(np.sum(fmask))        
    Xfilt[fmask] = win*2*Xts[fmask]
    Xfilt[ffper < 0] = 0
    return np.fft.ifft(np.fft.ifftshift(Xfilt*fs))  

def get_twoSidedFourierCoeff(Xos,N):
    '''
    mirror one-sided Fourier coefficients
    >>> for internal use <<< 
    INPUTS:
        :param Xos:    array_like
            one-sided Fourier coefficients
        :param N:      scalar
            number of samples    
    OUTPUTS:
        Xts: two-sided Fourier coefficients
    '''       
    if N % 2 == 0:
        Xts = np.append(Xos, np.flip(np.conjugate(Xos[1:-1])))
    else:
        Xts = np.append(Xos, np.flip(np.conjugate(Xos[1:])))
    return Xts   

def get_tf(f,H,feval):
    ''' prepare/preprocess a linear transfer function for calculating 
    structural responses
    >>> for internal use <<< 
    INPUTS:
        :param f:    array_like
            frequency vector
        :param H:    array_like
            frequency response function
        :param feval:    array_like
            desired frequency vector            
    OUTPUTS:
        Href: preprocessed frequency response function    
    '''
    dfeval = np.mean(np.diff(feval))
    lowerdiff = f[0] - feval[0]     # Check difference
    upperdiff = f[-1] - feval[-1]   # Check difference
    
    if lowerdiff > 2*dfeval:  # if feval has larger boundries add zeros
       f = np.append([feval[0],f[0]-dfeval],f) 
       H = np.append([0,0],H)
    elif (lowerdiff > dfeval) & (f[0] != feval[0]):    
       f = np.append(feval[0],f) 
       H = np.append(0,H)
       
    if upperdiff < -2*dfeval:
       f = np.append(f,[f[-1]+dfeval,feval[-1]]) 
       H = np.append(H,[0,0])    
    elif f[-1] != feval[-1]:
       f = np.append(f,feval[-1]) 
       H = np.append(H,0) 

    if np.any(f != feval):
        Href = np.interp(feval,f,H)
    else:
        Href = H
    return Href      

def get_nonstat(mue2,mue4,mue4sym):
    ''' postprocess calculation of non-stationarity matrix
    >>> for internal use <<<  
    SOURCES:
        A. Trapp, P. Wolfsteiner, Frequency-domain characterization of varying random vibration loading by a non-stationarity matrix, International Journal of Fatigue 146 (2021)
    '''
    eps = np.finfo(float).eps
    Nf  = len(mue2)
    nonstat = {'Gxx': mue2}
    real_mue4           = np.real(mue4)
    real_mue4[real_mue4 < eps] = eps
    nonstat['Rxx']      = 4*real_mue4
    nonstat['Rxx_sym']  = 4*mue4sym
    nonstat['Rxx_stat'] = np.outer(mue2,mue2)
    nonstat['Rxx_stat'][nonstat['Rxx_stat'] < 4*eps] = 4*eps

    nonstat['Axx']      = nonstat['Rxx']/nonstat['Rxx_stat']
    nonstat['Axx_sym']  = nonstat['Rxx_sym']/nonstat['Rxx_stat']
    nonstat['Axx_stat'] = nonstat['Rxx_stat']/nonstat['Rxx_stat']
    nonstat['Mxx']      = 3*nonstat['Rxx']
    nonstat['Mxx_sym']  = 3*nonstat['Rxx_sym']
    nonstat['Mxx_stat'] = 3*nonstat['Rxx_stat']
    nonstat['kurtosis'] = np.sum(nonstat['Mxx'])/np.sum(nonstat['Rxx_stat'])
    nonstat['Kxx']      = nonstat['Rxx']-nonstat['Rxx_stat']
    nonstat['Kxx_sym']  = nonstat['Rxx_sym']-nonstat['Rxx_stat']
    nonstat['Kxx_stat'] = nonstat['Rxx_stat']-nonstat['Rxx_stat'] # np.zeros([Nf, Nf])
    nonstat['Cxx']      = 3*(nonstat['Rxx']-nonstat['Rxx_stat'])
    nonstat['Cxx_sym']  = 3*(nonstat['Rxx_sym']-nonstat['Rxx_stat'])
    nonstat['Cxx_stat'] = 3*(nonstat['Rxx_stat']-nonstat['Rxx_stat']) # np.zeros([Nf, Nf])
    diag_m4             = np.sqrt(np.diag(real_mue4))
    divisMX             = np.outer(diag_m4,diag_m4)
    divisMX[divisMX < eps]  = eps
    mue4_norm           = real_mue4/divisMX
    mue4sym_norm        = mue4sym/divisMX
    nonstat['rhoxx']    = mue4_norm+np.diag(np.diag(nonstat['Axx'])-1)
    nonstat['rhoxx_sym']= mue4sym_norm+np.diag(np.diag(nonstat['Axx'])-1)
    nonstat['Nf']       = Nf
    return nonstat       

def get_GaussianSeries(inp):
    '''
    generate stationary Gaussian time series
    INPUTS:
        :param inp:    dictionary
            dictionary with keys:
                fpsd - input frequency vector -> fs = 2*fpsd[-1]
                psd  - input PSD
                N - number of samples (optional)
                T - duration of time series (optional)
    OUTPUTS:
        x:     array-like (vector)      
            random stationary Gaussian series
        tvec:   array-like (vector)       
            time vector
    '''  
    inp.setdefault('fs',2*inp['fpsd'][-1])  
    if 'N' in inp:
        inp.setdefault('T',inp['N']/inp['fs'])             
    inp.setdefault('N',round(inp['fs']*inp['T']))

    tvec        = np.arange(0,inp['N'])/inp['fs']
    N_os        = inp['N'] // 2 + 1 
    dfpsd       = np.mean(np.diff(inp['fpsd']))
    if inp['fpsd'][0] != 0:
        inp['fpsd'] = np.concatenate(0,inp['fpsd'])
        inp['psd']  = np.concatenate(0,inp['psd'])
    fvec = np.linspace(0,inp['fs']/2,N_os)    
    mue2 = np.sum(inp['psd']*dfpsd)
    per  = np.interp(fvec,inp['fpsd'],inp['psd'])
    per  = mue2/(np.sum(per/inp['T'])) * per
    Xabs = np.sqrt(per * inp['T']/ 2) 
    Xos  = Xabs * np.exp(1j * np.random.uniform(-np.pi, np.pi, len(Xabs)))
    x    = np.fft.irfft(Xos*inp['fs'], n=inp['N'])

    if mue2 <= 0:
        warnings.warn('PSD corrupted')
    return x,tvec

def get_weightFunc(x,inp):
    ''' get weighting function (optimization)
    >>> for internal use <<<  
    SOURCES:
        Wolfsteiner, P., Trapp, A., and Fertl, L., 'Random vibration fatigue with non-stationary loads using an extended fatigue damage spectrum with load spectra', ISMA 2022 Conference
    '''
    p = x[:inp['R']]
    wfunc = x[inp['R']:].reshape((inp['R'], inp['res']*inp['Nf']))
    wfunc_it = np.zeros((inp['R'],len(inp['fpsd'])))
    fAxis = np.linspace(0,inp['fend'],inp['res']*inp['Nf'])
    fAxis = np.append(fAxis,[fAxis[-1] + np.mean(np.diff(fAxis)),inp['fpsd'][-1]])
    for rr in range(inp['R']):
        interpfunc   = sp.interpolate.interp1d(fAxis,np.append(wfunc[rr],[1,1]))
        wfunc_it[rr] = interpfunc(inp['fpsd'])
    return wfunc_it

def get_lsQsDirlik(psd,fpsd,si_X,weights): 
    ''' get load spectra for quasi-stationary loading using Dirlik 
    >>> for internal use >>>
    '''
    dfpsd = fpsd[1]-fpsd[0]
    out = {'specm0': np.sum(psd*dfpsd,axis=1), 
           'specm1': np.sum(psd*[fpsd]*dfpsd,axis=1),
           'specm2': np.sum(psd*[fpsd**2]*dfpsd,axis=1),
           'specm4': np.sum(psd*[fpsd**4]*dfpsd,axis=1)}
    out['ny_0']   = np.sqrt(out['specm2']/out['specm0'])
    out['ny_p']   = np.sqrt(out['specm4']/out['specm2'])
    out['alpha1'] = out['specm1']/ np.sqrt(out['specm0']*out['specm2'])
    out['alpha2'] = out['specm2']/ np.sqrt(out['specm0']*out['specm4'])     
    Xm = out['specm1']/out['specm0']*np.sqrt(out['specm2']/out['specm4'])
    D1 = 2*(Xm-out['alpha2']**2)/(1+out['alpha2']**2)
    R  = (out['alpha2']-Xm-D1**2)/(1-out['alpha2']-D1+D1**2)
    D2 = (1-out['alpha2']-D1+D1**2)/(1-R)
    D3 = 1-D1-D2
    Q  = 1.25*(out['alpha2']-D3-D2*R)/D1
    
    Z  = si_X/np.sqrt(out['specm0'])[:,None] 

    out['DK_ls']   = D1[:,None]*np.exp(-Z/Q[:,None]) + D2[:,None]*np.exp(-Z**2/(2*R[:,None]**2)) + D3[:,None]*np.exp(-Z**2/2)
    out['DK_ls_ps']   = out['DK_ls']*out['ny_p'][:,None]
    return np.matmul(weights,out['DK_ls_ps'])

def soe_lsEquivalent(x,inp):
    ''' system of equations (SOE) for load spectral equivalence
    >>> for internal use <<< 
    SOURCES:
        Wolfsteiner, P., Trapp, A., and Fertl, L., 'Random vibration fatigue with non-stationary loads using an extended fatigue damage spectrum with load spectra', ISMA 2022 Conference 
    '''    
    psd_it = inp['psd'] * get_weightFunc(x,inp)
    res = np.zeros(inp['Nf'])
    for ff in range(inp['Nf']):
        resp  = psd_it*inp['H'][ff]
        si_X  = np.linspace(0,2*inp['si_eval'][ff,0],1200)
        ls_it = get_lsQsDirlik(resp,inp['fpsd'],si_X,x[:inp['R']]*inp['T'])
        si_it = np.interp(inp['ni_eval'],np.flip(ls_it),np.flip(si_X))
        res[ff] = np.sum((inp['si_eval'][ff]-si_it)**2)*inp['nfunc'][ff]**2
    return np.sum(res)/np.sum(inp['si_eval'])

def soe_FDSEquivalent(x,inp):  
    ''' system of equation (SOE) for FDS equivalance 
    >>> for internal use >>>
    SOURCES:
        Wolfsteiner, P., Trapp, A., and Fertl, L., 'Random vibration fatigue with non-stationary loads using an extended fatigue damage spectrum with load spectra', ISMA 2022 Conference
    '''

    '''interpolation'''
    x = np.append(x,[1,1])
    interpfunc   = sp.interpolate.interp1d(inp['fAxis'],x)
    psd_it = inp['psd']*interpfunc(inp['fpsd'])

    '''damage calculation'''
    d_vgl = []
    for ff in np.arange(len(inp['H'])):
        d_vgl.append((inp['T']*est_dirlikD(psd_it*inp['H'][ff],inp['fpsd'],inp['m']))**(1/inp['m']))

    '''residuals and normalization'''
    res = (inp['reffds']['sig_eq'].T-d_vgl)**2*inp['nfunc']**2

    '''return normalized MSE'''
    return np.sum(res)/np.sum(inp['reffds']['sig_eq']**2)

def plot_fds(inp,**kwargs):  
    '''
    plot FDS of (multiple) signals
    INPUTS:
        :param inp:    dictionary-(array)
            FDS calculations
    '''  
    try:
        for kk in range(len(inp[0]['m'])): 
            fig = plt.figure()
            for fds in inp:
                try:
                    plt.plot(fds['fD'],fds['sig_eq'][:,kk],label = fds['name'])
                except:
                    warnings.warn('SN-exponents do not fit')
            plt.title('FDS for SN-exponent m = ' + str(fds['m'][kk]))        
            plt.xlabel('resonant frequency $f_D$ [Hz]')
            plt.ylabel('pseudo-damage')
            plt.legend()
            plt.grid(True)
            fig.show()
    except:
        print('Input:',inp)
        warnings.warn('function "plot_fds()" cannot be carried out')

def plot_nonstat(inp,func = 'Mxx',lim = None):
    '''
    plot non-stationarity matrix
    INPUTS:
        :param inp:    dictionary
            non-stationary matrix dictionary
        :param func:   string
            function(s) to be plotted
    '''  
    if type(func) == str:
            func = [func] # make method iterable
    for fun in func:
        if fun == 'Rxx_asym':
            quant = [inp['Rxx'],inp['Rxx_stat']]
            def_title = 'non-stationarity matrix'
            def_clabl1 = '$R_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: {p:.2f}%) [('+inp['unit']+')$^4$]'.format(p = inp['m4rep'])
            def_clabl2 = '$R_{'+2*inp['var']+',stat}$ [('+inp['unit']+')$^4$]'
        elif fun == 'Rxx':
            quant = [inp['Rxx_sym']]
            def_title = 'non-stationarity matrix (symmetric)'
            def_clabl1 = '$R_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: {p:.2f}%) [('+inp['unit']+')$^4$]'.format(p = inp['m4rep'])
        elif fun == 'Rxx_full':
            quant = [inp['Rxx_sym'],inp['Rxx_stat']]
            def_title = 'non-stationarity matrix (symmetric)'
            def_clabl1 = '$R_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: {p:.2f}%) [('+inp['unit']+')$^4$]'.format(p = inp['m4rep'])
            def_clabl2 = '$R_{'+2*inp['var']+',stat}$ [('+inp['unit']+')$^4$]'
        elif fun == 'Mxx_asym':
            quant = [inp['Mxx'],inp['Mxx_stat']]
            def_title = 'non-stationarity moment matrix'
            def_labl1 = '$\mu_{{4,ns}}$ = {val:.2f}'.format(val = np.sum(quant[0]))
            def_labl2 = '$\mu_4$ = {val:.2f}'.format(val = inp['moms'][4]) 
            def_labl3 = '$\mu_{{4,stat}}$ = {val:.2f}'.format(val = 3*inp['moms'][2]**2) 
            def_clabl1 = '$M_{{'+2*inp['var']+'}}$ [('+inp['unit']+')$^4$] (' + def_labl1 +'/'+ def_labl2 + '/$\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
            def_clabl2 = '$M_{'+2*inp['var']+'}$ [('+inp['unit']+')$^4$] (' + def_labl3 +'/'+ def_labl2 + ')'
        elif fun == 'Mxx':
            quant = [inp['Mxx_sym']]
            def_title = 'non-stationarity moment matrix (symmetric)'
            def_labl1 = '$\mu_{{4,ns}}$ = {val:.2f}'.format(val = np.sum(quant[0]))
            def_labl2 = '$\mu_4$ = {val:.2f}'.format(val = inp['moms'][4]) 
            def_clabl1 = '$M_{{'+2*inp['var']+'}}$ [('+inp['unit']+')$^4$] (' + def_labl1 +'/'+ def_labl2 + '/$\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
        elif fun == 'Mxx_full':
            quant = [inp['Mxx_sym'],inp['Mxx_stat']]
            def_title = 'non-stationarity moment matrix (symmetric)'
            def_labl1 = '$\mu_{{4,ns}}$ = {val:.2f}'.format(val = np.sum(quant[0]))
            def_labl2 = '$\mu_4$ = {val:.2f}'.format(val = inp['moms'][4]) 
            def_labl3 = '$\mu_{{4,stat}}$ = {val:.2f}'.format(val = 3*inp['moms'][2]**2) 
            def_clabl1 = '$M_{{'+2*inp['var']+'}}$ [('+inp['unit']+')$^4$] (' + def_labl1 +'/'+ def_labl2 + '/$\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
            def_clabl2 = '$M_{'+2*inp['var']+'}$ [('+inp['unit']+')$^4$] (' + def_labl3 +'/'+ def_labl2 + ')'
        elif fun == 'Cxx_asym':
            quant = [inp['Cxx'],inp['Cxx_stat']]
            def_title = 'non-stationarity cumulant matrix'
            def_labl1 = '$c_{{4,ns}}$ = {val:.2f}'.format(val = np.sum(quant[0]))
            def_labl2 = '$c_4$ = {val:.2f}'.format(val = inp['moms'][4]-3*inp['moms'][2]**2) 
            def_labl3 = '$c_{{4,stat}}$ = {val:.2f}'.format(val = 0) 
            def_clabl1 = '$C_{{'+2*inp['var']+'}}$ [('+inp['unit']+')$^4$] (' + def_labl1 +'/'+ def_labl2 + '/$\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
            def_clabl2 = '$C_{'+2*inp['var']+'}$ [('+inp['unit']+')$^4$] (' + def_labl3 +'/'+ def_labl2 + ')'
        elif fun == 'Cxx_full':
            quant = [inp['Cxx_sym'],inp['Cxx_stat']]
            def_title = 'non-stationarity cumulant matrix (symmetric)'
            def_labl1 = '$c_{{4,ns}}$ = {val:.2f}'.format(val = np.sum(quant[0]))
            def_labl2 = '$c_4$ = {val:.2f}'.format(val = inp['moms'][4]-3*inp['moms'][2]**2) 
            def_labl3 = '$c_{{4,stat}}$ = {val:.2f}'.format(val = 0) 
            def_clabl1 = '$C_{{'+2*inp['var']+'}}$ [('+inp['unit']+')$^4$] (' + def_labl1 +'/'+ def_labl2 + '/$\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
            def_clabl2 = '$C_{'+2*inp['var']+'}$ [('+inp['unit']+')$^4$] (' + def_labl3 +'/'+ def_labl2 + ')'
        elif fun == 'Cxx':
            quant = [inp['Cxx_sym']]
            def_title = 'non-stationarity cumulant matrix (symmetric)'
            def_labl1 = '$c_{{4,ns}}$ = {val:.2f}'.format(val = np.sum(quant[0]))
            def_labl2 = '$c_4$ = {val:.2f}'.format(val = inp['moms'][4]-3*inp['moms'][2]**2) 
            def_clabl1 = '$C_{{'+2*inp['var']+'}}$ [('+inp['unit']+')$^4$] (' + def_labl1 +'/'+ def_labl2 + '/$\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
        elif fun == 'Axx_asym':
            quant = [inp['Axx'],inp['Axx_stat']]
            def_title  = '    modulation matrix'
            def_clabl1 = '$A_{'+2*inp['var']+'}}$ [-] ($\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
            def_clabl2 = '$A_{'+2*inp['var']+',stat}$ [-]'
        elif fun == 'Axx_full':
            quant = [inp['Axx_sym'],inp['Axx_stat']]
            def_title = '    modulation matrix (symmetric)'
            def_clabl1 = '$A_{'+2*inp['var']+'}}$ [-] ($\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
            def_clabl2 = '$A_{'+2*inp['var']+',stat}$ [-]'  
        elif fun == 'Axx':
            quant = [inp['Axx_sym']]
            def_title = '    modulation matrix (symmetric)'
            def_clabl1 = '$A_{'+2*inp['var']+'}}$ [-] ($\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
        elif fun == 'Kxx_asym':
            quant = [inp['Kxx'],inp['Kxx_stat']]
            def_title  = 'non-stationarity covariance matrix'
            def_clabl1 = '$K_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: ' + str("%.2f"% inp['m4rep']) + '%) [('+inp['unit']+')$^4$]'
            def_clabl2 = '$K_{'+2*inp['var']+'}$ [('+inp['unit']+')$^4$]'
        elif fun == 'Kxx_full':
            quant = [inp['Kxx_sym'],inp['Kxx_stat']]
            def_title = 'non-stationarity covariance matrix (symmetric)'
            def_clabl1 = '$K_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: ' + str("%.2f"% inp['m4rep']) + '%) [('+inp['unit']+')$^4$]'
            def_clabl2 = '$K_{'+2*inp['var']+'}$ [('+inp['unit']+')$^4$]'
        elif fun == 'Kxx':
            quant = [inp['Kxx_sym']]
            def_title = 'non-stationarity covariance matrix (symmetric)'
            def_clabl1 = '$K_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: ' + str("%.2f"% inp['m4rep']) + '%) [('+inp['unit']+')$^4$]'
        elif fun == 'rhoxx_asym':
            quant = [inp['rhoxx'],inp['Axx_stat']]
            def_title = 'non-stationarity synchronicity matrix'
            def_clabl1 = '$\\rho_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: {p:.2f}%) [-]'.format(p = inp['m4rep'])
            def_clabl2 = '$A_{'+2*inp['var']+',stat}$ [-]'   
        elif fun == 'rhoxx_full':
            quant = [inp['rhoxx_sym'],inp['Axx_stat']]
            def_title = 'non-stationarity synchronicity matrix (symmetric)'
            def_clabl1 = '$\\rho_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: {p:.2f}%) [-]'.format(p = inp['m4rep'])
            def_clabl2 = '$A_{'+2*inp['var']+',stat}$ [-]'   
        elif fun == 'rhoxx':
            quant = [inp['rhoxx_sym']]
            def_title = 'non-stationarity synchronicity matrix (symmetric)'
            def_clabl1 = '$\\rho_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: {p:.2f}%) [-]'.format(p = inp['m4rep'])
        cnt = 0 
        figs = []   
        for cc,nonstat in enumerate(quant):
            cnt = cnt +1 
            fig = plt.figure()
            if lim is not None:
                pos = plt.imshow(nonstat, extent=[inp['f_start'], inp['f_end'], inp['f_end'], inp['f_start']],vmin=lim[0], vmax=lim[1])   
            else:
                pos = plt.imshow(nonstat, extent=[inp['f_start'], inp['f_end'], inp['f_end'], inp['f_start']])   
            plt.xlabel("$f_1$ [Hz]")
            plt.ylabel("$f_2$ [Hz]")
            plt.grid(True)
            if cnt == 1:
                plt.title(def_title)
                cb = fig.colorbar(pos,label = def_clabl1)
            else: 
                plt.title(def_title[4:])
                cb = fig.colorbar(pos,label = def_clabl2)
            fig.show()
            figs.append(fig)     
    return figs

def plot_lsEquivalent(x,inp=None):
    ''' plot function for SOE: load spectral equivalence 
    >>> for internal use >>>
    '''
    global optiparam

    psd_it = optiparam['psd'] * get_weightFunc(x,optiparam)
    res = np.zeros(optiparam['Nf'])
    for ff in range(optiparam['Nf']):
        resp    = psd_it*optiparam['H'][ff]
        si_X    = np.linspace(0,1.5*optiparam['si_eval'][ff,0],1000)
        ls_it   = get_lsQsDirlik(resp,optiparam['fpsd'],si_X,x[:optiparam['R']]*optiparam['T'])
        si_it   = np.interp(optiparam['ni_eval'],np.flip(ls_it),np.flip(si_X))
        res[ff] = np.sum((optiparam['si_eval'][ff]-si_it)**2)
    plt.figure()
    plt.stem(res)
    plt.show()   
    plt.title('residuuals')

def plot_FDSEquivalent(x,inp = None): 
    ''' plot function for SOE: FDS equivalance 
    >>> for internal use >>>
    '''   
    global optiparam
    interpfunc   = sp.interpolate.interp1d(optiparam['fAxis'],x)
    psd_it       = optiparam['psd']*interpfunc(optiparam['fpsd'])
    d_vgl = []
    for ff in np.arange(len(optiparam['H'])):
        d_vgl.append((optiparam['T']*est_dirlikD(psd_it*optiparam['H'][ff],optiparam['fpsd'],optiparam['m']))**(1/optiparam['m']))
    res = (optiparam['reffds']['sig_eq']-d_vgl)**2

    plt.figure()
    plt.subplot(211)
    plt.plot(optiparam['fpsd'],optiparam['psd'])
    plt.plot(optiparam['fpsd'],psd_it)
    plt.subplot(212)
    plt.plot(optiparam['reffds']['sig_eq'])
    plt.plot(d_vgl)
    plt.plot(np.squeeze(optiparam['reffds']['sig_eq'].T-d_vgl))
    plt.show()    
    plt.title('residuuals')
    print('Iteration: ',optiparam['it'])
    optiparam['it'] += 1 

def txt_lsEquivalent(x,inp=None):
    ''' text output system of equation (SOE) for FDS equivalance 
    >>> for internal use >>>
    '''
    global optiparam

    psd_it = optiparam['psd'] * get_weightFunc(x,optiparam)
    res = np.zeros(optiparam['Nf'])
    for ff in range(optiparam['Nf']):
        resp  = psd_it*optiparam['H'][ff]
        si_X  = np.linspace(0,1.5*optiparam['si_eval'][ff,0],1000)
        ls_it = get_lsQsDirlik(resp,optiparam['fpsd'],si_X,x[:optiparam['R']]*optiparam['T'])
        si_it = np.interp(optiparam['ni_eval'],np.flip(ls_it),np.flip(si_X))
        res[ff] = np.sum((optiparam['si_eval'][ff]-si_it)**2)
    print(np.sum(res))  
    
    print('Iteration: ',optiparam['it'])
    optiparam['it'] += 1 
                          
class tsarr:
    '''
        Attributes
        ----------
        x:         (load/response)-series
        t:         time vector
        dt:        time step
        fs:        sampling frequency
        T:         duration of timeseries
        N:         number of samples
        var:       variable name
        unit:      unit to be displayed in plots
        prob:      estimate of probability density function (PDF)
        grid:      grid for estimate of PDF
        pgauss:    Gaussian PDF of same variance
        stat:      dictionary of statistical indicators
        psd:       power spectral density (PSD)
        fpsd:      frequency vector for PSD
        dfpsd:     increment of frequency vector for PSD
        fper:      one-sided frequency vector
        ffper:     two-sided frequency vector
        Xos:       one-sided Fourier coefficients
        Xts:       two-sided Fourier coefficients
                
        ----------
        Methods
        ---------- 
        plot():             plot time series
        plot_psd():         plot PSD that was lastly estimated via est_psd()
        plot_X():           plot abs of Fourier coefficients 
        plot_tf():          plot transfer function (interpretation: object is transfer function) 
        plot_ls(method):    plot load spectrum
        plot_fds(idx):      plot FDSs of signal given bei calculation indices
        plot_nonstat():     plot non-stationarity matrix

        est_psd(df):                        estimate power spectral density (PSD) 
        est_stats():                        estimate key statistical values
        est_ls():                           derive/estimate load spectrum via RFC/DK
        est_dirlik(m):                      estimate Dirlik load spectrum and damage 
        est_fds(m,D,method,Nf,func):        estimate Fatigue Damage Spectrum
        est_nonstat(Nf,fl,fu,olap,wfunc):   estimate non-stationary matrix

        der_statResponse(f,H):          derive statistical response (PSD & Non-stat.-Matrix)
        der_sdofResponse(fD, D, func):  derive response timeseries of single-degree-of-freedom system 
        der_sdofTransfer(fD, D, func):  derive single-degree-of-freedom system's transfer functions
        der_response(f,H):              derive timeseries of structural response for given transfer function
        der_highpass(f):                derive timeseries of ideal high pass filtered signal
        der_lowpass(f):                 derive timeseries of ideal low pass filtered signal
        der_bandpass(f):                derive timeseries of ideal band pass filtered signal
        der_scale2max(self,value):      derive timeseries scaled to reference value
        der_deriviative(opt):           derive deriviative timeseries 
        der_lsEquivalent():             derive quasi-stationary load definition on the basis of the load spectra of the Fatigue Damage Spectrum
        der_iFDS():                     derive load definition on the basis of the inverse Fatigue Damage Spectrum
        
        get_analyticalSignal(fl,fu):                    get analytical signal
        get_posTaperedAnalyticalSignal(fl,fu,wfunc):    get positive(!) tapered analytical signal
        get_corrFactors(fbands,olap,wfunc):             get correction factors
        ----------   
    '''
        
    def __init__(self, x, fs = 1000, var="x", unit="m/s$^2$", name = '', **kwargs):
        ''' Initialize underlying signal components that are managed by timeseries class '''
        self.x    = x
        self.N    = len(self.x)
        self.var  = var              # variable name
        if self.var.count('_') > 1:  # clean up underscores
            first_underscore = self.var.find('_')
            cleantext = self.var[first_underscore+1:].replace('_', ',').replace('{', '').replace('}', '')
            self.var = self.var[:first_underscore+1] + '{' + cleantext + '}'
        self.unit = unit             # unit of series
        self.name = name             # name of series
        self.fs  = fs
        self.dt  = 1/self.fs
        self.fNy = self.fs/2
        self.T   = self.N/self.fs
        self.df  = 1/self.T
        self.est_psd()
        
        if name:
            self.name = '(' + name + ')'
        
        self.est_stats()
        self.est_ls()
        self.FDScalcs = []
        
    @property # for storage efficiency
    def t(self):                                        # time-vector
        return np.arange(0,self.N)/self.fs
    
    @property # for storage efficiency
    def Xts(self):                                      # two-sided Fourier coefficients
        Xts_woShift = self.dt*np.fft.fft(self.x)
        return np.fft.fftshift(Xts_woShift)
    
    @property # for storage efficiency
    def Xos(self):                                      # one-sided Fourier coefficients
        Xts_woShift = self.dt*np.fft.fft(self.x)
        Nos         = np.ceil((self.N+1)/2)             # number of one-sided Fourier coefficients
        return Xts_woShift[0:int(Nos)]       
     
    @property # for storage efficiency
    def fper(self):                                     # one-sided frequency vector
        frq_woShift = np.fft.fftfreq(self.N, self.dt)
        Nos         = np.ceil((self.N+1)/2)             # number of one-sided Fourier coefficients
        return np.abs(frq_woShift[0:int(Nos)])
    
    @property # for storage efficiency
    def ffper(self):                                    # two-sided frequency vector
        frq_woShift = np.fft.fftfreq(self.N, self.dt)
        return np.fft.fftshift(frq_woShift) 
                
    def plot(self,ax = None):
        ''' plot timeseries
        INPUTS:
        :param ax:    pyplot axis-object
            for including plot in outer subplot call
        '''
        if ax is None:
            fig, ax = plt.subplots()
            
        legd = "$" + self.var + '$: $\sigma^2$ = {sigq: .3f} [({unit})$^2$]; $ \\beta $ = {beta: .3f} [-]'.format(sigq = self.stat['var'], unit = self.unit,beta = self.stat['kurtosis']) 
        ax.plot(self.t,self.x,'k',label = legd)    
        ax.set_xlabel("t [s]")
        ax.set_ylabel("$" + self.var + "$ [" + self.unit + "]")
        ax.grid(True)
        ax.legend()
        ax.set_title('time series ' + self.name)

    def plot_psd(self,ax = None):
        ''' plot PSD that was lastly estimated via est_psd()
        INPUTS:
        :param ax:    pyplot axis-object
            for including plot in outer subplot call
        '''
        if ax is None:
            fig, ax = plt.subplots()
        varvgl = np.sum(self.psd*self.dfpsd)
        lbl = "$G_{" + self.var + self.var + "}" + ': \sigma^2$ = {:.3f}'.format(self.stat['var']) + ' [(' + self.unit + ')$^2$]'
        ax.plot(self.fpsd,self.psd,'k',label=lbl)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel("$G_{" + self.var + self.var + "}$ [(" + self.unit + ")$^2$/Hz]")
        ax.legend()   
        ax.grid(True)
        ax.set_title('power spectral density ' + self.name)
                
    def plot_X(self,ax = None):
        ''' plot abs of Fourier coefficients 
        INPUTS:
        :param ax:    pyplot axis-object
            for including plot in outer subplot call
        '''
        if ax is None:
            fig, ax = plt.subplots()
        lbl = self.var[0].upper() + self.var[1:]
        ax.plot(self.ffper,abs(self.Xts),'k',label = lbl)   
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel("|$" + self.var[0].upper() + self.var[1:] + "$| [" + self.unit + "/Hz]")
        ax.grid(True)
        ax.set_title('(absolute) Fourier coefficients ' + self.name)   
    
    def plot_prob(self,ax = None):
        ''' plot probability density function (PDF) 
        INPUTS:
        :param ax:    pyplot axis-object
            for including plot in outer subplot call
        '''
        if ax is None:
            fig, ax = plt.subplots()    
        ax.plot(self.grid,self.prob,'k',label = "$p(" + self.var + ")$")
        ax.plot(self.grid,self.pgauss,'g--',label = "$p_g(" + self.var + ")$")
        ax.set_xlabel("$" + self.var + "$ [" + self.unit + "]")
        ax.set_ylabel("$p(" + self.var + ")$ [-]")
        ax.legend()
        ax.grid(True)
        ax.set_title('probability density function ' + self.name)
    
    def plot_ls(self,method = None,ax = None):
        ''' plot load spectrum
        INPUTS:
        :param method:    (list of) function handle(s)
            function for load spectrum estimation
        :param ax:    pyplot axis-object
            for including plot in outer subplot call
        SOURCES:
            - FLife - Vibration Fatigue by Spectral Methods: https://github.com/ladisk/FLife
            - J. Slavič, M. Boltežar, M. Mršnik, M. Cesnik, J. Javh, Vibration fatigue by spectral methods: From structural dynamics to fatigue, Elsevier, 2020
        '''
        if ax is None:
            fig, ax = plt.subplots()    
        ax.semilogx(self.ls['ni_x'],self.ls['si_x'],'r',label='RFC[$'+self.var+'$]')
        ax.semilogx(self.ls['ni_X'],self.ls['si_X'],'k',label='DK[$'+self.var+'$]')
        ax.set_xlabel("cumulated cycles")
        ax.set_ylabel("amplitudes [" + self.unit + "]")
        ax.grid(True) 
        ax.set_title('load spectrum ' + self.name)
        ax.set_xlim(0.5, 1.2*max(self.ls['ni_x']))
        if method != None:
            sd   = FLife.SpectralData(input=(self.psd,self.fpsd))
            if type(method) == type:
                method = [method] # make method iterable
            for pdfesti in method:
                method_name = str(pdfesti)[::-1]
                method_name = method_name[2:method_name.find(".")][::-1]
                spdf = pdfesti(sd).get_PDF(self.ls['si_X'])
                spdf = spdf/np.sum(spdf)
                loadspectrum = np.cumsum(spdf)*self.T*self.stat['ny_p']
                ax.semilogx(loadspectrum,self.ls['si_X'],label = method_name+'[$'+self.var+'$]')   
        ax.legend()
        
    def plot_pseudoDam(self,m = np.linspace(3,8,61),ax = None):
        ''' plot pseudo-damage
        INPUTS:
        :param m:    list
            SN-curve exponents for evaluation
        :param ax:    pyplot axis-object
            for including plot in outer subplot call
        SOURCES:
            ...
        '''
        if ax is None:
            fig, ax = plt.subplots()   
        
        counts = np.hstack((self.ls['ni_x'][0],np.diff(self.ls['ni_x']))) 
        sig_eq = []
        for mm in m:    
            dam_eq = np.sum(counts*self.ls['si_x']**mm)
            sig_eq.append(dam_eq**(1/mm))     
                    
        ax.plot(m,sig_eq,'k',label='s_{eq}[$'+self.var+'$]')
        ax.set_xlabel("SN curve exponents")
        ax.set_ylabel("pseudo-damage [" + self.unit + "]")
        ax.grid(True) 
        ax.set_title('pseudo-damage ' + self.name) 
        ax.legend()
                                   
    def plot_tf(self,plot ='bode',ax=None):
        ''' plot transfer function
        INPUTS:
        :param ax:    pyplot axis-object
            for including plot in outer subplot call
        '''
        if plot.lower() == 'aphase':
            func1 = np.abs
            func2 = lambda x: np.unwrap(np.angle(x))
            tit1  = 'Fourier coefficients (absolute)'
            ylbl1 = "|$" + self.var.upper() + "$| [" + self.unit + "/Hz]"
            tit2  = 'Fourier coefficients (phase)'
            ylbl2 = "$\\varphi[" + self.var.upper() + "$] [rad]"
        elif plot.lower() == 'bode':
            func1 = np.abs
            func2 = lambda x: np.degrees(np.unwrap(np.angle(x)))
            tit1  = 'Bode diagram'
            ylbl1 = "|$" + self.var.upper() + "$| [" + self.unit + "/Hz]"
            tit2  = ''
            ylbl2 = "$\\varphi[" + self.var.upper() + "$] [deg]"           
        else:
            func1 = np.real
            func2 = np.imag
            tit1  = 'Fourier coefficients (real)'
            ylbl1 = "Real[$" + self.var.upper() + "$] [" + self.unit + "/Hz]"
            tit2  = 'Fourier coefficients (imaginary)'
            ylbl2 = "Imag[$" + self.var.upper() + "$] [" + self.unit + "/Hz]"
        if ax is None:
            fig, ax = plt.subplots(2)    
        if type(ax) is not np.ndarray:
            ax = np.array([ax,ax.twinx()]) 
        ax[0].plot(self.fper,func1(self.Xos),'k')   
        ax[0].set_xlabel("f [Hz]")
        ax[0].set_ylabel(ylbl1)
        ax[0].grid(True)
        ax[0].set_title(tit1)
        ax[1].plot(self.fper,func2(self.Xos),'r')   
        ax[1].set_xlabel("f [Hz]")
        ax[1].set_ylabel(ylbl2)
        ax[1].grid(True)
        ax[1].set_title(tit2)
        
    def est_prob(self,bins='auto'):
        ''' calculate probability density function (PDF) 
        -> called by est_stats (thus indirect call by __init__)
        INPUTS:
        :param bins:    scalar or array-like
            number of bins / vector of bin-edges
        '''
        try:
            self.prob, bin_edges = np.histogram(self.x, bins, density=True)
            self.grid = (bin_edges[1:]+bin_edges[0:-1])/2
            self.pgauss = 1/ np.sqrt(self.stat['var']*2*np.pi) * np.exp(-(self.grid- self.stat['mean'])**2 / (2 * self.stat['var']))
        except:
            print("PDF estimation went wrong (can appear for tf objects)")
    
    def est_psd(self,df = 1):
        ''' estimate power spectral density (PSD) 
        -> called by __init__
        '''
        self.fpsd,self.psd = sp.signal.welch(self.x,self.fs,nperseg=round(self.fs/df))
        self.dfpsd = np.mean(np.diff(self.fpsd))    
            
    def est_stats(self):
        ''' estimate key statistical values 
        -> called by __init__     
        SOURCES:
            A.G. Davenport, Note on the distribution of the largest value of a random function with application to gust loading, Proceedings of the Institution of Civil Engineers 28 (2) (1964) 187–196
        '''    
        self.stat = {'mean': np.mean(self.x)}
        self.stat['std'] = np.std(self.x)
        self.stat['var'] = np.var(self.x)
        self.stat['skewness'] = sp.stats.skew(self.x)
        self.stat['kurtosis'] = sp.stats.kurtosis(self.x)+3
        
        # probability density function
        self.est_prob()
        
        # higher-order moments
        maxorder = 7
        moms  = np.zeros(maxorder)
        hmoms = np.zeros(maxorder)

        for ii in range(1,maxorder):
            moms[ii] = sp.stats.moment(self.x,ii)
            if ii > 2 :
                hmoms[ii] = moms[ii]/(sp.stats.moment(self.x,moment=2)**(ii/2))
        self.stat['moms'] = moms
        self.stat['hmoms'] = hmoms
        
        # Extreme values and Peak factors
        self.stat['largestvalue']   = np.max(self.x)
        self.stat['smallestvalue']  = np.min(self.x)
        self.stat['maxvalue']       = np.max(np.abs([self.stat['smallestvalue'],self.stat['largestvalue'] ]))
        self.stat['max']            = self.stat['maxvalue']  
        
        # spectral moments
        self.stat = {**self.stat, **est_specMoms(self.psd,self.fpsd)}
        # peak factors
        termPF                      = np.sqrt(2*np.log(self.stat['ny_0'] *self.T))
        self.stat['peakfactor']     = termPF + 0.57721566490153286061/termPF 
        self.stat['peakfactor_std'] = np.pi/ np.sqrt(6) / termPF
        self.stat['expectedpeak']   = self.stat['peakfactor'] *self.stat['std']
        self.stat['expectedpeakscatter']   = self.stat['peakfactor_std'] *self.stat['std'] 
         
    def est_ls(self,si_X = None):
        ''' derive/estimate load spectrum via RFC/DK
        -> called by __init__
        '''
        cycles = np.array(list(rf.extract_cycles(self.x)))
        cycles = cycles[np.argsort(cycles[:,0])]
        cycles = np.flipud(cycles)
        ampls  = cycles[:,0]/2
        counts = cycles[:,2]   
        self.ls = {'si_x': ampls, 'ni_x': np.cumsum(counts),'mues_x': cycles[:,1]}
        self.ls['npeaks']    = len(ampls)
        self.ls['npeaks_ps'] = self.ls['npeaks']/self.T
        zero_crossings = np.where(np.diff(np.sign(self.x)))[0]
        self.ls['nzerocrossings'] = round(len(zero_crossings)/2)
        self.ls['nzerocrossings_ps'] = self.ls['nzerocrossings']/self.T
        if np.any(si_X) == None: # no specification
            self.ls['si_X'] = np.linspace(1.2*self.ls['si_x'][0],0,1000)
        elif np.isscalar(si_X): # given limit
            self.ls['si_X'] = np.linspace(si_X,0,1000)   
        else: # given vector
            self.ls['si_X'] = si_X
        self.est_dirlik()
            
    def est_dirlik(self,m = 1):
        ''' estimate Dirlik load spectrum and damage 
        INPUTS:
        :param m:    scalar
            S-N-curve exponent
        SOURCES:
            - T. Dirlik, Application of computers in fatigue analysis. PhD Thesis, University of Warwick, 1985
            - D. Benasciutti, Fatigue analysis of random loadings. PhD Thesis, University of Ferrara, 2004
        '''
        Xm = self.stat['specm1']/self.stat['specm0']*np.sqrt(self.stat['specm2']/self.stat['specm4'])
        D1 = 2*(Xm-self.stat['alpha2']**2)/(1+self.stat['alpha2']**2)
        R  = (self.stat['alpha2']-Xm-D1**2)/(1-self.stat['alpha2']-D1+D1**2)
        D2 = (1-self.stat['alpha2']-D1+D1**2)/(1-R)
        D3 = 1-D1-D2
        Q  = 1.25*(self.stat['alpha2']-D3-D2*R)/D1
        Z  = self.ls['si_X']/(np.sqrt(self.stat['specm0']))    

        self.ls['DK_pk']   = (D1/Q*np.exp(-Z/Q) + D2*Z/R**2*np.exp(-Z**2/(2*R**2)) + D3*Z*np.exp(-Z**2/2) ) / (np.sqrt(self.stat['specm0']))
        self.ls['DK_ls']   = D1*np.exp(-Z/Q) + D2*np.exp(-Z**2/(2*R**2)) + D3*np.exp(-Z**2/2)
        self.ls['DK_lsps'] = self.ls['DK_ls']*self.stat['ny_p']
        self.ls['ni_X']    = self.ls['DK_lsps']*self.T 
        self.ls['DK_D']    = self.stat['ny_p']*np.sqrt(self.stat['specm0'])**m*(D1*Q**m*sp.special.gamma(1+m)+np.sqrt(2)**m*sp.special.gamma(1+m/2)*(D2*abs(R)**m+D3))              
    
    def der_lsEquivalent(self,R = 5, Nf = 10, res = 2, D = 0.03, func='acc2dis',optix = None,plot = False,maxit = 100):
        ''' derive quasi-stationary load definition on the basis of the load spectra of the Fatigue Damage Spectrum
        INPUTS:
            :param R:       scalar
                number of stationary processes
            :param Nf:      scalar
                number of resonant-frequency samples  
            :param res:     scalar
                factor for resolution of frequency axis (interpolation)
            :param D:       scalar
                damping coefficient   
            :param func:    string
                type of transfer function 'acc2dis','acc2vel','acc2acc'       
            :param optix:   array-like
                (use) input values
            :param plot:    boolean
                use plot function in optimization
            :param maxit:   scalar
                restrict number of iterations  
        OUTPUT:
            :timeseries object    
        SOURCES:
            Wolfsteiner, P., Trapp, A., and Fertl, L., 'Random vibration fatigue with non-stationary loads using an extended fatigue damage spectrum with load spectra', ISMA 2022 Conference
        '''
        print('LS-FDS Approximation')

        '''get load spectra'''
        ls,lsinterp = self.get_ls(Nf = Nf, D = D, func=func,plot=False)
        self.est_psd() # PSD estimation using standard settings
        Nf = len(ls)  

        '''get transfer functions for optimization'''
        tfuncs = [] # initialize array of transfer functions
        fDs = []    # initialize array of corresponding resonant frequencies
        for resp in ls:
            tf = get_tfSDOF(self.fpsd, fD = resp['fD'], D = D,func = func)
            tfuncs.append(abs(tf)**2)
            fDs.append(resp['fD'])

        '''generate start values'''
        if np.any(optix) == None: 
            x0_p = np.logspace(-1,0,R)
            x0_p = x0_p/sum(x0_p)
            x0_h = [np.random.uniform(0.1,0.5,Nf*res)/p for p in x0_p]
            x0   = np.append(x0_p,x0_h) 
        else:
            print('Use given start values...')
            x0 = optix

        '''set bounds and constraints for optimization '''
        bnds_p = [(0.001,0.99) for rr in range(R)]
        bnds_h = [(0.01,100) for rr in range(R*Nf*res)]
        bnds = bnds_p + bnds_h 
        A = np.zeros(x0.shape)
        A[:R] = 1
        LC = sp.optimize.LinearConstraint(A,0.995,1.005)

        '''normalization case acc2dis'''        
        if func == 'acc2dis':
            nfunc = np.array(fDs)
        elif func == 'acc2acc':
            nfunc = np.ones(Nf)
        
        inp = {'nfunc': nfunc,'fend': resp['fD'],'R': R, 'fpsd': self.fpsd, 'psd': npm.repmat(self.psd,R,1), 'H': tfuncs, 'si_eval': np.array([x['si_x'] for x in lsinterp]), 'Nf': Nf, 'T': self.T,'ni_eval': lsinterp[-1]['ni_x'], 'res': res, 'it': 0}

        '''optimization''' 
        global optiparam # for plot functions
        optiparam = inp
        if plot:
            optix = sp.optimize.minimize(soe_lsEquivalent,x0,args = inp,method = 'SLSQP',bounds = bnds,constraints=LC,callback=plot_lsEquivalent,tol=1e-8,options={'disp': True,'ftol':1e-7,'maxiter':maxit})
        else:
            optix = sp.optimize.minimize(soe_lsEquivalent,x0,args = inp,method = 'SLSQP',bounds = bnds,constraints=LC,callback=txt_lsEquivalent,tol=1e-8,options={'disp': True,'ftol':1e-7,'maxiter':maxit})
        print(optix) # print results    
        
        '''generate PSD'''
        psd_it = optiparam['psd'] * get_weightFunc(optix['x'],optiparam)

        '''generate qs-timeseries object (time-domain realization)''' # TODO
        sigs = []
        for rr in range(R):
            sigs.append({'psd': np.abs(psd_it[rr,:]),'fpsd':inp['fpsd'],'T':optix['x'][rr]*self.T,'var': 'iRLS,proc = ' + str(rr) + '[' + self.var + ']'})

        '''save optimization results'''
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        np.save('res_LSOpti_' + current_time + '.npy', optix['x'])

        '''return object'''
        return sigs
        
    def der_iFDS(self, res = 1, m = 5, Nf = 10, D = 0.03, func='acc2dis',optix = None,plot = False,maxit = 100):
        ''' derive load definition on the basis of the inverse Fatigue Damage Spectrum
        INPUTS:
            :param res:     scalar
                factor for resolution of frequency axis (interpolation)
            :param m:       scalar
                S-N-exponent (stress-life-curve exponent)
            :param Nf:      scalar
                number of resonant-frequency samples  
            :param D:       scalar
                damping coefficient   
            :param func:    string
                type of transfer function 'acc2dis','acc2vel','acc2acc'       
            :param optix:   array-like
                (use) input values
            :param plot:    boolean
                use plot function in optimization
            :param maxit:   scalar
                restrict number of iterations  
        OUTPUT:
            :timeseries object    
        SOURCES:
            Wolfsteiner, P., Trapp, A., and Fertl, L., 'Random vibration fatigue with non-stationary loads using an extended fatigue damage spectrum with load spectra', ISMA 2022 Conference
      
        '''
        print('iFDS Approximation')

        '''generate referencing Fatigue Damage Spectrum (FDS)'''
        reffds = self.est_fds(m = m,Nf = Nf, D = D, func= func)
        cf = reffds['Nf'] # number of fDs (resonant frequencies)

        '''get transfer functions for optimization'''
        tfuncs = [] # initialize array of transfer functions
        fDs = []    # initialize array of corresponding resonant frequencies
        for fD in reffds['fD']:
            tf = get_tfSDOF(self.fpsd, fD = fD, D = D,func = func)
            fDs.append(fD)
            tfuncs.append(abs(tf)**2)

        '''generate start values'''
        if np.any(optix) == None: 
            x0 = np.random.uniform(0.5,5,cf*res)
        else:
            print('Use given start values...')
            x0 = optix

        '''get frequency vector for interpolation'''
        if fD < self.fpsd[-1]: # largest resonant frequency vs. Nyquist frequency 
            fAxis = np.linspace(0,fD,res*cf)
            fAxis = np.append(fAxis,[fAxis[-1] + np.mean(np.diff(fAxis)),self.fpsd[-1]])
        else:
            fAxis = np.linspace(0,self.fpsd[-1],res*cf)
            fAxis = np.append(fAxis,[fAxis[-1] +1,fAxis[-1] +2 ])

        '''set bounds for optimization '''
        bnds = [(0.01,100) for rr in range(cf*res)]

        '''normalization: case acc2dis'''
        if func == 'acc2dis':
            nfunc = np.array(fDs)
        elif func == 'acc2acc':
            nfunc = np.ones(Nf)
        inp = {'nfunc': nfunc, 'fpsd': self.fpsd, 'psd': self.psd, 'H': tfuncs, 'res': res, 'it': 0, 'fAxis': fAxis, 'reffds': reffds, 'T': self.T, 'm': m}

        '''optimization''' 
        if plot:
            global optiparam
            optiparam = inp
            optix = sp.optimize.minimize(soe_FDSEquivalent,x0,args = inp,method = 'SLSQP',callback=plot_FDSEquivalent,bounds = bnds,tol=1e-8,options={'disp': True,'ftol':1e-7,'maxiter':maxit})
        else:
            optix = sp.optimize.minimize(soe_FDSEquivalent,x0,args = inp,method = 'SLSQP',bounds = bnds,tol=1e-8,options={'disp': True,'ftol':1e-7,'maxiter':maxit})
        print(optix) # print results

        '''generate PSD'''
        interpfunc   = sp.interpolate.interp1d(inp['fAxis'],np.append(optix['x'],[1,1]))
        psd_it = self.psd*interpfunc(inp['fpsd'])

        '''generate timeseries object (time-domain realization)'''
        siginp = {'psd': np.abs(psd_it),'fpsd':inp['fpsd'],'T':self.T,'var': 'iFDS[' + self.var + ']'}

        '''save optimization results'''
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        np.save('res_FDSOpti_m' + str(m) + '_' + current_time + '.npy', optix['x'])

        '''return object'''
        return siginp

    def est_fds(self,m =[3,5,7],D = 0.03,method = "RFC", Nf = 10,func = 'acc2dis',**kwargs):
        ''' estimate Fatigue Damage Spectrum 
        INPUTS:
        :param m:       scalar or array-like
            S-N-exponent
        :param D:       scalar
            damping coefficient      
        :param method:  string or function handle
            evaluation method
                'RFC':  Rainflow-counting
                'DK':   Dirlik
                FLife Function handle
        :param Nf:      scalar
            number of resonant-frequency samples       
        :param func:    string
            type of transfer function 'acc2dis','acc2vel','acc2acc'
        OUTPUTS:  
            dictionary
                FDS calculation
        SOURCES:
            P. Wolfsteiner, A. Trapp, Fatigue life due to non-Gaussian excitation – An analysis of the fatigue damage spectrum using higher order spectra, International Journal of Fatigue 127 (2019) 203–216
        '''
        if type(m) != list and type(m) != tuple: # make m iterable
            m = [m]
        if np.isscalar(Nf):
            evalf  = np.linspace(0,round(self.fNy,0),Nf+2)
            evalf = evalf[1:-1]
        else:
            evalf  = Nf
        Nf = len(evalf)
        out = {'fD': evalf,'Nf': Nf, 'm': m, 'D': D, 'method': method, 'tfunc': func}       
        for pastCalc in self.FDScalcs: # check for past calculations
            if pastCalc['Nf'] == out['Nf'] and pastCalc['m'] == out['m'] and pastCalc['D'] == out['D'] and pastCalc['tfunc'] == out['tfunc'] and pastCalc['method'] == out['method']:
                if all(pastCalc['fD'] == out['fD']):
                    print('Calculation has already been carried out')
                    return pastCalc 
        f_0    = out['fD']/np.sqrt(1-out['D']**2)
        sig_eq = np.zeros((Nf,len(m)))
        print('calculate FDS via: '+str(method))
        for ff in tqdm(np.arange(Nf)):
            if method == 'RFC': 
                method_name = method
                resp = self.der_sdofResponse(fD = f_0[ff], D = D, func = func)
                cycles = np.array(list(rf.extract_cycles(resp.x)))
                ampls  = cycles[:,0]/2
                counts = cycles[:,2]
                for mm in np.arange(len(m)):
                    dam_eq =  np.sum(counts*ampls**m[mm])
                    sig_eq[ff,mm] = dam_eq**(1/m[mm]) 
            elif method == 'DK':
                method_name = method
                H    = get_tfSDOF(self.fpsd,D = D, fD = f_0[ff],func = func)
                resp = self.psd*abs(H)**2       
                for mm in np.arange(len(m)):
                    sig_eq[ff,mm] = (self.T*est_dirlikD(resp,self.fpsd,m[mm]))**(1/m[mm]) 
            else:     
                method_name = str(method)[::-1]
                method_name = method_name[2:method_name.find(".")][::-1]
                H    = get_tfSDOF(self.fpsd,D = D, fD = f_0[ff],func = func)
                resp = self.psd*abs(H)**2
                sd   = FLife.SpectralData(input=(resp,self.fpsd))
                damesti = method(sd)
                for mm in np.arange(len(m)):
                    sig_eq[ff,mm] = (self.T/damesti.get_life(C=1,k=m[mm]))**(1/m[mm])                        
        out['sig_eq'] = sig_eq
        out['name'] = method_name + '[$' + self.var + '$]'
        self.FDScalcs.append(out)
        return out  

    def plot_fds(self,indx = None,**kwargs):
        '''
        plot FDSs of signal given bei calculation indices
        INPUTS:
            :param indx:    scalar or list
                indices of FDS calculations
        '''  
        if indx == None:
            plot_fds(self.FDScalcs,**kwargs)
        else:
            fsd2Bplotted = [self.FDScalcs[ii] for ii in indx]
            plot_fds(fsd2Bplotted,**kwargs)

    def plot_nonstat(self,**kwargs):
        ''' plot non-stationarity matrix 
        -> pass to static method
        INPUTS:
            ---
        '''
        plot_nonstat(self.nonstat,**kwargs)

    def est_nonstat(self,Nf = 20, fl = 0, fu = None,olap = 1,wfunc = sp.signal.windows.triang):
        ''' estimate non-stationary matrix
        INPUTS:
        :param Nf:    scalar 
            number of frequency bins
        :param fl:    scalar 
            lower frequency 
        :param fu:    scalar 
            upper frequency 
        :param olap:    scalar 
            overlapping factor 
        :param wfunc: function handle
            window function
        SOURCES:
            A. Trapp, P. Wolfsteiner, Frequency-domain characterization of varying random vibration loading by a non-stationarity matrix, International Journal of Fatigue 146 (2021)
        '''
        if fu == None: # upper frequency limit
            fu     = self.fNy       
        ''' preprocessing '''
        Nf     = Nf*olap
        dfband = fu/Nf
        Nf     = Nf + (olap-1)
        fstart = fl - (olap-1)*dfband
        fend   = fu + (olap-1)*dfband
        fbands = np.linspace(fstart,fend,Nf+1+(olap-1))
        ''' preallocation '''
        alsigs = np.zeros([Nf,self.N],dtype='cdouble')
        mue2   = np.zeros(Nf)
        mue4   = np.zeros([Nf, Nf],dtype='cdouble')
        mue4sym= np.zeros([Nf, Nf])
        c4     = np.zeros([Nf, Nf],dtype='cdouble')
        ''' overlap correction for window func. '''        
        corrf  = self.get_corrFactors(fbands,olap,wfunc)
        ''' calculation ''' 
        Xts   = self.Xts    # for efficiency in upcoming loop
        ffper = self.ffper  # for efficiency in upcoming loop
        for ff in range(Nf):
            alsigs[ff] = 0.5*get_posTaperedAnalyticalSignal(Xts,ffper,fbands[ff],fbands[ff+1+(olap-1)],wfunc)
            mue2[ff]   = 2*np.real(np.mean(self.x*alsigs[ff]))/corrf['mue2']/olap
        print('calculate non-stationarity matrix for number of freq-bins Nf: '+str(Nf))    
        for f1 in tqdm(range(Nf)):
            mult = alsigs[f1]*np.conj(alsigs[f1])
            for f2 in range(Nf):
                entry = self.x*mult*alsigs[f2]
                mean_entry = np.mean(entry)/corrf['mue4']/olap**2
                if f1 == f2:
                    mue4[f1,f2]    = mean_entry/2
                    mue4sym[f1,f2] = np.real(mue4[f1,f2])
                    c4[f1,f2]      = mue4[f1,f2] - mue2[f1]*mue2[f2]
                else:
                    diff = abs(f1-f2)
                    if diff < olap:
                        mue4[f1,f2] = mean_entry/(1+corrf[diff])
                    else:
                        mue4[f1,f2] = mean_entry
                    c4[f1,f2]   = mue4[f1,f2] - mue2[f1]*mue2[f2]
                    mue4sym[f1,f2] = mue4sym[f1,f2]+0.5*np.real(mue4[f1,f2])
                    mue4sym[f2,f1] = mue4sym[f2,f1]+0.5*np.real(mue4[f1,f2])
                    
        ''' saving results '''  
        self.nonstat = get_nonstat(mue2,mue4,mue4sym) 

        fbands[fbands < fl] = fl
        fbands[fbands > fu] = fu
        self.nonstat['f_delta'] = dfband
        self.nonstat['f_mid']   = (fbands[:-olap]+fbands[olap:])/2
        self.nonstat['f_start'] = fl
        self.nonstat['f_end']   = fu
        self.nonstat['var']     = self.var
        self.nonstat['unit']    = self.unit
        self.nonstat['moms']    = self.stat['moms']     
        if fl != 0 and fu!=self.fNy:
            refsig = self.der_bandpass([fl, fu])
            self.nonstat['m4ref'] = refsig.stat['moms'][4]
        else:
            self.nonstat['m4ref'] = self.stat['moms'][4]
        self.nonstat['m4rep'] = 100*np.sum(self.nonstat['Mxx'])/self.nonstat['m4ref']
        return self.nonstat          
       
    def der_statResponse(self,f,H,**kwargs):  
        ''' derive statistical response (PSD & Non-stat.-Matrix) 
            via frequency response function
        INPUTS:
            :param f:    array_like
                frequency vector
            :param H:    array_like
                frequency response function
            :param kwargs: dictionary
                'var': variable of response
                'unit': unit of response
        OUTPUTS:  
            dictionary
                statistical response (PSD,non-stat.-matrix) 
        SOURCES:
            A. Trapp, P. Wolfsteiner, Frequency-domain characterization of varying random vibration loading by a non-stationarity matrix, International Journal of Fatigue 146 (2021)
        '''
        Hpsd = get_tf(f,H,self.fpsd)            
        Gyy = self.psd*np.abs(Hpsd)**2
        try:
            Hnonstat = get_tf(f,H,self.nonstat['f_mid']) 
        except:
            self.est_nonstat(olap = 2)
            Hnonstat = get_tf(f,H,self.nonstat['f_mid'])      
        
        mue2 = self.nonstat['Gxx']*np.abs(Hnonstat)**2
        mue4 = self.nonstat['Rxx']/4*np.outer(np.abs(Hnonstat**2),np.abs(Hnonstat**2))
        mue4sym = self.nonstat['Rxx_sym']/4*np.outer(np.abs(Hnonstat**2),np.abs(Hnonstat**2))
        nonstat = get_nonstat(mue2,mue4,mue4sym)
        nonstat['f_delta'] = self.nonstat['f_delta']
        nonstat['f_mid']   = self.nonstat['f_mid']
        nonstat['f_end']   = self.nonstat['f_end']
        nonstat['f_start'] = self.nonstat['f_start']
        nonstat['var']  = kwargs.get('var',self.var.replace('x','y'))
        nonstat['unit'] = kwargs.get('unit',self.unit)
        nonstat['moms'] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        nonstat['m4ref'],nonstat['m4rep'] = np.nan,np.nan
        return [self.fpsd,Gyy],nonstat
                         
    def get_analyticalSignal(self,fl,fu):
        ''' get analytical signal
        INPUTS:
            :param fl:    scalar
                lower frequency
            :param fu:    scalar
                upper frequency 
        OUTPUTS:  
            array-like
                tapered analytical signal 
        SOURCE:
            A. Trapp, P. Wolfsteiner, Estimating higher-order spectra via filtering-averaging, Mechanical Systems and Signal Processing 150 (2021)
        '''
        return get_analyticalSignal(self.Xts,self.ffper,fl,fu)   

    def get_posTaperedAnalyticalSignal(self,fl,fu,wfunc=sp.signal.windows.boxcar):
        ''' get positive(!) tapered analytical signal
        INPUTS:
        :param fl:    scalar
            lower frequency
        :param fu:    scalar
            upper frequency 
        :param wfunc: function handle
            window function
        OUTPUTS:  
            array-like
                tapered analytical signal 
        '''
        return get_posTaperedAnalyticalSignal(self.Xts,self.ffper,fl,fu,wfunc)   

    def get_corrFactors(self,fbands,olap,wfunc):
        ''' get correction factors 
        >>> for internal use >>>
        '''
        fmask       = (self.ffper >= fbands[0]) & (self.ffper < fbands[0+olap])
        win         = wfunc(np.sum(fmask))   
        idxc        = np.zeros(self.N)
        idxc[fmask] = win 
        out    = {'mue2': np.mean(win), 'mue4': np.mean(win**2)*np.mean(win)}
        for ii in np.arange(1,olap):
            fmask   = (self.ffper >= fbands[ii]) & (self.ffper < fbands[ii+olap])
            win        = wfunc(np.sum(fmask))   
            idx        = np.zeros(self.N)
            idx[fmask] = win 
            out[ii]    = np.sum(idx*idxc)/np.sum(idxc**2)
        return out                 

    def der_sdofResponse(self,fD = 50, D = 0.03,func = 'acc2dis',**kwargs):
        ''' derive response timeseries of single-degree-of-freedom system 
        INPUTS:
        :param fD:   scalar
            resonant frequency
        :param D:    scalar
            damping coefficient            
        :param func: string
            type of transfer function 'acc2dis','acc2vel','acc2acc'
        OUTPUTS:  
            outobj: timeseries object
                output timeseries (response) 
        '''
        H = get_tfSDOF(self.fper,fD = fD, D = D, func = func)
        if func == 'acc2dis':
            if (self.unit == 'm/s$^2$') | (self.unit == 'm/s^2'):
                newunit = 'm'
            else:
                newunit = self.unit + ' $s^2$'
        elif func == 'acc2dis4mm':
            if (self.unit == 'm/s$^2$') | (self.unit == 'm/s^2'):
                newunit = 'mm'
            else:
                newunit = 'm' + self.unit + ' $s^2$'
        elif func == 'acc2vel':
            newunit = self.unit + ' s'
        elif func == 'acc2acc':
            newunit = self.unit       
        y      = np.fft.irfft(self.Xos*H*self.N*self.df,n=self.N)
        if 'var' not in kwargs.keys():
            kwargs = dict(kwargs,var = self.var.replace('x','y'))
        if 'unit' not in kwargs.keys():
            kwargs = dict(kwargs,unit = newunit) 
        if 'name' not in kwargs.keys():
            kwargs = dict(kwargs,name = func + ' SDOF response for ' + self.name.replace('(','').replace(')',''))                               
        outobj = tsarr(y,fs=self.fs,**kwargs)
        return  outobj

    def der_sdofTransfer(self,fD = 50, D = 0.03,func = 'acc2dis',**kwargs):
        ''' derive single-degree-of-freedom system's transfer functions
        INPUTS:
        :param fD:   scalar
            resonant frequency
        :param D:    scalar
            damping coefficient            
        :param func: string
            type of transfer function 'acc2dis','acc2vel','acc2acc'
        OUTPUTS:  
            outobj: timeseries object
                output timeseries (transfer function) 
        '''
        H = get_tfSDOF(self.fper,fD = fD, D = D, func = func)
        # H = get_tfSDOF(self.fpsd,fD = fD, D = D, func = func)
        if func == 'acc2dis':
            newunit = self.unit + '$*s^2$'
        elif func == 'acc2dis4mm':
            newunit = 'm' + self.unit + '*s^2'
        elif func == 'acc2vel':
            newunit = self.unit + '*s'
        elif func == 'acc2acc':
            newunit = self.unit       
        y      = np.fft.irfft(H*self.N*self.df,n=self.N)
        # y      = np.fft.irfft(H*self.N*self.df,n=2*len(H))
        if 'var' not in kwargs.keys():
            kwargs = dict(kwargs,var = self.var.replace('x','y'))
        if 'unit' not in kwargs.keys():
            kwargs = dict(kwargs,unit = newunit)                   
        outobj = tsarr(y,fs=self.fs,**kwargs)
        return  outobj  
    
    def der_tfObj(self,f,H,**kwargs):  
        ''' derive timeseries of structural response for given transfer function
        INPUTS:
            :param f:    array_like
                frequency vector
            :param H:    array_like
                frequency response function
        OUTPUTS:  
            outobj: timeseries object
                output timeseries (transfer function)    
        SOURCES:
            J. Slavič, M. Boltežar, M. Mršnik, M. Cesnik, J. Javh, Vibration fatigue by spectral methods: From structural dynamics to fatigue, Elsevier, 2020          
        '''
        H = get_tf(f,H,self.fper)            
        y = np.fft.irfft(H*self.N*self.df,n=self.N)                    
        outobj = tsarr(y,fs=self.fs,**kwargs)
        return  outobj    

    def der_highpass(self,f,**kwargs):
        ''' derive timeseries of ideal high pass filtered signal
        INPUTS:
            :param f:    scaler
                cuttingfrequency
        '''
        Xfilt  = np.zeros(len(self.Xos),dtype='cdouble')
        fmask  = self.fper >= f
        Xfilt[fmask] = self.Xos[fmask]
        if 'var' not in kwargs.keys():
            kwargs = dict(kwargs,var = self.var + '_{hp}')
        if 'unit' not in kwargs.keys():
            kwargs = dict(kwargs,unit = self.unit) 
        if 'name' not in kwargs.keys():
            kwargs = dict(kwargs,name = 'highpass filtered ts of ' + self.name.replace('(','').replace(')',''))   
        # return timeseries(np.fft.irfft(Xfilt*self.N*self.df,n=self.N),self.t,**kwargs)
        return tsarr(np.fft.irfft(Xfilt*self.N*self.df,n=self.N),fs=self.fs,**kwargs)
    
    def der_lowpass(self,f,**kwargs):
        ''' derive timeseries of ideal low pass filtered signal
        INPUTS:
            :param f:    scaler
                cuttingfrequency
        '''
        Xfilt  = np.zeros(len(self.Xos),dtype='cdouble')
        fmask  = self.fper <= f
        Xfilt[fmask] = self.Xos[fmask]
        if 'var' not in kwargs.keys():
            kwargs = dict(kwargs,var = self.var + '_{lp}')
        if 'unit' not in kwargs.keys():
            kwargs = dict(kwargs,unit = self.unit) 
        if 'name' not in kwargs.keys():
            kwargs = dict(kwargs,name = 'lowpass filtered ts of ' + self.name.replace('(','').replace(')',''))  
        # return timeseries(np.fft.irfft(Xfilt*self.N*self.df,n=self.N),self.t,**kwargs)
        return tsarr(np.fft.irfft(Xfilt*self.N*self.df,n=self.N),fs=self.fs,**kwargs)
    
    def der_bandpass(self,f,**kwargs):
        ''' derive timeseries of ideal band pass filtered signal
        INPUTS:
            :param f:    scalar
                cuttingfrequency
        '''
        Xfilt  = np.zeros(len(self.Xos),dtype='cdouble')
        fmask  = (self.fper >= f[0]) & (self.fper <= f[1])
        Xfilt[fmask] = self.Xos[fmask]
        if 'var' not in kwargs.keys():
            kwargs = dict(kwargs,var = self.var + '_{bp}')
        if 'unit' not in kwargs.keys():
            kwargs = dict(kwargs,unit = self.unit) 
        if 'name' not in kwargs.keys():
            kwargs = dict(kwargs,name = 'bandpass filtered ts of ' + self.name.replace('(','').replace(')',''))    
        return tsarr(np.fft.irfft(Xfilt*self.N*self.df,n=self.N),fs=self.fs,**kwargs)
    
    def der_deriviative(self,opt='dt',**kwargs):
        ''' derive deriviative timeseries 
        INPUTS:
            :param opt:    string
                deriviative option
        '''
        if opt == 'dt':
            H = 1j*2*np.pi*self.fper
        elif opt == 'dtdt':
            H = (1j*2*np.pi*self.fper)**2
        elif opt == 'intint':
            H = 1/(1j*2*np.pi*self.fper)
        elif opt ==  'int':
            H = 1/(1j*2*np.pi*self.fper)**2
        y = np.fft.irfft(self.Xos*H*self.N*self.df,n=self.N)
        return tsarr(y,fs=self.fs,**kwargs)
            
    
    def der_response(self,f,H,**kwargs):  
        ''' derive timeseries of structural response for given transfer function
        INPUTS:
            :param f:    array_like
                frequency vector
            :param H:    array_like
                frequency response function
        OUTPUTS:  
            outobj: timeseries object
                output timeseries (response)    
        SOURCES:
            J. Slavič, M. Boltežar, M. Mršnik, M. Cesnik, J. Javh, Vibration fatigue by spectral methods: From structural dynamics to fatigue, Elsevier, 2020          
        '''
        H = get_tf(f,H,self.fper)            
        y = np.fft.irfft(self.Xos*H*self.N*self.df,n=self.N)
        if 'var' not in kwargs.keys():
            kwargs = dict(kwargs,var = self.var.replace('x','y'))
        if 'unit' not in kwargs.keys():
            kwargs = dict(kwargs,unit = self.unit)        
        if 'name' not in kwargs.keys():
            kwargs = dict(kwargs,name = 'response for ' + self.name.replace('(','').replace(')',''))                     
        # outobj = timeseries(y,self.t,**kwargs)
        outobj = tsarr(y,fs=self.fs,**kwargs)
        return  outobj
    
    def der_scale2max(self,value = 1,**kwargs):
        ''' derive timeseries scaled to reference value
        INPUTS:
            :param value:    scalar
                value to scale max value to
        OUTPUTS:  
            outobj: timeseries object
                scaled timeseries
        ''' 
        if 'var' not in kwargs.keys():
            kwargs = dict(kwargs,var = self.var + '(scaled)')
        outobj = tsarr(self.x/self.stat['max']*value,fs=self.fs,**kwargs)
        return  outobj        
        
    def der_zeropadded(self,zpfactor):
        ''' derive zeropadded timeseries
        INPUTS:
            :param tpfactor:    scalar
                zero-padding factor
        OUTPUTS:  
            outobj: timeseries object
                zeropadded timeseries
        SOURCES:
            A. Trapp, Q. Hoesch, P. Wolfsteiner, Effects of insufficient sampling on counting algorithms in vibration fatigue, Procedia Structural Integrity 38 (2022) 260–270            
        ''' 
        newN = int(np.ceil(self.N*zpfactor))
        Y = np.append(self.Xos, np.zeros(newN-self.N))
        y = np.fft.irfft(Y*newN*self.df,n=newN)
        t = np.linspace(0,self.T,newN+1)
        t = t[:-1]
        outobj = tsarr(y,fs=self.fs)
        return  outobj 

    def get_ls(self,Nf = 10, func='acc2dis', D = 0.03,p_interp = 30, plot = True):
        # eval input resonant frequencies
        if np.isscalar(Nf):
            evalf  = np.linspace(0,round(self.fNy,0),Nf+2)
            evalf  = evalf[1:-1]
        else:
            evalf  = Nf  
            Nf     = len(evalf)       
        check_df  = evalf[0]/100 
        if self.dfpsd > check_df:
            self.est_psd(df = check_df)
        # initialize
        ls, lsinterp = [],[]

        for ii in np.arange(len(evalf)):
            resp = self.der_sdofResponse(fD = evalf[ii],func = func, D = D)
            ls.append({'si_x': resp.ls['si_x'], 'ni_x': resp.ls['ni_x'], 'max_ni': resp.ls['npeaks'],
                       'si_X': resp.ls['si_X'], 'ni_X': resp.ls['ni_X'], 'fD': evalf[ii]})

        ni_interp = np.logspace(0,np.log10(0.9*min([x['max_ni'] for x in ls])),p_interp)
        ni_interp = ni_interp[ np.append(1,np.diff(ni_interp)) >= 1]

        for rsp in ls:
            si_interp = np.interp(ni_interp,rsp['ni_x'],rsp['si_x'])
            si_Interp = np.interp(ni_interp,rsp['ni_X'],rsp['si_X'])
            lsinterp.append({'si_x': si_interp, 'ni_x': ni_interp,
                       'si_X': si_Interp, 'ni_X': ni_interp})

        if plot:  
            fig = plt.figure()
            ax = Axes3D(fig) 
        #  ax = fig.add_subplot(projection='3d')
            FD,NI = np.meshgrid(evalf,ni_interp) 
            ax.plot_surface(FD, NI, np.array([x['si_x'] for x in lsinterp]).T,linewidth=0, antialiased=False,label='RFC')
            ax.plot_surface(FD, NI, np.array([x['si_X'] for x in lsinterp]).T,linewidth=0, antialiased=False,label='DK')
            ax.set_xlabel('resonant frequency $f_D$ [Hz]')
            ax.set_ylabel('cumulated cycles [-]')
            ax.set_zlabel('amplitude [$' + resp.var + '$]')
            ax.set_title('RFC vs. DK ($' + self.var + '$)')
            #ax.legend()
            plt.show()    


            fig = plt.figure()
            ax = Axes3D(fig) 
        #  ax = fig.add_subplot(projection='3d')
            ax.plot_surface(FD, np.log10(NI), np.array([x['si_x'] for x in lsinterp]).T,linewidth=0, antialiased=False,label='RFC')
            ax.plot_surface(FD, np.log10(NI), np.array([x['si_X'] for x in lsinterp]).T,linewidth=0, antialiased=False,label='DK')
            ax.set_xlabel('resonant frequency $f_D$ [Hz]')
            ax.set_ylabel('cumulated cycles (log) [-]')
            ax.set_zlabel('amplitude [$' + resp.var + '$]')
            ax.set_title('RFC vs. DK ($' + self.var + '$; log)')
            #ax.legend()
            plt.show()    
        return ls,lsinterp
  
    #   self.p    = [sig.N/N for sig in inp]
    
class timeseries: 
    ''' outer class definition
    outer object for arranging time series components (arr) and channel definition (chan)
    '''
    '''
        Attributes
        ----------
        arr:        array of tsarr-objects for managing multi-channel multi-process data
        chan:       array of tsarr-objects for managing multi-channel data
        ----------
        nc:         numper of channels
        np:         number of processes 
        scsp:       Is single-channel, single-process?
        psdmx:      PSD Matrix
        fpsdmx:     frequency vector for PSD Matrix
        dfpsdmx:    resolution of PSD Matrix
        qsls:       quasi-stationary load spectra
        
        ...various attributes from tsarr        
        ----------
        Methods
        ---------- 
        genChan():      internal: generate channel attribute to simplify multi-process configurations
        whos():         generate console output for configuration output
        callPlots():    internal: callPlots
        callProps():    internal: callProps
        callMethods():  internal: callMethods
        est_qsls():     estimate quasi-stationary load spectrum
        est_stats():    init call: estimate key statistical values
        est_psdmx():    estimate PSDs and CPSDs (timeseries class)
        plot_psdmx():   plot PSD Matrix  
        ...various methods from tsarr 
    '''
            
    def __init__(self, srs,var = 'x', **kwargs):
        # prepare timeseries array (microstructure)
        self.arr = []
        if isinstance(srs,(np.ndarray,dict,tsarr)):
            srs = [[srs]]
        elif isinstance(srs[0],(np.ndarray,dict,tsarr)):
            srs = [srs]
            
        # generate timeseries array (microstructure)
        for ii,chan in enumerate(srs):
            self.arr.append([])
            for proc in chan:
                if type(proc) == dict:
                    if 'psd' in proc.keys(): # Dictionary with PSD Input -> random time series
                        kwargs['x'],_ = get_GaussianSeries(proc)
                        if len(chan) > 1:
                            window = sp.signal.windows.tukey(len(kwargs['x']),alpha=5e-4)
                            window[window == 0] = 1e-10
                            kwargs['x'] = kwargs['x']*window # smoothing of transitions
                    # else: Dictionary with time series input
                    kwargs.update(proc)
                    self.arr[ii].append(tsarr(**kwargs))
                elif type(proc) == tsarr:    # tsarr input
                    self.arr[ii].append(proc)
                else:                        # list with time series input
                    self.arr[ii].append(tsarr(proc, **kwargs))
        self.var = var
        
        # provide a channel array (easy access, option: replacing it with a property function)
        if self.np > 1:
            self.genChan()
        
        # basic statistical properties
        self.est_stats()
        
        # determine if single channel single process     
        if self.nc == 1 and self.np == 1: 
            self.scsp = True
            self.sameTprops = True
        else:
            self.scsp = False
            if np.isscalar(self.T) or np.isscalar(self.N):
                self.sameTprops = True
            else:
                self.sameTprops = False
            # auto-naming of subelements if nothing was passed
            if np.all([item == 'x' for sublist in self.vars for item in sublist]):
                    for cc,chan in enumerate(self.arr):
                        for pp,proc in enumerate(chan):
                            if self.np == 1:
                                proc.var = self.var + '_{c=' + str(cc+1) + '}'
                            elif self.nc == 1:
                                proc.var = self.var + '_{p=' + str(pp+1) + '}'
                            else:
                                proc.var = self.var + '_{c=' + str(cc+1) + ',p=' + str(pp+1) + '}'
    
    def genChan(self):   
        '''generate channel attribute to simplify multi-process configurations'''
        self.__chans = []                
        if self.nc == 1:
            self.__chans.append(tsarr(np.concatenate(self.xs[0]),fs = np.mean(self.fss),var = self.var,unit = self.arr[0][0].unit)) 
        else:       
            for cc,chan in enumerate(self.arr):
                self.__chans.append(tsarr(np.concatenate(self.xs[cc]),fs = np.mean(self.fss[cc]),var = self.var + '_{c=' + str(cc+1) + '}',unit = self.arr[0][0].unit))                
                
    
    # timeseries properties                     
    @property
    def nc(self): # number of channels
        return len(self.arr)
    @property
    def np(self): # number of processes
        return len(self.arr[0])  
    
    @property
    def x(self): # data
        if self.nc == 1:
            return np.concatenate(self.xs[0])
        else:
            return [np.concatenate(chan) for chan in self.xs]
    @property
    def T(self): # duration
        T = np.sum(self.Ts,axis = 1)
        if self.nc == 1 or np.all(T == T[0]): # same for all channels
            return T[0]
        else:                                # different for channels
            return T
    @property
    def N(self): # number of samples
        N = np.sum(self.Ns,axis = 1)
        if self.nc == 1 or np.all(N == N[0]): # same for all channels
            return N[0]
        else:                                # different for channels
            return N
    @property
    def t(self): # time vector
        T = np.sum(self.Ts,axis = 1)
        if self.sameTprops: # same for all channels
            return self.chan[0].t
        else:                                # different for channels
            return [np.concatenate(chan) for chan in self.ts]        
    @property
    def fs(self): # sampling rate
        if self.sameTprops:
            return self.chan[0].fs
        else:
            return np.mean(self.fss,axis = 1)
    @property
    def dt(self): # time step
        if self.sameTprops:
            return self.chan[0].dt
        else:
            return np.mean(self.dts,axis = 1)
    @property
    def prob(self): # estimate of probability density function (PDF)
        if self.nc == 1:
            return self.chan[0].prob
        else:
            return [np.concatenate(chan) for chan in self.probs]      
    @property
    def grid(self): # grid for estimate of PDF
        if self.nc == 1:
            return self.chan[0].grid
        else:
            return [np.concatenate(chan) for chan in self.grids]      
    @property
    def pgauss(self): # Gaussian PDF of same variance
        if self.nc == 1:
            return self.chan[0].pgauss
        else:
            return [np.concatenate(chan) for chan in self.pgausss]   
    @property
    def psd(self): # power spectral density (PSD)
        if self.nc == 1:
            return self.chan[0].psd
        else:
            return [np.concatenate(chan) for chan in self.psds]   
    @property
    def fpsd(self): # frequency vector for PSD
        if self.nc == 1:
            return self.chan[0].fpsd
        else:
            if self.dfpsd.shape == ():
                return self.chan[0].fpsd
            else:  
                return [chan.fpsd for chan in self.chan]   
    @property
    def dfpsd(self): # increment of frequency vector for PSD
        if self.nc == 1:
            return self.chan[0].dfpsd
        else:
            dfpsds = [chan.dfpsd for chan in self.chan]
            if np.all(dfpsds == self.chan[0].dfpsd):
                return self.chan[0].dfpsd    
            else:
                return dfpsds   
    @property
    def fper(self): # one-sided frequency vector
        if self.sameTprops:
            return self.chan[0].fper
        else:
            return [np.concatenate(chan) for chan in self.fper]   
    @property
    def ffper(self): # two-sided frequency vector
        if self.sameTprops:
            return self.chan[0].ffper
        else:
            return [np.concatenate(chan) for chan in self.ffper]               
    @property
    def Xos(self): # one-sided Fourier coefficients
        if self.nc == 1:
            return self.chan[0].Xos
        else:
            return [np.concatenate(chan) for chan in self.Xoss]   
    @property
    def Xts(self): # two-sided Fourier coefficients
        if self.nc == 1:
            return self.chan[0].Xts
        else:
            return [np.concatenate(chan) for chan in self.Xtss]  
    # properties that return tsarr perspective (multi-process)            
    @property
    def xs(self):  # multi-process return of x
        return self.callProps('x')    
    @property
    def ts(self): # multi-process return of t
        return self.callProps('t')   
    @property
    def vars(self): # multi-process return of var
        return self.callProps('var')  
    @property
    def Ts(self): # multi-process return of T
        return self.callProps('T')    
    @property
    def Ns(self): # multi-process return of N
        return self.callProps('N')     
    @property
    def fss(self): # multi-process return of fs
        return self.callProps('fs')   
    @property
    def dts(self): # multi-process return of fs
        return self.callProps('dt')  
    @property
    def units(self): # multi-process return of units
        return self.callProps('unit')  
    @property
    def probs(self): # multi-process return of prob
        return self.callProps('prob')  
    @property
    def grids(self): # multi-process return of grid
        return self.callProps('grid')  
    @property
    def pgausss(self): # multi-process return of Gaussian PDF
        return self.callProps('pgauss') 
    @property
    def fpsds(self): # multi-process return of frequency vector of psd
        return self.callProps('fpsd')  
    @property
    def psds(self): # multi-process return of psd
        return self.callProps('psd')  
    @property
    def dfpsds(self): # multi-process return of resolution of psd
        return self.callProps('dfpsd')  
    @property
    def Xoss(self): # multi-process return of one-sided Fourier coefficients
        return self.callProps('Xos')  
    @property
    def Xtss(self): # multi-process return of two-sided Fourier coefficients
        return self.callProps('Xts')   
    @property
    def fpers(self): # multi-process return of one-sided frequency vector
        return self.callProps('fper')  
    @property
    def ffpers(self): # multi-process return of two-sided frequency vector
        return self.callProps('ffper')      
    @property
    def chan(self): # multi-channel array (np = 1)
        if self.np == 1:
            return [chan[0] for chan in self.arr]
        else:
            return self.__chans
          
    def whos(self):
        ''' generate console output for configuration output  '''
        if self.nc == 1:
            str1 = 'single-channel'
        else:
            str1 = 'multi-channel (nc = {})'.format(self.nc)
        if self.np == 1:
            str2 = 'single-process'
        else:
            str2 = 'multi-process (np = {})'.format(self.np)
        print(str1,str2)
         
    def callPlots(self,passmethod,*args,access='chan',**kwargs):
        ''' manage plot functions for timeseries class
        >>> for internal use >>>
        '''
        if access == 'chan':
            fig, axs = plt.subplots(self.nc, 1)
        elif access == 'proc':
            fig, axs = plt.subplots(self.nc, self.np)
        elif access == 'comp':
            fig, axs = plt.subplots(1, 1)
            cls = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        else:
            warnings.warn('internal calling error')
        axs = np.atleast_2d(axs)
        
        if access == 'chan':
            for cc,chan in enumerate(self.chan):
                    callmethd = getattr(chan,passmethod)
                    callmethd(*args,ax=axs[0,cc],**kwargs) 
        elif access == 'proc':
            for cc,chan in enumerate(self.arr):
                for pp,proc in enumerate(chan):
                    callmethd = getattr(proc,passmethod)
                    callmethd(*args,ax=axs[cc,pp],**kwargs) 
        elif access == 'comp':
            for cc,chan in enumerate(self.chan):
                    callmethd = getattr(chan,passmethod)
                    callmethd(*args,ax=axs[0,0],**kwargs) 
            for nn,line in enumerate(axs[0,0].lines):
                line.set_color(cls[nn % 10])       
            axs[0,0].legend()
        else:
            warnings.warn('internal calling error')        
        fig.show()
        # plt.show() alternative: e.g. spyder..
        return axs
             
    def callProps(self,passprop):
        ''' manage calling of properties for timeseries class
        >>> for internal use >>>
        '''
        retvalues = np.empty((self.nc,self.np),dtype='object')
        for cc,chan in enumerate(self.arr):
            for pp,proc in enumerate(chan):
                retvalues[cc,pp] = getattr(proc,passprop)
        return retvalues   
             
    def callMethods(self,passmethod,*args,access='chan',squeeze = False,**kwargs):
        ''' manage calling of methods for timeseries class
        >>> for internal use >>>
        '''
        retvalues = []
        if access == 'chan':
            for cc,chan in enumerate(self.chan):
                retvalues.append([])
                callmethd = getattr(chan,passmethod)
                retvalues[cc].append(callmethd(*args,**kwargs))
        elif access == 'proc':
            for cc,chan in enumerate(self.arr):
                retvalues.append([])
                for proc in chan:
                    callmethd = getattr(proc,passmethod)
                    retvalues[cc].append(callmethd(*args,**kwargs)) 
        else:
            warnings.warn('internal calling error') 
        if squeeze:
            return np.squeeze(retvalues).tolist()
        else:
            return retvalues  
    
    # def callMethods(self,passmethod,*args,access='chan',**kwargs):
    #     ''' manage calling of methods for timeseries class
    #     >>> for internal use >>>
    #     '''
        
    #     retvalues = np.empty((self.nc,self.np),dtype='object')
    #     for cc,chan in enumerate(self.arr):
    #         for pp,proc in enumerate(chan):
    #             callmethd = getattr(proc,passmethod)
    #             retvalues[cc,pp] = callmethd(*args,**kwargs)
    #     if self.scsp:
    #         retvalues.squeeze()          
    #     return retvalues.tolist()  
                
    def est_qsls(self,si_X = None):
        ''' estimate quasi-stationary load spectrum
        INPUTS:
        :param  si_X: vector
            evaluate spectral load spectra estimators for given amplitude vector     
        SOURCES:
            A. Trapp, P. Wolfsteiner, Fatigue assessment of non-stationary random loading in the frequency domain by a quasi-stationary Gaussian approximation, International Journal of Fatigue 148 (2021) 106214     
        ''' 
        if self.scsp:
            self.qsls = self.arr[0,0].ls
            return
        self.qsls = []
        for chan in self.arr:
            maxvals = [proc.stat['maxvalue'] for proc in chan]
            overallmax = np.max(maxvals)
            ls = ({'si_X': np.linspace(0,overallmax*1.1,1000)})
            ls['ni_X'] = np.zeros(ls['si_X'].shape)
            for proc in chan:
                proc.est_ls(si_X = ls['si_X'])
                ls['ni_X'] = ls['ni_X'] + proc.ls['ni_X'] 
            self.qsls.append(ls) 
            
    def est_stats(self): 
        ''' estimate key statistical values (timeseries class)
        -> called by __init__
        '''        
        self.stat = {'mean': [np.mean(chan.x) for chan in self.chan]}
        self.stat['std'] = [np.std(chan.x) for chan in self.chan]
        self.stat['var'] = [np.var(chan.x) for chan in self.chan]
        self.stat['skewness'] = [sp.stats.skew(chan.x) for chan in self.chan]
        self.stat['kurtosis'] = [sp.stats.kurtosis(chan.x)+3 for chan in self.chan]
        xMV  = [chan.x for chan in self.chan]
        self.stat['cov'] = np.cov(xMV)
        self.stat['corrcoef'] = np.corrcoef(xMV)
        self.stat['largestvalues']  = [np.max(chan.x) for chan in self.chan]
        self.stat['smallestvalues'] = [np.min(chan.x) for chan in self.chan]
        self.stat['maxvalues']      = np.max(np.abs([self.stat['smallestvalues'],self.stat['largestvalues'] ]),axis = 0)
        self.stat['max']            = self.stat['maxvalues'] 
        self.stat['largestvalue']   = np.max(self.stat['largestvalues'])
        self.stat['smallestvalue']  = np.min(self.stat['smallestvalues'])
        self.stat['maxvalue']       = np.max(self.stat['maxvalues'])
        self.stat['max']            = self.stat['maxvalue']  
    
    def est_psdmx(self,df = 1):  
        ''' estimate PSDs and CPSDs (timeseries class)
        INPUTS:
        :param df:    scalar
            frequency resolution of PSDmx
        '''   
        self.psdmx = np.empty((self.nc, self.nc), dtype=object)
        for ii,chan1 in enumerate(self.chan):
            for jj,chan2 in enumerate(self.chan):
                self.fpsdmx,self.psdmx[ii,jj] = sp.signal.csd(chan1.x,chan2.x,chan1.fs,nperseg=round(chan1.fs/df))
        self.dfpsdmx = np.mean(np.diff(self.fpsdmx)) 
    
    def plot_psdmx(self):
        ''' plot PSD Matrix'''
        if not hasattr(self, 'fpsdmx'):
            self.est_psdmx()
        fig,ax = plt.subplots(self.nc,self.nc)
        ax = np.atleast_2d(ax)
        
        for ii in np.arange(self.nc):
            for jj in np.arange(self.nc):
                lbl = "$G_{" + self.chan[ii].var + self.chan[jj].var + "}" + ': \sigma^2$ = {:.3f}'.format(self.stat['cov'][ii,jj]) + ' [(' + self.chan[ii].unit + ')$^2$]'
                if ii>jj: # Real/Imag
                    ax[ii,jj].plot(self.fpsdmx,np.real(self.psdmx[ii,jj]),'k',label = 'Real, Cov = {:.3f}'.format(self.stat['cov'][ii,jj]) + ' [(' + self.chan[ii].unit + ')$^2$]') 
                    ax2 = ax[ii,jj].twinx()
                    ax2.plot(self.fpsdmx,np.imag(self.psdmx[ii,jj]),'r',label = 'Imag')
                    ax2.legend()    
                elif ii<jj:
                    ax[ii,jj].plot(self.fpsdmx,np.abs(self.psdmx[ii,jj]),'k',label = 'Abs') 
                    ax2 = ax[ii,jj].twinx()
                    ax2.plot(self.fpsdmx,np.real(self.psdmx[ii,jj])/(np.sqrt(self.psdmx[ii,ii])*np.sqrt(self.psdmx[jj,jj])),'r',label = 'Coh')    
                    ax2.set_ylim((-1,1)) 
                    ax2.set_ylabel('Coherence [-]')  
                    ax2.legend()                  
                elif ii==jj:
                    ax[ii,jj].plot(self.fpsdmx,self.psdmx[ii,jj],'k',label = lbl)         
                ax[ii,jj].set_xlabel("f [Hz]")
                ax[ii,jj].set_ylabel("$G_{" + self.chan[ii].var + self.chan[jj].var + "}$ [(" + self.chan[ii].unit  + ")$^2$/Hz]")
                ax[ii,jj].legend()   
                ax[ii,jj].grid(True) 
                       
    def plot(self,*args,access = 'chan',**kwargs):
        return self.callPlots('plot',*args,access = access, **kwargs)                           
    def plot_psd(self,*args,access = 'chan',**kwargs):
        return self.callPlots('plot_psd',*args,access = access, **kwargs)                  
    def plot_X(self,*args,access = 'chan',**kwargs):
        return self.callPlots('plot_X',*args,access = access, **kwargs)                    
    def plot_prob(self,*args,access = 'chan',**kwargs):
        return self.callPlots('plot_prob',*args,access = access, **kwargs)          
    def plot_ls(self,*args,access = 'chan',**kwargs):
        return self.callPlots('plot_ls',*args,access = access, **kwargs)   
    def plot_pseudoDam(self,*args,access = 'chan',**kwargs):
        return self.callPlots('plot_pseudoDam',*args,access = access, **kwargs)    
    def plot_tf(self,*args,access = 'chan',**kwargs):
        return self.callPlots('plot_tf',*args,access = access, **kwargs)       
    def plot_fds(self,*args,access = 'chan',**kwargs):
        fdscalcs = []
        if access == 'chan' or access == 'comp':
            for chan in self.chan:
                fdscalcs.extend(chan.FDScalcs)    
        else:
            for chan in self.chan:
                for proc in chan:
                    fdscalcs.extend(proc.FDScalcs)    
        plot_fds(fdscalcs)  
    def plot_nonstat(self,*args,access = 'chan',**kwargs): 
        for chan in self.chan:
            chan.plot_nonstat(**kwargs)                              
    def est_psd(self,*args,access = 'chan',**kwargs):
        return self.callMethods('est_psd',*args,access = access, **kwargs)             
    def est_ls(self,*args,access = 'chan',**kwargs):
        return self.callMethods('est_ls',*args,access = access, **kwargs)            
    def est_dirlik(self,*args,access = 'chan',**kwargs):
        return self.callMethods('est_dirlik',*args,access = access, **kwargs)              
    def est_fds(self,*args,access = 'chan',**kwargs):
        return self.callMethods('est_fds',*args,access = access,squeeze = True, **kwargs)              
    def est_nonstat(self,*args,access = 'chan',**kwargs):
        return self.callMethods('est_nonstat',*args,access = access,squeeze = True, **kwargs)           
    def der_statResponse(self,*args,access = 'chan',**kwargs):
        return self.callMethods('der_statResponse',*args,access = access,squeeze = True, **kwargs)        
    def der_sdofResponse(self,*args,access = 'chan',**kwargs):
        return timeseries(self.callMethods('der_sdofResponse',*args,access = access, **kwargs))           
    def der_sdofTransfer(self,*args,access = 'chan',**kwargs):
        return timeseries(self.callMethods('der_sdofTransfer',*args,access = access, **kwargs))        
    def der_response(self,*args,access = 'chan',**kwargs):
        return timeseries(self.callMethods('der_response',*args,access = access, **kwargs))         
    def der_highpass(self,*args,access = 'chan',**kwargs):
        return timeseries(self.callMethods('der_highpass',*args,access = access, **kwargs)) 
    def der_lowpass(self,*args,access = 'chan',**kwargs):
        return timeseries(self.callMethods('der_lowpass',*args,access = access, **kwargs))     
    def der_bandpass(self,*args,access = 'chan',**kwargs):
        return timeseries(self.callMethods('der_bandpass',*args,access = access, **kwargs))
    def der_zeropadded(self,*args,access = 'chan',**kwargs):
        return timeseries(self.callMethods('der_zeropadded',*args,access = access, **kwargs))
    def der_deriviative(self,*args,access = 'chan',**kwargs):
        return timeseries(self.callMethods('der_deriviative',*args,access = access, **kwargs))
    def der_scale2max(self,*args,access = 'chan',**kwargs):
        return timeseries(self.callMethods('der_scale2max',*args,access = access, **kwargs))
    def der_lsEquivalent(self,*args,access = 'chan',**kwargs):
        return timeseries(self.callMethods('der_lsEquivalent',*args,access = access, **kwargs)) 
    def der_iFDS(self,*args,access = 'chan',**kwargs):
        return timeseries(self.callMethods('der_iFDS',*args,access = access, **kwargs))                                              
    def der_qsSTFT(self,*args,access = 'chan',**kwargs):
        return timeseries(self.callMethods('der_qsSTFT',*args,access = access, **kwargs))
               
if __name__ == "__main__":  
    xc1p1 = np.random.randn(20000)
    xc1p2 = np.random.randn(10000)
    xc1p3 = np.random.randn(5000)
    xc2p1 = np.random.randn(20000)
    xc2p2 = np.random.randn(10000)
    xc2p3 = np.random.randn(5000)

    c1 = [xc1p1,xc1p2,xc1p3]
    c2 = [xc2p1,xc2p2,xc2p3]
    x  = [c1,c2]

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


    scspdict = timeseries(inp[0])
    scspdict.whos()
    scspdict.plot()
    scmpdict = timeseries(inp)
    scmpdict.whos()
    scmpdict.plot()
    mcmpdict = timeseries([inp,inp,inp])
    mcmpdict.whos()
    mcmpdict.plot()

    scmp = timeseries(c1,fs = 1200)
    scmp.whos()
    scmp.plot()
    mcmp = timeseries(x,fs = 1200)
    mcmp.whos()
    mcmp.plot()
    scsp = timeseries(xc1p1,fs = 1000)
    scsp.whos()
    scsp.plot()





