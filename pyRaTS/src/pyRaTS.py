# -*- coding: utf-8 -*-
"""
PYthon module for the processing of RAndom vibration TimeSeries - pyRaTS
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
- get_weightFunc_WOI:       get weighting function (optimization)
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
    nonstat['kurtosis'] = np.sum(nonstat['Mxx'])/np.sum(nonstat['Rxx_stat'])-3
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
        # wfunc_it[rr] = np.interp(inp['fpsd'],np.linspace(0,inp['fpsd'][-1],inp['res']*inp['Nf']),wfunc[rr])
        # interpfunc   = sp.interpolate.interp1d(fAxis,wfunc[rr],kind = 'cubic')
        interpfunc   = sp.interpolate.interp1d(fAxis,np.append(wfunc[rr],[1,1]))
        wfunc_it[rr] = interpfunc(inp['fpsd'])
        # wfunc_it[rr] = glsInterp(np.linspace(0,inp['fpsd'][-1],inp['res']*inp['Nf']),wfunc[rr],inp['fpsd'])
    return wfunc_it

def get_weightFunc_WOI(x,inp):
    ''' get weighting function (optimization)
    >>> for internal use <<<  
    SOURCES:
        Wolfsteiner, P., Trapp, A., and Fertl, L., 'Random vibration fatigue with non-stationary loads using an extended fatigue damage spectrum with load spectra', ISMA 2022 Conference
    '''
    p = x[:inp['R']]
    wfunc = x[inp['R']:].reshape((inp['R'], inp['res']*inp['Nf']))
    wfunc_it = np.zeros((inp['R'],len(inp['fpsd'])))
    fAxis = np.linspace(0,inp['fpsd'][-1],inp['res']*inp['Nf'])
    for rr in range(inp['R']):
        interpfunc   = sp.interpolate.interp1d(fAxis,wfunc[rr])
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

    out['DK_lk']   = D1[:,None]*np.exp(-Z/Q[:,None]) + D2[:,None]*np.exp(-Z**2/(2*R[:,None]**2)) + D3[:,None]*np.exp(-Z**2/2)
    out['DK_lk_ps']   = out['DK_lk']*out['ny_p'][:,None]
    return np.matmul(weights,out['DK_lk_ps'])

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

def plot_fds(inp):  
    '''
    plot FDS of (multiple) signals
    INPUTS:
        :param inp:    dictionary-(array)
            FDS calculations
    '''  
    try:
        for kk in range(len(inp[0]['m'])): 
            plt.figure()
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
        plt.show()
    except:
        print('Input:',inp)
        warnings.warn('function "plot_fds()" cannot be carried out')

def plot_nonstat(inp,func = 'Mxx'):
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
    for func in func:
        if func == 'Rxx_asym':
            quant = [inp['Rxx'],inp['Rxx_stat']]
            def_title = 'non-stationarity matrix'
            def_clabl1 = '$R_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: {p:.2f}%) [('+inp['unit']+')$^4$]'.format(p = inp['m4rep'])
            def_clabl2 = '$R_{'+2*inp['var']+',stat}$ [('+inp['unit']+')$^4$]'
        elif func == 'Rxx':
            quant = [inp['Rxx_sym'],inp['Rxx_stat']]
            def_title = 'non-stationarity matrix (symmetric)'
            def_clabl1 = '$R_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: {p:.2f}%) [('+inp['unit']+')$^4$]'.format(p = inp['m4rep'])
            def_clabl2 = '$R_{'+2*inp['var']+',stat}$ [('+inp['unit']+')$^4$]'
        elif func == 'Mxx_asym':
            quant = [inp['Mxx'],inp['Mxx_stat']]
            def_title = 'non-stationarity moment matrix'
            def_labl1 = '$\mu_{{4,ns}}$ = {val:.2f}'.format(val = np.sum(quant[0]))
            def_labl2 = '$\mu_4$ = {val:.2f}'.format(val = inp['moms'][4]) 
            def_labl3 = '$\mu_{{4,stat}}$ = {val:.2f}'.format(val = 3*inp['moms'][2]**2) 
            def_clabl1 = '$M_{{'+2*inp['var']+'}}$ [('+inp['unit']+')$^4$] (' + def_labl1 +'/'+ def_labl2 + '/$\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
            def_clabl2 = '$M_{'+2*inp['var']+'}$ [('+inp['unit']+')$^4$] (' + def_labl3 +'/'+ def_labl2 + ')'
        elif func == 'Mxx':
            quant = [inp['Mxx_sym'],inp['Mxx_stat']]
            def_title = 'non-stationarity moment matrix (symmetric)'
            def_labl1 = '$\mu_{{4,ns}}$ = {val:.2f}'.format(val = np.sum(quant[0]))
            def_labl2 = '$\mu_4$ = {val:.2f}'.format(val = inp['moms'][4]) 
            def_labl3 = '$\mu_{{4,stat}}$ = {val:.2f}'.format(val = 3*inp['moms'][2]**2) 
            def_clabl1 = '$M_{{'+2*inp['var']+'}}$ [('+inp['unit']+')$^4$] (' + def_labl1 +'/'+ def_labl2 + '/$\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
            def_clabl2 = '$M_{'+2*inp['var']+'}$ [('+inp['unit']+')$^4$] (' + def_labl3 +'/'+ def_labl2 + ')'
        elif func == 'Cxx_asym':
            quant = [inp['Cxx'],inp['Cxx_stat']]
            def_title = 'non-stationarity cumulant matrix'
            def_labl1 = '$c_{{4,ns}}$ = {val:.2f}'.format(val = np.sum(quant[0]))
            def_labl2 = '$c_4$ = {val:.2f}'.format(val = inp['moms'][4]-3*inp['moms'][2]**2) 
            def_labl3 = '$c_{{4,stat}}$ = {val:.2f}'.format(val = 0) 
            def_clabl1 = '$C_{{'+2*inp['var']+'}}$ [('+inp['unit']+')$^4$] (' + def_labl1 +'/'+ def_labl2 + '/$\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
            def_clabl2 = '$C_{'+2*inp['var']+'}$ [('+inp['unit']+')$^4$] (' + def_labl3 +'/'+ def_labl2 + ')'
        elif func == 'Cxx':
            quant = [inp['Cxx_sym'],inp['Cxx_stat']]
            def_title = 'non-stationarity cumulant matrix (symmetric)'
            def_labl1 = '$c_{{4,ns}}$ = {val:.2f}'.format(val = np.sum(quant[0]))
            def_labl2 = '$c_4$ = {val:.2f}'.format(val = inp['moms'][4]-3*inp['moms'][2]**2) 
            def_labl3 = '$c_{{4,stat}}$ = {val:.2f}'.format(val = 0) 
            def_clabl1 = '$C_{{'+2*inp['var']+'}}$ [('+inp['unit']+')$^4$] (' + def_labl1 +'/'+ def_labl2 + '/$\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
            def_clabl2 = '$C_{'+2*inp['var']+'}$ [('+inp['unit']+')$^4$] (' + def_labl3 +'/'+ def_labl2 + ')'
        elif func == 'Axx_asym':
            quant = [inp['Axx'],inp['Axx_stat']]
            def_title  = '    modulation matrix'
            def_clabl1 = '$A_{'+2*inp['var']+'}}$ [-] ($\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
            def_clabl2 = '$A_{'+2*inp['var']+',stat}$ [-]'
        elif func == 'Axx':
            quant = [inp['Axx_sym'],inp['Axx_stat']]
            def_title = '    modulation matrix (symmetric)'
            def_clabl1 = '$A_{'+2*inp['var']+'}}$ [-] ($\mu_4$-coverage: {p:.2f}%)'.format(p = inp['m4rep'])
            def_clabl2 = '$A_{'+2*inp['var']+',stat}$ [-]'   
        elif func == 'Kxx_asym':
            quant = [inp['Kxx'],inp['Kxx_stat']]
            def_title  = 'non-stationarity covariance matrix'
            def_clabl1 = '$K_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: ' + str("%.2f"% inp['m4rep']) + '%) [('+inp['unit']+')$^4$]'
            def_clabl2 = '$K_{'+2*inp['var']+'}$ [('+inp['unit']+')$^4$]'
        elif func == 'Kxx':
            quant = [inp['Kxx_sym'],inp['Kxx_stat']]
            def_title = 'non-stationarity covariance matrix (symmetric)'
            def_clabl1 = '$K_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: ' + str("%.2f"% inp['m4rep']) + '%) [('+inp['unit']+')$^4$]'
            def_clabl2 = '$K_{'+2*inp['var']+'}$ [('+inp['unit']+')$^4$]'
        elif func == 'rhoxx_asym':
            quant = [inp['rhoxx'],inp['Axx_stat']]
            def_title = 'non-stationarity synchronicity matrix'
            def_clabl1 = '$\\rho_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: {p:.2f}%) [-]'.format(p = inp['m4rep'])
            def_clabl2 = '$A_{'+2*inp['var']+',stat}$ [-]'   
        elif func == 'rhoxx':
            quant = [inp['rhoxx_sym'],inp['Axx_stat']]
            def_title = 'non-stationarity synchronicity matrix (symmetric)'
            def_clabl1 = '$\\rho_{{'+2*inp['var']+'}}$ ($\mu_4$-coverage: {p:.2f}%) [-]'.format(p = inp['m4rep'])
            def_clabl2 = '$A_{'+2*inp['var']+',stat}$ [-]'   
        cnt = 0    
        for qq in quant:
            cnt = cnt +1 
            fig = plt.figure()
            pos = plt.imshow(qq, extent=[0, inp['f_end'], inp['f_end'], 0])   
            plt.xlabel("$f_1$ [Hz]")
            plt.ylabel("$f_2$ [Hz]")
            plt.grid(True)
            if cnt == 1:
                plt.title(def_title)
                fig.colorbar(pos,label = def_clabl1)
            else: 
                plt.title(def_title[4:])
                fig.colorbar(pos,label = def_clabl2)
            plt.show()      

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

    
###############################################################################
class series: 
    '''
        Attributes
        ----------
        x:         (load/response)-series
        N:         number of samples
        var:       variable name
        unit:      unit to be displayed in plots
        prob:      estimate of probability density function (PDF)
        grid:      grid for estimate of PDF
        pgauss:    Gaussian PDF of same variance
        stat:      Dictionary of statistical indicators
        psd:       power spectral density (PSD)
        fpsd:      normalized frequency vector for PSD
        dfpsd:     increment of frequency vector for PSD
        Xos:       one-sided Fourier coefficients
        Xts:       two-sided Fourier coefficients
        ----------
        Methods
        ---------- 
        plot():         plot of series
        est_prob():     estimate probability density function (PDF)
        est_bstats():   estimate key statistical values 
        est_psd_woFS(): estimate PSD via Welch, without temporal information
        plot_psd():     plot power spectral density (PSD)
        plot_prob():    plot probability density function (PDF)
        ----------   
    '''
    def __init__(self, srs,var="x", unit="m/s$^2$", name = ''):
        self.x   = srs               # inputseries
        self.N   = self.x.size       # number of samples
        self.var  = var              # variable name
        self.unit = unit             # unit of series
        self.name = name
        if name:
            self.name = '(' + name + ')'
        '''  basic methods (to be carried out when initialized)  '''
        self.est_bstats()
        self.est_psd_woFS()

    @property # for storage efficiency
    def Xts(self):                                      # two-sided Fourier coefficients
        Xts_woShift = np.fft.fft(self.x)
        return np.fft.fftshift(Xts_woShift)
    @property # for storage efficiency
    def Xos(self):                                      # one-sided Fourier coefficients
        Xts_woShift = np.fft.fft(self.x)
        Nos         = np.ceil((self.N+1)/2)             # number of one-sided Fourier coefficients
        return Xts_woShift[0:int(Nos)] 

    def plot(self):
        ''' plot of series '''  
        plt.figure()
        plt.plot(self.x,'k')
        plt.xlabel("index [-]")
        plt.ylabel("$" + self.var + "$ [" + self.unit + "]")
        plt.show()
        plt.title('realization ' + self.name)

    def est_bstats(self):
        ''' estimate key statistical values 
        -> called by __init__
        '''        
        self.stat = {'mean': np.mean(self.x)}
        self.stat['std'] = np.std(self.x)
        self.stat['var'] = np.var(self.x)
        self.stat['skewness'] = sp.stats.skew(self.x)
        self.stat['kurtosis'] = sp.stats.kurtosis(self.x)
        
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

    def est_prob(self,bins='auto'):
        ''' calculate probability density function (PDF) 
        -> called by est_bstats (thus indirect call by __init__)
        INPUTS:
        :param bins:    scalar or array-like
            number of bins / vector of bin-edges
        '''
        self.prob, bin_edges = np.histogram(self.x, bins, density=True)
        self.grid = (bin_edges[1:]+bin_edges[0:-1])/2
        self.pgauss = 1/ np.sqrt(self.stat['var']*2*np.pi) * np.exp(-(self.grid- self.stat['mean'])**2 / (2 * self.stat['var']))

    def est_psd_woFS(self):
        ''' estimate power spectral density (PSD) via Welch, without any temporal information
        -> called by __init__
        '''
        self.fpsd,self.psd = sp.signal.welch(self.x)
        self.dfpsd          = np.mean(np.diff(self.fpsd)) 

    def plot_psd(self):
        ''' plot power spectral density (PSD) '''
        plt.figure()
        plt.plot(self.fpsd,self.psd,'k')
        plt.xlabel("normalized frequency [x\pi rad/sample]")
        plt.ylabel("$G_{" + self.var + self.var + "}$ [(" + self.unit + ")$^2$/Hz]")
        plt.legend([r'$\sigma^2 =  ${:.3f}'.format(np.sum(self.psd*self.dfpsd)) + ' [(' + self.unit + ')$^2$]'])   
        plt.grid(True)
        plt.title('power spectral density ' + self.name)
        plt.show()    

    def plot_prob(self):
        ''' plot probability density function (PDF) '''
        plt.figure()
        plt.plot(self.grid,self.prob,'k',label = "$p(" + self.var + ")$")
        plt.plot(self.grid,self.pgauss,'g--',label = "$p_g(" + self.var + ")$")
        plt.xlabel("$" + self.var + "$ [" + self.unit + "]")
        plt.ylabel("$p(" + self.var + ")$ [-]")
        plt.legend()
        plt.grid(True)
        plt.title('probability density function ' + self.name)
        plt.show()

###############################################################################
# Inheritance of series for timeseries (including time stamp) 
class timeseries(series):  
    def __init__(self, srs, tvec=None,**kwargs):
        '''
            Attributes
            ----------
            t:         time vector
            dt:        time step
            fs:        sampling frequency
            T:         duration of timeseries
            ----------
            Methods
            ---------- 
            plot():             plot time series
            plot_psd():         plot PSD that was lastly estimated via est_psd()
            plot_X():           plot abs of Fourier coefficients 
            plot_tf():          plot transfer function (interpretation: object is transfer function) 
            plot_ls(method):    plot load spectrum
            plot_FDS(idx):      plot FDSs of signal given bei calculation indices
            plot_nonstat():     plot non-stationarity matrix

            est_psd():                          estimate power spectral density (PSD) 
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
            der_lsEquivalent():             derive quasi-stationary load definition on the basis of the load spectra of the Fatigue Damage Spectrum
            der_iFDS():                     derive load definition on the basis of the inverse Fatigue Damage Spectrum
            
            get_analyticalSignal(fl,fu):                    get analytical signal
            get_posTaperedAnalyticalSignal(fl,fu,wfunc):    get positive(!) tapered analytical signal
            get_corrFactors(fbands,olap,wfunc):             get correction factors
            ----------   
        '''
        if type(srs) is dict:
            if 'var' in srs.keys():
                kwargs.update({'var': srs['var']})
            if 'unit' in srs.keys():
                kwargs.update({'unit': srs['unit']})
            srs,tvec = get_GaussianSeries(srs)  
        if tvec is None:
            tvec = np.linspace(0,1,len(srs))
        super().__init__(srs,**kwargs)        
        self.dt  = np.mean(np.diff(tvec)) 
        self.fs  = 1/self.dt
        self.fNy = self.fs/2
        self.T   = self.N/self.fs
        self.df  = 1/self.T
        self.est_psd()
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

    def est_psd(self,df = 1):
        ''' estimate power spectral density (PSD) 
        -> called by __init__
        '''
        self.fpsd,self.psd = sp.signal.welch(self.x,self.fs,nperseg=round(self.fs/df))
        self.dfpsd = np.mean(np.diff(self.fpsd))
        
    def plot(self):
        ''' plot timeseries
        '''
        legd = "$" + self.var + '$: $\sigma^2$ = {sigq: .2f} [({unit})$^2$]; $ \\beta $ = {beta: .2f} [-]'.format(sigq = self.stat['var'], unit = self.unit,beta = self.stat['kurtosis']) 
        plt.figure()
        plt.plot(self.t,self.x,'k',label = legd)    
        plt.xlabel("t [s]")
        plt.ylabel("$" + self.var + "$ [" + self.unit + "]")
        plt.grid(True)
        plt.legend()
        plt.title('time series ' + self.name)
        plt.show()

    def plot_psd(self):
        ''' plot PSD that was lastly estimated via est_psd()
        '''
        plt.figure()
        plt.plot(self.fpsd,self.psd,'k')
        plt.xlabel("f [Hz]")
        plt.ylabel("$G_{" + self.var + self.var + "}$ [(" + self.unit + ")$^2$/Hz]")
        plt.legend([r'$\sigma^2$ = {:.3f}'.format(np.sum(self.psd*self.dfpsd)) + ' [(' + self.unit + ')$^2$]'])   
        plt.grid(True)
        plt.title('power spectral density ' + self.name)
        plt.show()
                
    def plot_X(self):
        ''' plot abs of Fourier coefficients 
        '''
        plt.figure()
        plt.plot(self.ffper,abs(self.Xts),'k')   
        plt.xlabel("f [Hz]")
        plt.ylabel("|$" + self.var.upper() + "$| [" + self.unit + "/Hz]")
        plt.grid(True)
        plt.title('(absolute) Fourier coefficients ' + self.name)
        plt.show()      

    def plot_tf(self,plot ='bode'):
        ''' plot transfer function
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
        fig, axs = plt.subplots(2)
        axs[0].plot(self.fper,func1(self.Xos),'k')   
        axs[0].set_xlabel("f [Hz]")
        axs[0].set_ylabel(ylbl1)
        axs[0].grid(True)
        axs[0].set_title(tit1)
        axs[1].plot(self.fper,func2(self.Xos),'k')   
        axs[1].set_xlabel("f [Hz]")
        axs[1].set_ylabel(ylbl2)
        axs[1].grid(True)
        axs[1].set_title(tit2)
        fig.show() 

    def est_stats(self):
        ''' estimate key statistical values 
        -> called by __init__
        SOURCES:
            A.G. Davenport, Note on the distribution of the largest value of a random function with application to gust loading, Proceedings of the Institution of Civil Engineers 28 (2) (1964) 187–196
        '''    
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
        self.lk = {'si_x': ampls, 'ni_x': np.cumsum(counts),'mues_x': cycles[:,1]}
        self.lk['npeaks']    = len(ampls)
        self.lk['npeaks_ps'] = self.lk['npeaks']/self.T
        zero_crossings = np.where(np.diff(np.sign(self.x)))[0]
        self.lk['nzerocrossings'] = round(len(zero_crossings)/2)
        self.lk['nzerocrossings_ps'] = self.lk['nzerocrossings']/self.T
        if np.any(si_X) == None: # no specification
            self.lk['si_X'] = np.linspace(1.2*self.lk['si_x'][0],0,1000)
        elif np.isscalar(si_X): # given limit
            self.lk['si_X'] = np.linspace(si_X,0,1000)   
        else: # given vector
            self.lk['si_X'] = si_X
        self.est_dirlik()
        if isinstance(self,qstimeseries):
            self.est_qsLS()

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
        Z  = self.lk['si_X']/(np.sqrt(self.stat['specm0']))    

        self.lk['DK_pk']   = (D1/Q*np.exp(-Z/Q) + D2*Z/R**2*np.exp(-Z**2/(2*R**2)) + D3*Z*np.exp(-Z**2/2) ) / (np.sqrt(self.stat['specm0']))
        self.lk['DK_lk']   = D1*np.exp(-Z/Q) + D2*np.exp(-Z**2/(2*R**2)) + D3*np.exp(-Z**2/2)
        self.lk['DK_lkps'] = self.lk['DK_lk']*self.stat['ny_p']
        self.lk['ni_X']    = self.lk['DK_lkps']*self.T 
        self.lk['DK_D']    = self.stat['ny_p']*np.sqrt(self.stat['specm0'])**m*(D1*Q**m*sp.special.gamma(1+m)+np.sqrt(2)**m*sp.special.gamma(1+m/2)*(D2*abs(R)**m+D3))              

    def plot_ls(self,method = None):
        ''' plot load spectrum
        INPUTS:
        :param method:    (list of) function handle(s)
            function for load spectrum estimation
        SOURCES:
            - FLife - Vibration Fatigue by Spectral Methods: https://github.com/ladisk/FLife
            - J. Slavič, M. Boltežar, M. Mršnik, M. Cesnik, J. Javh, Vibration fatigue by spectral methods: From structural dynamics to fatigue, Elsevier, 2020
        '''
        plt.figure()
        plt.semilogx(self.lk['ni_x'],self.lk['si_x'],'r',label='RFC[$'+self.var+'$]')
        plt.semilogx(self.lk['ni_X'],self.lk['si_X'],'k',label='DK[$'+self.var+'$]')
        if isinstance(self,qstimeseries):
            plt.semilogx(self.qs['ni_X'],self.qs['si_X'],'b',label='QS[DK[$'+self.var+'$]]')
        plt.xlabel("cumulated cycles")
        plt.ylabel("amplitudes [" + self.unit + "]")
        plt.grid(True) 
        plt.title('load spectrum ' + self.name)
        plt.xlim(0.5, 1.2*max(self.lk['ni_x']))
        if method != None:
            sd   = FLife.SpectralData(input=(self.psd,self.fpsd))
            if type(method) == type:
                method = [method] # make method iterable
            for pdfesti in method:
                method_name = str(pdfesti)[::-1]
                method_name = method_name[2:method_name.find(".")][::-1]
                spdf = pdfesti(sd).get_PDF(self.lk['si_X'])
                spdf = spdf/np.sum(spdf)
                loadspectrum = np.cumsum(spdf)*self.T*self.stat['ny_p']
                plt.semilogx(loadspectrum,self.lk['si_X'],label = method_name+'[$'+self.var+'$]')   
        plt.legend()
        plt.show()

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
            :(qs)timeseries object    
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

        '''generate qs-timeseries object (time-domain realization)'''
        sigs = []
        for rr in range(R):
            sigs.append({'psd': np.abs(psd_it[rr,:]),'fpsd':inp['fpsd'],'T':optix['x'][rr]*self.T})
        out = qstimeseries(sigs,var = 'iRLS[' + self.var + ']')

        '''save optimization results'''
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        np.save('res_LSOpti_' + current_time + '.npy', optix['x'])

        '''return object'''
        return out
        
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
        siginp = {'psd': np.abs(psd_it),'fpsd':inp['fpsd'],'T':self.T}
        out = timeseries(siginp,var = 'iFDS[' + self.var + ']')

        '''save optimization results'''
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        np.save('res_FDSOpti_m' + str(m) + '_' + current_time + '.npy', optix['x'])

        '''return object'''
        return out

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
                resp, _ = self.der_sdofResponse(fD = f_0[ff], D = D, func = func)
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

    def plot_fds(self,indx = None):
        '''
        plot FDSs of signal given bei calculation indices
        INPUTS:
            :param indx:    scalar or list
                indices of FDS calculations
        '''  
        if indx == None:
            plot_fds(self.FDScalcs)
        else:
            fsd2Bplotted = [self.FDScalcs[ii] for ii in indx]
            plot_fds(fsd2Bplotted)

    def plot_nonstat(self,**kwargs):
        ''' plot non-stationarity matrix 
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
            newunit = self.unit + '*s^2'
        elif func == 'acc2vel':
            newunit = self.unit + '*s'
        elif func == 'acc2acc':
            newunit = self.unit       
        y      = np.fft.irfft(self.Xos*H*self.N*self.df,n=self.N)
        if 'var' not in kwargs.keys():
            kwargs = dict(kwargs,var = self.var.replace('x','y'))
        if 'unit' not in kwargs.keys():
            kwargs = dict(kwargs,unit = newunit) 
        if 'name' not in kwargs.keys():
            kwargs = dict(kwargs,name = func + ' SDOF response for ' + self.name.replace('(','').replace(')',''))                               
        outobj = timeseries(y,self.t,**kwargs)
        return  outobj, H

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
                output timeseries (response) 
        '''
        H = get_tfSDOF(self.fper,fD = fD, D = D, func = func)
        # H = get_tfSDOF(self.fpsd,fD = fD, D = D, func = func)
        if func == 'acc2dis':
            newunit = self.unit + '$*s^2$'
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
        outobj = timeseries(y,self.t,**kwargs)
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
        return timeseries(np.fft.irfft(Xfilt*self.N*self.df,n=self.N),self.t,**kwargs)
    
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
        return timeseries(np.fft.irfft(Xfilt*self.N*self.df,n=self.N),self.t,**kwargs)
    
    def der_bandpass(self,f,**kwargs):
        ''' derive timeseries of ideal band pass filtered signal
        INPUTS:
            :param f:    scaler
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
        return timeseries(np.fft.irfft(Xfilt*self.N*self.df,n=self.N),self.t,**kwargs)    
    
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
        outobj = timeseries(y,self.t,**kwargs)
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
        outobj = timeseries(y,t)
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
            resp,tf = self.der_sdofResponse(fD = evalf[ii],func = func, D = D)
            ls.append({'si_x': resp.lk['si_x'], 'ni_x': resp.lk['ni_x'], 'max_ni': resp.lk['npeaks'],
                       'si_X': resp.lk['si_X'], 'ni_X': resp.lk['ni_X'], 'fD': evalf[ii]})
            # if isinstance(resp,qstimeseries):
            #     qsdk.append({'si_x': resp.qs['si_X'], 'ni_x': resp.qs['ni_X']})


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
        
 ###############################################################################
# Inheritance of timeseries for qstimeseries (quasi-stationary process) 
class qstimeseries(timeseries):  
    '''
        Attributes
        ----------
        attributes from timeseries...
        sigs:      list of timeseries objects ('stationary components')
        R:         number of stationary processes
        p:         portion of each processes
        ---------- 
        Methods
        ---------- 
        methods from timeseries...
        est_qsLS(): estimate quasi-stationary load spectrum
        ----------   
    '''

    def __init__(self, inp,tukwin = 5e-4,**kwargs): 
        if type(inp) == list:
            inp = [timeseries(psds) for psds in inp]
        N = 0
        x = []
        for sig in inp:
            x.append(sig.x*sp.signal.windows.tukey(sig.N,alpha=tukwin))  
            N += sig.N   
        fs = np.mean([sig.fs for sig in inp])
        dt = 1/fs
        t = np.arange(0,N*dt,dt)
        x = np.concatenate(x)
        minN = np.min((x.shape,t.shape))
        self.sigs = inp 
        super().__init__(x[:minN],t[:minN],**kwargs)  
        self.R    = len(inp)
        self.p    = [sig.N/N for sig in inp]
        
        
    def est_qsLS(self):
        ''' estimate quasi-stationary load spectrum
        INPUTS:
            ---     
        SOURCES:
            A. Trapp, P. Wolfsteiner, Fatigue assessment of non-stationary random loading in the frequency domain by a quasi-stationary Gaussian approximation, International Journal of Fatigue 148 (2021) 106214     
        ''' 
        self.qs = {'si_X': np.linspace(0,self.stat['maxvalue']*1.1,1000)}
        self.qs['ni_X'] = np.zeros(self.qs['si_X'].shape)
        for sig in self.sigs:
            sig.est_ls(si_X = self.qs['si_X'])
            self.qs['ni_X'] = self.qs['ni_X'] + sig.lk['ni_X']  
        
            
                



        