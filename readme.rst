pyRaTS - processing of (RAndom) TimeSeries for vibration fatigue
---------------------------------------------

Providing an object-oriented framework to analyze and process time series with the focus on random vibration fatigue. PyRaTS  is capable of handling single- and multi-channel, as well as single- and multi-process time series configurations. 
Implementation of the non-stationarity matrix, the Fatigue Damage Spectrum and quasi-stationary signal definitions to deal with the challenges of non-stationary loading. 

Installing this package
-----------------------

Use `pip` to install it by:

.. code-block:: console

    $ pip install pyRaTS

Simple example
---------------

Here is a simple example for running a basic code. Further examples can be found on: 
https://github.com/ArvidTrapp/pyRaTS

.. code-block:: python

    import pyRaTS as ts
    import numpy as np

    # Defining the series by pseudo-random generator...
    T  = 10
    fs = 1024
    N  = T*fs
    x  = np.random.randn(N)

    # Initialize series and some basic plots...
    sig = ts.timeseries(x,name = 'sample timeseries', fs = fs)
    sig.plot()
    sig.plot_prob()
    sig.plot_psd()

    # derive response series and some further basic plots...
    respsig = sig.der_sdofResponse(fD = 50)
    respsig.plot_psd()
    respsig.plot_ls()
	
Some methods for a statistical analysis of random time series / estimation of statistical descriptors
-----------------------------------------

	* spectral moments (est_specMoms)
	* Dirlik estimator (est_dirlik/est_dirlikD)
	* PSD (est_psd)
	* load spectra (est_ls)
	* Fatigue Damage Spectrum (est_fds) ...accepts list of FLife methods for damage estimation
	* non-stationarity matrix (est_nonstat)

Some methods for plotting
-----------------------------------------

	* time series (plot)
	* PSD (plot_psd)
	* absolute of Fourier transform (plot_X)
	* load spectra (plot_ls) ...accepts list of FLife methods with PDF definition
	* transfer function (plot_tf)
	* Fatigue Damage Spectrum (plot_fds) 
	* non-stationarity matrix (plot_nonstat)

Some methods for processing time series
-----------------------------------------

	* statistical response...PSD & Non-stat.-Matrix (der_statResponse(f,H)) 
	* response timeseries of single-degree-of-freedom system (der_sdofResponse(fD, D, func))
	* response timeseries for linear transfer function (der_response(f,H))
	* quasi-stationary load definition on the basis of the load spectra of the Fatigue Damage Spectrum (der_lsEquivalent())
	* load definition on the basis of the inverse Fatigue Damage Spectrum (der_iFDS())
	* ideal high pass filtered signal (der_highpass(f))
	* ideal low pass filtered signal  (der_lowpass(f))
	* ideal band pass filtered signal (der_bandpass(f))
