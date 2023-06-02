# # Use inline matlib plots
# %matplotlib inline

# Import python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Get specific functions from some other python libraries
from math import floor, log
from scipy.stats import skew, kurtosis
from scipy.io import loadmat   # For loading MATLAB data (.dat) files
import pywt

'''
Calculates the FFT of the epoch signal. Removes the DC component and normalizes the area to 1
'''
def calcNormalizedFFT(epoch, lvl, nt, fs):
    
    lseg = np.round(nt/fs*lvl).astype('int')
    D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))
    D[0,:]=0                                # set the DC component to zero
    D /= D.sum()                      # Normalize each channel               

    return D

def defineEEGFreqs():
    
    '''
    EEG waveforms are divided into frequency groups. These groups seem to be related to mental activity.
    alpha waves = 8-13 Hz = Awake with eyes closed
    beta waves = 14-30 Hz = Awake and thinking, interacting, doing calculations, etc.
    gamma waves = 30-45 Hz = Might be related to conciousness and/or perception (particular 40 Hz)
    theta waves = 4-7 Hz = Light sleep
    delta waves < 3.5 Hz = Deep sleep

    There are other EEG features like sleep spindles and K-complexes, but I think for this analysis
    we are just looking to characterize the waveform based on these basic intervals.
    '''
    return (np.array([0.1, 4, 8, 14, 30, 45, 70, 180]))  # Frequency levels in Hz


def calcDSpect(epoch, lvl, nt, nc,  fs):
    
    D = calcNormalizedFFT(epoch, lvl, nt, fs)
    lseg = np.round(nt/fs*lvl).astype('int')
    
    dspect = np.zeros((len(lvl)-1,nc))
    for j in range(len(dspect)):
        dspect[j,:] = 2*np.sum(D[lseg[j]:lseg[j+1],:], axis=0)
        
    return dspect


'''
Computes Shannon Entropy
'''
def calcShannonEntropy(epoch, lvl, nt, nc, fs):
    
    # compute Shannon's entropy, spectral edge and correlation matrix
    # segments corresponding to frequency bands
    dspect = calcDSpect(epoch, lvl, nt, nc, fs)

    # Find the shannon's entropy
    spentropy = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)
    
    return spentropy


'''
Compute spectral edge frequency
'''
def calcSpectralEdgeFreq(epoch, lvl, nt, nc, fs):
    
    # Find the spectral edge frequency
    sfreq = fs
    tfreq = 40
    ppow = 0.5

    topfreq = int(round(nt/sfreq*tfreq))+1
    D = calcNormalizedFFT(epoch, lvl, nt, fs)
    A = np.cumsum(D[:topfreq,:], axis=0)
    B = A - (A.max()*ppow)    
    spedge = np.min(np.abs(B), axis=0)
    spedge = (spedge - 1)/(topfreq-1)*tfreq
    
    return spedge



'''
Calculate cross-correlation matrix
'''
def corr(data, type_corr):
    
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0  # Replace any NaN with 0
    C[np.isinf(C)] = 0  # Replace any Infinite values with 0
    w,v = np.linalg.eig(C)
    #print(w)
    x = np.sort(w)
    x = np.real(x)
    return x


'''
Compute correlation matrix across channels
'''
def calcCorrelationMatrixChan(epoch):
    
    # Calculate correlation matrix and its eigenvalues (b/w channels)
    data = pd.DataFrame(data=epoch)
    type_corr = 'pearson'
    
    lxchannels = corr(data, type_corr)
    
    return lxchannels



'''
Calculate correlation matrix across frequencies
'''
def calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs):
    
        # Calculate correlation matrix and its eigenvalues (b/w freq)
        dspect = calcDSpect(epoch, lvl, nt, nc, fs)
        data = pd.DataFrame(data=dspect)
        
        type_corr = 'pearson'
        
        lxfreqbands = corr(data, type_corr)
        
        return lxfreqbands


def calcActivity(epoch):
    '''
    Calculate Hjorth activity over epoch
    '''
    
    # Activity
    activity = np.nanvar(epoch, axis=0)
    
    return activity





def calcMobility(epoch):
    '''
    Calculate the Hjorth mobility parameter over epoch
    '''
      
    # Mobility
    # N.B. the sqrt of the variance is the standard deviation. So let's just get std(dy/dt) / std(y)
    mobility = np.divide(
                        np.nanstd(np.diff(epoch, axis=0)), 
                        np.nanstd(epoch, axis=0))
    
    return mobility




def calcComplexity(epoch):
    '''
    Calculate Hjorth complexity over epoch
    '''
    
    # Complexity
    complexity = np.divide(
        calcMobility(np.diff(epoch, axis=0)), 
        calcMobility(epoch))
        
    return complexity  



def hjorthFD(X, Kmax=3):
    """ Compute Hjorth Fractal Dimension of a time series X, kmax
     is an HFD parameter. Kmax is basically the scale size or time offset.
     So you are going to create Kmax versions of your time series.
     The K-th series is every K-th time of the original series.
     This code was taken from pyEEG, 0.02 r1: http://pyeeg.sourceforge.net/
    """
    L = []
    x = []
    N = len(X)
    for k in range(1,Kmax):
        Lk = []
        
        for m in range(k):
            Lmk = 0
            for i in range(1,floor((N-m)/k)):
                Lmk += np.abs(X[m+i*k] - X[m+i*k-k])
                
            Lmk = Lmk*(N - 1)/floor((N - m) / k) / k
            Lk.append(Lmk)
            
        L.append(np.log(np.nanmean(Lk)))   # Using the mean value in this window to compare similarity to other windows
        x.append([np.log(float(1) / k), 1])

    (p, r1, r2, s)= np.linalg.lstsq(x, L)  # Numpy least squares solution
    
    return p[0]



def petrosianFD(X, D=None):
    """Compute Petrosian Fractal Dimension of a time series from either two 
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided, 
           recommended to speed up)

    In case 1, D is computed by first_order_diff(X) function of pyeeg

    To speed up, it is recommended to compute D before calling this function 
    because D may also be used by other functions whereas computing it here 
    again will slow down.
    
    This code was taken from pyEEG, 0.02 r1: http://pyeeg.sourceforge.net/
    """
    
    # If D has been previously calculated, then it can be passed in here
    #  otherwise, calculate it.
    if D is None:   ## Xin Liu
        D = np.diff(X)   # Difference between one data point and the next
        
    # The old code is a little easier to follow
    N_delta= 0; #number of sign changes in derivative of the signal
    for i in range(1,len(D)):
        if D[i]*D[i-1]<0:
            N_delta += 1

    n = len(X)
    
    # This code is a little more compact. It gives the same
    # result, but I found that it was actually SLOWER than the for loop
    #N_delta = sum(np.diff(D > 0)) 
    
    return np.log10(n)/(np.log10(n)+np.log10(n/n+0.4*N_delta))


def katzFD(epoch):
    ''' 
    Katz fractal dimension 
    '''
    
    L = np.abs(epoch - epoch[0]).max()
    d = len(epoch)
    
    return (np.log(L)/np.log(d))


def hurstFD(epoch):
    
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.nanstd(np.subtract(epoch[lag:], epoch[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0



def logarithmic_n(min_n, max_n, factor):
    """
    Creates a list of values by successively multiplying a minimum value min_n by
    a factor > 1 until a maximum value max_n is reached.

    Non-integer results are rounded down.

    Args:
    min_n (float): minimum value (must be < max_n)
    max_n (float): maximum value (must be > min_n)
    factor (float): factor used to increase min_n (must be > 1)

    Returns:
    list of integers: min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
                      without duplicates
    """
    assert max_n > min_n
    assert factor > 1
    
    # stop condition: min * f^x = max
    # => f^x = max/min
    # => x = log(max/min) / log(f)
    
    max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
    ns = [min_n]
    
    for i in range(max_i+1):
        n = int(np.floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
            
    return ns



def dfa(data, nvals= None, overlap=True, order=1, debug_plot=False, plot_file=None):
    
    total_N = len(data)
    if nvals is None:
        nvals = logarithmic_n(4, 0.1*total_N, 1.2)
        
    # create the signal profile (cumulative sum of deviations from the mean => "walk")
    walk = np.nancumsum(data - np.nanmean(data))
    fluctuations = []
    
    for n in nvals:
        # subdivide data into chunks of size n
        if overlap:
            # step size n/2 instead of n
            d = np.array([walk[i:i+n] for i in range(0,len(walk)-n,n//2)])
        else:
            # non-overlapping windows => we can simply do a reshape
            d = walk[:total_N-(total_N % n)]
            d = d.reshape((total_N//n, n))
            
        # calculate local trends as polynomes
        x = np.arange(n)
        tpoly = np.array([np.polyfit(x, d[i], order) for i in range(len(d))])
        trend = np.array([np.polyval(tpoly[i], x) for i in range(len(d))])
        
        # calculate standard deviation ("fluctuation") of walks in d around trend
        flucs = np.sqrt(np.nansum((d - trend) ** 2, axis=1) / n)
        
        # calculate mean fluctuation over all subsequences
        f_n = np.nansum(flucs) / len(flucs)
        fluctuations.append(f_n)
        
        
    fluctuations = np.array(fluctuations)
    # filter zeros from fluctuations
    nonzero = np.where(fluctuations != 0)
    nvals = np.array(nvals)[nonzero]
    fluctuations = fluctuations[nonzero]
    if len(fluctuations) == 0:
        # all fluctuations are zero => we cannot fit a line
        poly = [np.nan, np.nan]
    else:
        poly = np.polyfit(np.log(nvals), np.log(fluctuations), 1)
    # if debug_plot:
    #     plot_reg(np.log(nvals), np.log(fluctuations), poly, "log(n)", "std(X,n)", fname=plot_file)
        
    return poly[0]



def higuchiFD(epoch, Kmax = 8):
    '''
    Ported from https://www.mathworks.com/matlabcentral/fileexchange/30119-complete-higuchi-fractal-dimension-algorithm/content/hfd.m
    by Salai Selvam V
    '''
    
    N = len(epoch)
    
    Lmk = np.zeros((Kmax,Kmax))
    
    #TODO: I think we can use the Katz code to refactor resampling the series
    for k in range(1, Kmax+1):
        
        for m in range(1, k+1):
               
            Lmki = 0
            
            maxI = floor((N-m)/k)
            
            for i in range(1,maxI+1):
                Lmki = Lmki + np.abs(epoch[m+i*k-1]-epoch[m+(i-1)*k-1])
             
            normFactor = (N-1)/(maxI*k)
            Lmk[m-1,k-1] = normFactor * Lmki
    
    Lk = np.zeros((Kmax, 1))
    
    #TODO: This is just a mean. Let's use np.mean instead?
    for k in range(1, Kmax+1):
        Lk[k-1,0] = np.nansum(Lmk[range(k),k-1])/k/k

    lnLk = np.log(Lk) 
    lnk = np.log(np.divide(1., range(1, Kmax+1)))
    
    fit = np.polyfit(lnk,lnLk,1)  # Fit a line to the curve
     
    return fit[0]   # Grab the slope. It is the Higuchi FD



def calcFractalDimension(epoch):
    
    '''
    Calculate fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append( [petrosianFD(epoch[:,j])      # Petrosan fractal dimension
                    , hjorthFD(epoch[:,j],3)     # Hjorth exponent
                    , hurstFD(epoch[:,j])        # Hurst fractal dimension
                    , katzFD(epoch[:,j])         # Katz fractal dimension
                    , higuchiFD(epoch[:,j])      # Higuchi fractal dimension
                   #, dfa(epoch[:,j])    # Detrended Fluctuation Analysis - This takes a long time!
                   ] )
    
    return pd.DataFrame(fd, columns=['Petrosian FD', 'Hjorth FD', 'Hurst FD', 'Katz FD', 'Higuichi FD'])
    #return pd.DataFrame(fd, columns=['Petrosian FD', 'Hjorth FD', 'Hurst FD', 'Katz FD', 'Higuichi FD', 'DFA'])
    
    
def calcPetrosianFD(epoch):
    
    '''
    Calculate Petrosian fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(petrosianFD(epoch[:,j]))    # Petrosan fractal dimension
                   
    
    return fd



def calcHjorthFD(epoch):
    
    '''
    Calculate Hjorth fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(hjorthFD(epoch[:,j],3))     # Hjorth exponent
                   
    
    return fd


def calcHurstFD(epoch):
    
    '''
    Calculate Hurst fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(hurstFD(epoch[:,j]))       # Hurst fractal dimension
                   
    
    return fd


def calcHiguchiFD(epoch):
    
    '''
    Calculate Higuchi fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(higuchiFD(epoch[:,j]))      # Higuchi fractal dimension
                   
    
    return fd


def calcKatzFD(epoch):
    
    '''
    Calculate Katz fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(katzFD(epoch[:,j]))      # Katz fractal dimension
                   
    
    return fd


def calcDFA(epoch):
    
    '''
    Calculate Detrended Fluctuation Analysis
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(dfa(epoch[:,j]))      # DFA
                   
    
    return fd


def calcSkewness(epoch):
    '''
    Calculate skewness
    '''
    # Statistical properties
    # Skewness
    sk = skew(epoch)
        
    return sk


def calcKurtosis(epoch):
    
    '''
    Calculate kurtosis
    '''
    # Kurtosis
    kurt = kurtosis(epoch)
    
    return kurt


def calcDSpectDyad(epoch, lvl, nt, nc, fs):
    
    # Spectral entropy for dyadic bands
    # Find number of dyadic levels
    ldat = int(floor(nt/2.0))
    no_levels = int(floor(log(ldat,2.0)))
    seg = floor(ldat/pow(2.0, no_levels-1))

    D = calcNormalizedFFT(epoch, lvl, nt, fs)
    
    # Find the power spectrum at each dyadic level
    dspect = np.zeros((no_levels,nc))
    for j in range(no_levels-1,-1,-1):
        dspect[j,:] = 2*np.sum(D[int(floor(ldat/2.0))+1:ldat,:], axis=0)
        ldat = int(floor(ldat/2.0))

    return dspect


'''
Computes Shannon Entropy for the Dyads
'''
#! PROBLEMES ICI
def calcShannonEntropyDyad(epoch, lvl, nt, nc, fs):
    
    dspect =    (epoch, lvl, nt, nc, fs)
                           
    # Find the Shannon's entropy
    spentropyDyd = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)
        
    return spentropyDyd



def calcXCorrChannelsDyad(epoch, lvl, nt, nc, fs):
    
    dspect = calcDSpectDyad(epoch, lvl, nt, nc, fs)
    
    # Find correlation between channels
    data = pd.DataFrame(data=dspect)
    type_corr = 'pearson'
    lxchannelsDyd = corr(data, type_corr)
    
    return lxchannelsDyd


def removeDropoutsFromEpoch(epoch):
    
    '''
    Return only the non-zero values for the epoch.
    It's a big assumption, but in general 0 should be a very unlikely value for the EEG.
    '''
    return epoch[np.nonzero(epoch)]



#? NEW FEATURES ---------------------------------------------------------------

def calcTWE(epoch, fs, wavelet='db4', levels=range(1, 5)):
    """
    Calculate Tsallis wavelet entropy (TWE) for each column of epoch.
    
    Parameters
    ----------
    epoch : numpy.ndarray
        Array of shape (n_samples, n_channels) containing the EEG signal.
    fs : float
        Sampling frequency of the EEG signal.
    wavelet : str, optional
        Wavelet to use for the wavelet transform. Default is 'db4'.
    levels : range or list of int, optional
        Range of wavelet decomposition levels to use for the TWE calculation.
        Default is range(1, 5).
    
    Returns
    -------
    numpy.ndarray
        Array of shape (n_channels,) containing the TWE value for each channel.
    """
    n_channels = epoch.shape[1]
    TWE = np.zeros(n_channels)
    for i in range(n_channels):
        signal = epoch[:, i]
        coeffs = pywt.wavedec(signal, wavelet, level=max(levels))
        energy = [np.sum(np.square(coeffs[j])) for j in range(len(coeffs))]
        prob = energy / np.sum(energy)
        TWE[i] = -np.sum(prob ** 2)
    return TWE



from scipy.spatial.distance import cdist

def calcApEn(epoch, m=2, r=0.2):
    """
    Calcule l'entropie d'approximation (ApEn) pour chaque colonne de l'epoch donné en entrée.

    Args:
        epoch (numpy array): Un tableau Numpy 2D de forme (n_samples, n_channels) contenant l'ensemble des epochs EEG.
        m (int): Ordre de la similarité, nombre de points de données inclus dans la comparaison. (par défaut 2)
        r (float): Rayon de tolérance. (par défaut 0.2)

    Returns:
        numpy array : Un tableau Numpy 1D contenant l'ApEn pour chaque canal EEG.
    """

    n_samples, n_channels = epoch.shape
    ApEn = np.zeros(n_channels)

    for i in range(n_channels):
        signal = epoch[:, i]
        N = len(signal)

        # Calculate the distance between each pair of windows of length m
        distance_matrix = cdist(np.expand_dims(signal, axis=1), np.expand_dims(signal[:-m], axis=1))
        distance_matrix = distance_matrix[:-m, :]

        # Count the number of windows that are within the tolerance r of each other
        phi_m = np.sum(distance_matrix <= r, axis=1)

        # Calculate phi_m+1 and the ApEn for this channel
        phi_m_plus_1 = np.sum(cdist(np.expand_dims(signal, axis=1), np.expand_dims(signal[1:-m+1], axis=1)) <= r, axis=1)
        ApEn[i] = np.log(np.sum(phi_m) / np.sum(phi_m_plus_1))

    return ApEn


def calc_PP_SampEn(epoch, m=2, r=0.2):
    """
    Calculate peak-to-peak sample entropy (PP-SampEn) for each channel in an epoch.

    Parameters:
    -----------
    epoch : numpy array
        A 2D numpy array of shape (n_samples, n_channels) containing the EEG data.
    m : int
        The embedding dimension (default: 2).
    r : float
        The tolerance level as a fraction of the standard deviation of the signal (default: 0.2).

    Returns:
    --------
    PP_SampEn : numpy array
        A 1D numpy array of length n_channels containing the PP-SampEn values for each channel.
    """
    n_channels = epoch.shape[1]
    PP_SampEn = np.zeros(n_channels)

    for i in range(n_channels):
        signal = epoch[:, i]
        PP = np.max(signal) - np.min(signal)
        PP_SampEn[i] = calc_SampEn(signal, m, r) - calc_SampEn(PP*np.ones(len(signal)), m, r)

    return PP_SampEn


def calc_SampEn(x, m, r):
    """
    Calculate sample entropy (SampEn) for a given signal.

    Parameters:
    -----------
    x : numpy array
        A 1D numpy array containing the signal.
    m : int
        The embedding dimension.
    r : float
        The tolerance level as a fraction of the standard deviation of the signal.

    Returns:
    --------
    SampEn : float
        The SampEn value for the given signal.
    """
    r *= np.std(x)

    # Calculate distance matrix
    N = len(x)
    d = np.zeros((N-m+1, N-m+1))
    for i in range(N-m+1):
        for j in range(i, N-m+1):
            d[i, j] = np.max(np.abs(x[i:i+m] - x[j:j+m]))

    # Calculate number of matches
    B = np.sum(d <= r, axis=1) - 1
    C = np.sum(d <= r, axis=0) - 1

    # Calculate SampEn
    D = np.sum(B) + np.sum(C)
    if D != 0:
        SampEn = -np.log(D / (N-m+1)**2)
    else:
        SampEn = np.nan

    return SampEn


from scipy.signal import welch

def calcSPEn(epoch, fs, method='welch', nperseg=512, m=3, r=0.4):
    """Calculates the Spectrum Entropy (SPEn) of the input signal.

    Parameters:
    epoch (numpy array): input signal with shape (n_samples, n_channels)
    fs (int): sampling frequency of the input signal
    method (str): method used for power spectral density (default='welch')
    nperseg (int): length of each segment for Welch's method (default=256)
    m (int): embedding dimension for sample entropy calculation (default=2)
    r (float): tolerance parameter for sample entropy calculation (default=0.2)

    Returns:
    numpy array: SPEn values for each channel in the input signal
    """

    def calc_Pxx(signal):
        # Calculate the power spectral density
        if method == 'welch':
            f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
        else:
            raise ValueError('Invalid method for power spectral density')
        return f, Pxx

    def calc_A_B(Pxx, m, r):
        """
        Calculate A and B matrices for given power spectral density and parameters m and r
        """
        # Calculate A
        A = np.zeros((Pxx.shape[0]-m, m))
        for i in range(Pxx.shape[0]-m):
            A[i] = Pxx[i:i+m]

        # Calculate B
        B = np.zeros(Pxx.shape[0]-m)
        for i in range(Pxx.shape[0]-m):
            temp = np.abs(A[i] - A[i+1:Pxx.shape[0]-m+i+1])
            B[i] = np.sum(np.prod(temp < r, axis=1))
            
        return A, B

    def calc_entropy(B):
        # Compute the sample entropy for the channel
        matches = np.zeros(Pxx.shape[0]-m)
        for i in range(Pxx.shape[0]-m):
            matches[i] = np.sum(B[i] <= r)
        if np.sum(matches) == 0:
            entropy = 0
        else:
            entropy = -log(np.sum(matches)/(Pxx.shape[0]-m))
        return entropy

    SPEn = np.zeros(epoch.shape[1])
    for i in range(epoch.shape[1]):
        signal = epoch[:, i]
        f, Pxx = calc_Pxx(signal)
        A, B = calc_A_B(Pxx,m,r)
        entropy = calc_entropy(B)
        SPEn[i] = entropy

    return SPEn


def calcSE(signal, m=2, r=0.4):
    """
    Compute the sample entropy of each channel in the signal.

    Args:
        signal (numpy.ndarray): The input signal. The signal should have dimensions (n_samples, n_channels).
        m (int): The embedding dimension for the sample entropy calculation.
        r (float): The tolerance level for the sample entropy calculation.

    Returns:
        numpy.ndarray: An array containing the sample entropy of each channel in the input signal. The array should have
        dimensions (n_channels,).
    """
    # Initialize the sample entropy array
    se = np.zeros(signal.shape[1])

    for c in range(signal.shape[1]):
        # Initialize A, B, and matches
        A = np.zeros((signal.shape[0]-m, m))
        B = np.zeros(signal.shape[0]-m)
        matches = np.zeros(signal.shape[0]-m)

        # Calculate A
        for i in range(signal.shape[0]-m):
            A[i] += signal[i:i+m, c]
            for j in range(i+1, signal.shape[0]-m):
                if np.max(np.abs(A[i,:]-A[j,:])) < r:
                    matches[i] += 1
                    matches[j] += 1

        # Calculate B
        for i in range(signal.shape[0]-m):
            B[i] = np.sum(np.abs(A[i]-signal[i+1:i+m+1, c]), axis=0) / m

        # Compute the sample entropy for the channel
        se[c] = -np.log(np.sum(matches) / (signal.shape[0]-m)**2)

    return se


def calcWE(signal, wavelet='db4', level=4):
    # Create a DataFrame to store the results for each column
    result = []
    
    for i in range((signal.shape[1])):
        # Extract the column as a 1D array
        col_data = signal[:,i]
        
        # Calculate the wavelet entropy for this column
        we = calcWE_single(col_data, wavelet=wavelet, level=level)
        
        # Add the result to the DataFrame
        result.append(we)
        
    return result


def calcWE_single(signal, wavelet='db4', level=4):
    # Calculate the wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Calculate the standard deviation of each level
    stds = []
    for i in range(level+1):
        stds.append(np.std(coeffs[i]))
    
    # Calculate the normalized standard deviations
    norm_stds = stds / np.sum(stds)
    
    # Calculate the wavelet entropy
    we = -np.sum(norm_stds * np.log2(norm_stds))
    
    # Return the wavelet entropy
    return we



def calcSampleEntropy(epoch):
    """
    Calcule l'entropie d'échantillon pour un tableau d'époques.
    """
    m = 2  # Longueur des sous-séquences
    r = 0.2  # Tolérance

    n_epochs, n_channels = epoch.shape
    entropies = []
    for i in range(n_channels):
        data = epoch[:, i]
        N = len(data)

        # Initialize the matrices
        A = np.zeros((N-m+1, m))
        B = np.zeros((N-m+1, m))

        # Populate the matrices
        for i in range(N-m+1):
            A[i,:] = data[i:i+m]
            B[i,:] = data[i:i+m]

        # Calculate the number of matches
        matches = np.zeros((N-m+1,))
        for i in range(N-m+1):
            for j in range(N-m+1):
                if i != j:
                    matches[i] += np.all(np.abs(A[i,:]-B[j,:]) < r)

        # Calculate the sample entropy
        entropy = -np.log(np.mean(matches)/(N-m+1))
        entropies.append(entropy)
    return entropies



#! a verifier
