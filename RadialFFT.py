import yt
yt.enable_parallelism()
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy import stats
import scipy.ndimage
import matplotlib
print(matplotlib.get_configdir())
plt.style.use('science')



def rad_avg_2dfft(image, L=0.1):
    """
    Compute the radially averaged power spectrum of a 2D FFT of an image.
    
    Parameters:
    ----------
    image : np.ndarray
        2D array representing the image data for which the FFT and radial average are computed.
    L : float, optional
        Physical length of the domain (assuming a square domain). Default is 0.1.
        
    Returns:
    -------
    kvals : np.ndarray
        Array of radially averaged wavenumber values.
    Abins : np.ndarray
        Array of radially averaged Fourier amplitude values corresponding to each wavenumber.
    
    Notes:
    -----
    The function calculates the 2D FFT of the input image, squares the Fourier amplitudes, and then averages them
    in radial bins. The result is the radially averaged power spectrum of the image.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")
    
    # FFT and Fourier amplitude calculations
    npix = image.shape[0]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2
    
    # Compute 2D wavenumbers
    kfreq = np.fft.fftfreq(npix) * npix  # Frequency bins in Fourier space
    kfreq2D = np.meshgrid(kfreq, kfreq)  # 2D grid of frequency bins
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2).flatten()  # Radial wavenumber values
    fourier_amplitudes = fourier_amplitudes.flatten()
    
    # Normalize the wavenumber to the physical scale
    knrm_normalized = knrm * 2 * np.pi / L

    # Radial binning
    kbins = np.arange(0.5, npix//2 + 1, 1.)
    kbins_normalized = kbins * 2 * np.pi / L  # Normalize bins to the physical scale
    kvals = 0.5 * (kbins_normalized[1:] + kbins_normalized[:-1])  # Midpoint of bins
    
    # Compute the radially averaged Fourier amplitudes using binning
    Abins, _, _ = stats.binned_statistic(knrm_normalized, fourier_amplitudes, statistic="mean", bins=kbins_normalized)
   
    # Correct for area of annulus in 2D Fourier space
    Abins *= np.pi * (kbins_normalized[1:]**2 - kbins_normalized[:-1]**2)

    return kvals, Abins




f = sorted(glob.glob('/path/*.dat'))


FFTn = 120; STN = 400
Zcl=107;    Zn=27;   # chosen subdomain in z direction

variables = ['V', 'B', 'Vp', 'Bp', 'stdV', 'stdB']
time_Ave = {f'time_Ave_{var}': np.empty((0, FFTn)) for var in variables}

t=109
for frame in f[t:117:20]:        # number of datafile​    
    ds = yt.load(frame, unit_system='code')
    # doit(ds)
    ad = ds.all_data()
    level = 2
    ad_grid = ds.covering_grid(level, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions * ds.refine_by**level)
    density = ad_grid['rho'].value
    b_fields = [ad_grid[f'b{i}'].value for i in range(1, 4)]
    velocities = [ad_grid[f'velocity_{comp}'].value for comp in ['x', 'y', 'z']]
    coords = [ad_grid[coord].value for coord in ['x', 'y', 'z']]
    zzn = np.size(coords[1][1, 1, :])
    
    
    power_Z = {var: np.empty((0, FFTn)) for var in ['V', 'B', 'Vp', 'Bp']}
    stdZ = {var: np.empty((0, FFTn)) for var in ['Vp', 'Bp']}
    
    for zn in range(Zcl-Zn,Zcl+Zn,1):
        
        rho = density[:, :, zn]
        m = [vel[:, :, zn] for vel in velocities]
        b = [b_field[:, :, zn] for b_field in b_fields]
       
        kymin =1;   kymax = 10;
        xmin = 200; xmax = 300;
        
        # The zoom factor of 2.4 along the x-axis and 1 along the y-axis maintains 
        # the original size along the y-axis while expanding the x-axis, creating a square-like region.

        m_resampled = [scipy.ndimage.zoom(m_comp[range(xmin, xmax), :], zoom=(2.4, 1), order=1) for m_comp in m]
        b_resampled = [scipy.ndimage.zoom(b_comp[range(xmin, xmax), :], zoom=(2.4, 1), order=1) for b_comp in b]
        rho_resampled = scipy.ndimage.zoom(rho[range(xmin, xmax), :], zoom=(2.4, 1), order=1)
        
        # Calculate FFT power for each field
        ky11, FFTV = rad_avg_2dfft(rho_resampled * (m_resampled[0]**2 + m_resampled[1]**2 + m_resampled[2]**2) / 2)
        _, FFTB = rad_avg_2dfft((b_resampled[0]**2 + b_resampled[1]**2 + b_resampled[2]**2) / 2)
        _, FFTVp = rad_avg_2dfft(rho_resampled * (m_resampled[0]**2 + m_resampled[1]**2) / 2)
        _, FFTBp = rad_avg_2dfft((b_resampled[0]**2 + b_resampled[1]**2) / 2)
        
        
        # Append power data
        for var, FFT in zip(['V', 'B', 'Vp', 'Bp'], [FFTV, FFTB, FFTVp, FFTBp]):
            power_Z[var] = np.append(power_Z[var], [FFT], axis=0)
            
        # Append standard deviation data
        stdZ['Vp'] = np.append(stdZ['Vp'], [np.std(FFTVp, axis=0)], axis=0)
        stdZ['Bp'] = np.append(stdZ['Bp'], [np.std(FFTBp, axis=0)], axis=0)
        GVpp = stats.linregress(np.log10(ky11[kymin:kymax]),np.log10(FFTVp[kymin:kymax]))
        
    # Compute means and append to time_Ave  
    for var in ['V', 'B', 'Vp', 'Bp']:
        AveZ = np.mean(power_Z[var], axis=0)
        time_Ave[f'time_Ave_{var}'] = np.append(time_Ave[f'time_Ave_{var}'], [AveZ], axis=0)
    
    for var in ['stdV', 'stdB']:
        AveZ_std = np.mean(stdZ[var[-2:]], axis=0)
        time_Ave[f'time_Ave_{var}'] = np.append(time_Ave[f'time_Ave_{var}'], [AveZ_std], axis=0)
    
    # Linear regression analysis
    regressions = {var: stats.linregress(np.log10(ky11[kymin:kymax]), np.log10(np.mean(power_Z[var], axis=0)[kymin:kymax])) 
                   for var in ['V', 'B', 'Vp', 'Bp']}
    
 







