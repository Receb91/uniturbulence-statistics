

import scipy.ndimage

import yt
yt.enable_parallelism()
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy import stats
from matplotlib.pyplot import cm
import matplotlib
print(matplotlib.get_configdir())
plt.style.use('science')



f = sorted(glob.glob('/volume2/scratch/Rajab/Simulations/highZL_time/A006_dc5_L3_t20/*.dat'))


def analyze_spectral_power(x_path, m1E, m2E, b1E, b2E, SpecL, Xstep):
    """
    Analyze the spectral power of velocity and magnetic fields along a given path.

    Parameters:
    ----------
    x_path : np.ndarray
        Array of x-coordinates along the path.
    m1E, m2E : np.ndarray
        Interpolated velocity components along the path.
    b1E, b2E : np.ndarray
        Interpolated magnetic field components along the path.
    SpecL : int
        Length of the spectrum for Fourier transform.
    Xstep : np.ndarray
        Step sizes for linear regression analysis.
    
    Returns:
    -------
    dict
        A dictionary containing various computed 'var' such as mean power, standard deviations,
        total power, and regression results.
    """
    
    # Initialize storage arrays
    half_length = int(len(m1E[1, :]) / 2) + 1
    var = {key: np.empty((0, half_length)) for key in ['std_Vp', 'std_Bp', 'power_Vp', 'power_Bp']}


    # Fourier Transform and Power Spectrum Analysis
    for i in range(0, len(x_path), 2):
        # Compute power spectra using Welch's method
        ky11, fftVx1 = signal.welch(1.5 * (m1E[i, :]**2 + m2E[i, :]**2) / 2, nfft=SpecL, window='hanning',
                                    scaling='spectrum', nperseg=SpecL//2, average='median')
        _, fftBx1 = signal.welch((b1E[i, :]**2 + b2E[i, :]**2) / 2, nfft=SpecL, window='hanning',
                                 scaling='spectrum', nperseg=SpecL//2, average='median')

        # Append power and standard deviation values
        var['power_Vp'] = np.append(var['power_Vp'], [np.abs(fftVx1)], axis=0)
        var['power_Bp'] = np.append(var['power_Bp'], [np.abs(fftBx1)], axis=0)
        var['std_Vp'] = np.append(var['std_Vp'], [np.std(var['power_Vp'], axis=0)], axis=0)
        var['std_Bp'] = np.append(var['std_Bp'], [np.std(var['power_Bp'], axis=0)], axis=0)

    # Calculate averages and total power
    AveAm = np.mean(var['power_Vp'], axis=0)
    AveBx = np.mean(var['power_Bp'], axis=0)
    Avestd_Vp = np.mean(var['std_Vp'], axis=0)
    Avestd_Bp = np.mean(var['std_Bp'], axis=0)
    total_power_Vp = np.sum(AveAm)
    total_power_Bp = np.sum(AveBx)

    # Wavenumber range for regression
    ky = np.linspace(0, 1200, len(AveAm))


    # Regression analysis for power spectra
    kmin = 3
    kmax = int(len(AveAm) // 9)

    regressions = {
        'GAve': stats.linregress(np.log10(ky[kmin:kmax]), np.log10(AveAm[kmin:kmax])),
        'GAveB': stats.linregress(np.log10(ky[kmin:kmax]), np.log10(AveBx[kmin:kmax])),
        'GVp_std': stats.linregress(np.log10(ky[kmin:kmax]), np.log10((AveAm + Avestd_Vp)[kmin:kmax])),
        'GBp_std': stats.linregress(np.log10(ky[kmin:kmax]), np.log10((AveBx + Avestd_Bp)[kmin:kmax]))
    }

    # Collect results in a dictionary
    results = {
        'mean_power_Vp': AveAm,
        'mean_power_Bp': AveBx,
        'std_power_Vp': Avestd_Vp,
        'std_power_Bp': Avestd_Bp,
        'total_power_Vp': total_power_Vp,
        'total_power_Bp': total_power_Bp,
        'regressions': regressions
    }

    return results



t = 104

for frame in f[t:117:3]:        # number of datafile​    
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
    
   
    for zn in range(70,92,3):
        rho = density[:, :, zn] #160 for level 2
        x_slice, y_slice = coords[0][:, :, zn], coords[1][:, :, zn]
        
        # Extract velocity and magnetic field slices
        vel_slices = [vel[:, :, zn] for vel in velocities]
        b_slices = [b[:, :, zn] for b in b_fields]
        
     # Plot and calculate contours
        plt.figure(dpi=300)
        cs = plt.contour(y_slice, x_slice, rho, [1.5])
        p = cs.collections[0].get_paths()[0]
        plt.close()

        # Extract path vertices
        vertices = p.vertices
        x_path, y_path = vertices[:, 1], vertices[:, 0]

        # Interpolate data over the path
        x2, y2 = x_slice[:, 0], y_slice[0, :]

        interpolators = {
                'm1': interpolate.interp2d(x2, y2, vel_slices[0], kind='quintic'),
                'm2': interpolate.interp2d(x2, y2, vel_slices[1], kind='quintic'),
                'm3': interpolate.interp2d(x2, y2, vel_slices[2], kind='quintic'),
                'b1': interpolate.interp2d(x2, y2, b_slices[0], kind='quintic'),
                'b2': interpolate.interp2d(x2, y2, b_slices[1], kind='quintic'),
                'b3': interpolate.interp2d(x2, y2, b_slices[2], kind='quintic'),
                'rho': interpolate.interp2d(x2, y2, rho, kind='quintic')
            }

        # Interpolate along the contour path
        ynew = x_path
        xnew = y_path
        xxn, yyn = np.meshgrid(x_path, y_path)
        interpolated_data = {key: interpolator(x_path, y_path) for key, interpolator in interpolators.items()}
        
        # Extract and analyze interpolated data
        m1E, m2E, m3E = interpolated_data['m1'], interpolated_data['m2'], interpolated_data['m3']
        b1E, b2E, b3E = interpolated_data['b1'], interpolated_data['b2'], interpolated_data['b3']
        mrhoE = interpolated_data['rho']
               
        
        # Calculate averages
        m1A = np.mean(m1E[:len(m1E[1, :]) // 2, :], axis=0)
        m2A = np.mean(m2E[:len(m2E[1, :]) // 2, :], axis=0)
        
        # Debug output for length
        print(f'len(V) = {len(x_path)}, titZ = {zn * (3/160):.3f}')
        SpecL = len(m2E[1,:])
     
       
        # Continue or skip based on path length
        if len(x_path) <= 240:
            continue  # Skip further processing if the path is too short

       
       # Define the Xstep range (for completeness, as it was used in deleted parts)
        Xstep = np.arange(1, len(m1E[1, :]) + 1)
        
        # Call the function to analyze spectral power
        results = analyze_spectral_power(x_path, m1E, m2E, b1E, b2E, SpecL, Xstep)
        
        # Extract results from the function
        AveAm = results['mean_power_Vp']
        AveBx = results['mean_power_Bp']
        Avestd_Vp = results['std_power_Vp']
        Avestd_Bp = results['std_power_Bp']
        total_power_Vp = results['total_power_Vp']
        total_power_Bp = results['total_power_Bp']
        regressions = results['regressions']
        
      