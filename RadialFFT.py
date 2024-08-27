import yt
yt.enable_parallelism()
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy import signal
from scipy import stats
import scipy.ndimage

import matplotlib
print(matplotlib.get_configdir())
plt.style.use('science')

from pysteps.utils import spectral

import matplotlib.pyplot as plt
import numpy as np

import yt

"""
Make a turbulent KE power spectrum.  Since we are stratified, we use
a rho**(1/3) scaling to the velocity to get something that would
look Kolmogorov (if the turbulence were fully developed).

Ultimately, we aim to compute:

                      1  ^      ^*
     E(k) = integral  -  V(k) . V(k) dS
                      2

             n                                               ^
where V = rho  U is the density-weighted velocity field, and V is the
FFT of V.

(Note: sometimes we normalize by 1/volume to get a spectral
energy density spectrum).


"""
def test_rapsd(field):
    rapsd, freq = spectral.rapsd(field, return_freq=True)

    m, n = field.shape
    l = max(m, n)

    if l % 2 == 0:
        assert len(rapsd) == int(l / 2)
    else:
        assert len(rapsd) == int(l / 2 + 1)
    assert len(rapsd) == len(freq)
    assert np.all(freq >= 0.0)
    
def autocorr(x):
    acf = []
    for i in range(0,len(x)):
        ui= x[0:len(x)-i]
        uit = x[i:len(x)]
        Rii = np.mean(ui*uit)/np.var(x)
        acf = np.append(acf, [Rii], axis = 0)
        
        
    return acf

'''
def rad_avg_2dfft(image):
    npix = image.shape[0]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                      statistic = "mean",
                                      bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins
'''
def rad_avg_2dfft(image):
    npix = image.shape[0]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    
    # Normalize the wavenumber to the physical scale
    L = 0.1  # Length of the domain (assuming a square domain)
    knrm_normalized = knrm * 2 * np.pi / L
    
    kbins = np.arange(0.5, npix//2+1, 1.)
    
    # Normalize the bins to the physical scale
    kbins_normalized = kbins * 2 * np.pi / L
    
    kvals = 0.5 * (kbins_normalized[1:] + kbins_normalized[:-1])
    Abins, _, _ = stats.binned_statistic(knrm_normalized, fourier_amplitudes,
                                      statistic = "mean",
                                      bins = kbins_normalized)
    Abins *= np.pi * (kbins_normalized[1:]**2 - kbins_normalized[:-1]**2)
    return kvals, Abins

def calculate_anisotropy(v_k_t, k_perp, k_z):
    num = np.sum(k_perp**2 * np.abs(v_k_t)**2)
    den = np.sum(k_z**2 * np.abs(v_k_t)**2)
    anisotropy = num/den
    return anisotropy

def blend_colors(color1, color2, alpha):
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    r = r1 * alpha + r2 * (1 - alpha)
    g = g1 * alpha + g2 * (1 - alpha)
    b = b1 * alpha + b2 * (1 - alpha)
    return r, g, b

# Choose your background color, e.g., white
bg_color = (1, 1, 1)

# Choose your original fill color, e.g., orange-red
fill_color = (1, 0.27, 0.31)
fill_colorM = (0.5, 0.2, 0.3)
# Set the desired alpha value
alpha = 0.25
# Compute the blended color
blended_color = blend_colors(fill_color, bg_color, alpha)
blended_colorM = blend_colors(fill_colorM, bg_color, alpha)

# f = sorted(glob.glob('/volume2/scratch/Rajab/Simulations/sim1_60x60x40/dc5_and_A006/*.dat'))
f = sorted(glob.glob('/volume2/scratch/Rajab/Simulations/highZL_time/A006_dc5_L3_t20/*.dat'))


FFTn = 120
STN = 400
time_Ave = np.empty((0,FFTn))
SFVtime = np.empty((0,STN))
error = np.empty((0,FFTn))
Gzerr = np.empty((0))
# tn = 100;time = 0;t = 0
# CL = 300; Wn = 150;
#      z = 0.5
# Zn=3;

Lz = 3; Lzn = 160;
Zcl=107; 
Zn=27;

# Zstep = np.linspace(0,159,159,dtype=np.float64)
# for Zc in Zstep:
#     # print(Zc)
#     Zcl = Zc.astype(int)
#     print(Zcl)

time_Ave_V = np.empty((0,FFTn))
time_Ave_B = np.empty((0,FFTn))
time_Ave_Vp = np.empty((0,FFTn))
time_Ave_Bp = np.empty((0,FFTn))
time_Ave_Zp = np.empty((0,FFTn))
time_Ave_Zm = np.empty((0,FFTn))
time_Ave_stdV = np.empty((0,FFTn))
time_Ave_stdB = np.empty((0,FFTn))


t=109
for frame in f[t:117:20]:        # number of datafile​    
    ds = yt.load(frame, unit_system='code')
    # doit(ds)
    ad = ds.all_data()
    level = 2
    ad_grid = ds.covering_grid(level, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions * ds.refine_by**level)
    density = ad_grid['rho'].value
    
    b1 = ad_grid['b1'].value;             b2 = ad_grid['b2'].value;             b3 = ad_grid['b3'].value
    
    m1 = ad_grid['velocity_x'].value;     m2 = ad_grid['velocity_y'].value;     m3 = ad_grid['velocity_z'].value;
    
    pr = ad_grid['e'].value
    x = ad_grid['x'].value
    y = ad_grid['y'].value
    z = ad_grid['z'].value
    zzn = np.size(y[1,1,:])
    
    
    tit_time = ds.current_time.value      #int(str(frame[-6:-4]))
    tit_z = np.round(Lz*Zcl/Lzn,2)
    power_Z = np.empty((0,FFTn))
    power_ZB = np.empty((0,FFTn))
    power_ZVp = np.empty((0,FFTn))
    power_ZBp = np.empty((0,FFTn))
    power_Els_plus = np.empty((0,FFTn))
    power_Els_m = np.empty((0,FFTn))
    stdZ_Vp = np.empty((0,FFTn))
    stdZ_Bp = np.empty((0,FFTn))
    
    for zn in range(Zcl-Zn,Zcl+Zn,1):
        # print(zn)
        rho = density[:,:,zn]#160 for level 2
        x1 = x[:,:,zn];                 y1 = y[:,:,zn];            yn = np.size(y1[1,:]);
        m11 = m1[:,:,zn];m22 = m2[:,:,zn];m33 = m3[:,:,zn];        b11 = b1[:,:,zn];b22 = b2[:,:,zn];b33 = b3[:,:,zn]
        x2 = x1[:,0];    y2 = y1[0,:]
        
        
       
        kymin =1; kymax = 10;
        step = 1
        lengt = 0.1 # lenght of y domain
        KKymax = yn/(lengt*2)
        LOR = yn/2
        Amax = LOR/KKymax    
        interval = lengt / yn
        FS = 1/interval
        # kymin =2; kymax = 25;
        xmin = 200; xmax = 300
        std = np.empty((0,FFTn))
        power_0 = np.empty((0,FFTn))
        
        # m1F = m11[range(xmin,xmax),:];m2F = m22[range(xmin,xmax),:];m3F = m33[range(xmin,xmax),:];rhoF = rho[range(xmin,xmax),:]
        # rapsd, freq = spectral.rapsd(m11,fft_method='fft', return_freq=True)
        m1test = m11[range(xmin,xmax),:];    m2test = m22[range(xmin,xmax),:];m3test = m33[range(xmin,xmax),:];
        b1test = b11[range(xmin,xmax),:];    b2test = b22[range(xmin,xmax),:];b3test = b33[range(xmin,xmax),:];
        
        rhoF = rho[range(xmin,xmax),:]
        xts = 2.4
        newm1 = scipy.ndimage.zoom(m1test, zoom = (xts,1), order=1)
        newm2 = scipy.ndimage.zoom(m2test, zoom = (xts,1), order=1)
        newm3 = scipy.ndimage.zoom(m3test, zoom = (xts,1), order=1)
        newb1 = scipy.ndimage.zoom(b1test, zoom = (xts,1), order=1)
        newb2 = scipy.ndimage.zoom(b2test, zoom = (xts,1), order=1)
        newb3 = scipy.ndimage.zoom(b3test, zoom = (xts,1), order=1)
        newrho = scipy.ndimage.zoom(rhoF,  zoom = (xts,1), order=1)

        ky11, FFTV = rad_avg_2dfft(newrho*(newm1**2+newm2**2+newm3**2)/2)
        ky11, FFTB = rad_avg_2dfft((newb1**2+newb2**2+newb3**2)/2)
        ky11, FFTVp = rad_avg_2dfft(newrho*(newm1**2+newm2**2)/2)
        ky11, FFTBp = rad_avg_2dfft((newb1**2+newb2**2)/2)
        # ky11, FFTVp = rad_avg_2dfft(newrho*(newm1**2+newm2**2+newm3**2)/2)
        # ky11, FFTBp = rad_avg_2dfft((newb1**2+newb2**2+newb3**2)/2)
        
        
        VarV = np.sqrt(newm1**2+newm2**2)
        VarB = np.sqrt(newb1**2+newb2**2)
        # plt.plot(x1,m11[120,:])
        Zp = VarV  + VarB/np.sqrt(newrho)
        Zm = VarV - VarB/np.sqrt(newrho)
        # vmean=np.mean(VarV+VarB)
        # Zp = (VarV - np.mean(VarV)) + (VarB - np.mean(VarB))/np.sqrt(newrho)
        # Zm = (VarV - np.mean(VarV)) - (VarB - np.mean(VarB))/np.sqrt(newrho)
        #ky11, Zpfft = rad_avg_2dfft(Zp)
        #ky11, Zmfft = rad_avg_2dfft(Zm)
        # plt.figure(dpi=300)
        # plt.loglog(ky11[3:20],10**6*ky11[3:20]**(-2))
        # plt.loglog(ky11, FFTV)
        # plt.show()
        
        #power_Els_plus = np.append(power_Els_plus,[Zpfft],axis=0) 
        #power_Els_m = np.append(power_Els_m,[Zmfft],axis=0) 

        power_Z = np.append(power_Z,[FFTV],axis=0) 
        power_ZB = np.append(power_ZB,[FFTB],axis=0) 
        power_ZVp = np.append(power_ZVp,[FFTVp],axis=0) 
        power_ZBp = np.append(power_ZBp,[FFTBp],axis=0) 
        stdZ_Vp = np.append(stdZ_Vp, [np.std(power_ZVp,axis=0)], axis=0)
        stdZ_Bp = np.append(stdZ_Bp, [np.std(power_ZBp,axis=0)], axis=0)
        GVpp = stats.linregress(np.log10(ky11[kymin:kymax]),np.log10(FFTVp[kymin:kymax]))

        
        
        # dt = np.column_stack(([GVpp.slope]))
        # outfile = open('text files/t30_slope_raft.txt','a')
        # np.savetxt(outfile,dt,fmt='%e')
        # outfile.close()
        
      
    AveZ = np.mean(power_Z,axis = 0)
    AveZB = np.mean(power_ZB,axis = 0)
    AveZVp = np.mean(power_ZVp,axis = 0)
    AveZBp = np.mean(power_ZBp,axis = 0)
    # Ave_Els_var = np.mean(power_Els_plus,axis = 0)
    # Ave_Elsm_var = np.mean(power_Els_m,axis = 0)
    AveZ_stdVp = np.mean(stdZ_Vp,axis = 0)
    AveZ_stdBp = np.mean(stdZ_Bp,axis = 0)
    
    time_Ave_V = np.append(time_Ave_V,[AveZ],axis=0) 
    time_Ave_B = np.append(time_Ave_B,[AveZB],axis=0) 
    time_Ave_Vp = np.append(time_Ave_Vp,[AveZVp],axis=0) 
    time_Ave_Bp = np.append(time_Ave_Bp,[AveZBp],axis=0) 
    # time_Ave_Zp = np.append(time_Ave_Zp,[Ave_Els_var],axis=0) 
    # time_Ave_Zm = np.append(time_Ave_Zm,[Ave_Elsm_var],axis=0) 
    time_Ave_stdV = np.append(time_Ave_stdV,[AveZ_stdVp],axis=0) 
    time_Ave_stdB = np.append(time_Ave_stdB,[AveZ_stdBp],axis=0)


    
    GV = stats.linregress(np.log10(ky11[kymin:kymax]),np.log10(AveZ[kymin:kymax]))
    GB = stats.linregress(np.log10(ky11[kymin:kymax]),np.log10(AveZB[kymin:kymax]))
    GVp = stats.linregress(np.log10(ky11[kymin:kymax]),np.log10(AveZVp[kymin:kymax]))
    GBp = stats.linregress(np.log10(ky11[kymin:kymax]),np.log10(AveZBp[kymin:kymax]))
    # Zpp = stats.linregress(np.log10(ky11[kymin:kymax]),np.log10(Ave_Els_var[kymin:kymax]))
    # Zmp = stats.linregress(np.log10(ky11[kymin:kymax]),np.log10(Ave_Elsm_var[kymin:kymax]))
    
    # dt = np.column_stack(([GV.slope, GB.slope, GVp.slope, GBp.slope, Zpp.slope, Zmp.slope]))
    # outfile = open('text files/Lz1p5_pm1_new.txt','a')
    # np.savetxt(outfile,dt,fmt='%e')
    # outfile.close()
    # from scipy.signal import savgol_filter

    # Psder = np.gradient(np.log10(AveZVp[1:]),np.log10(ky11[1:]))
    # iner_slope = savgol_filter(Psder, 15, 1)
    
    # plt.figure(dpi=300)
    # plt.semilogx(ky11[1:],Psder)
    # plt.semilogx(ky11[1:],iner_slope)
    # plt.ylim(-7,0)
    # plt.show()
    # plt.close()
    # ky11 = 10*ky11
    # # print(GVp.slope)
    # # # plt.loglog(ky11,AveZVp)
    
    # plt.figure(dpi=300)
    # plt.plot(np.log10(ky11),np.log10(AveZBp))
    # plt.plot(np.log10(ky11),np.log10(AveZVp),'r')
    # # plt.plot(np.log10(ky11),np.log10(AveZ))
    # # plt.plot(np.log10(ky11),np.log10(AveZV))
    

    # plt.plot(np.log10(ky11[kymin:kymax]),6+(np.log10(ky11[kymin:kymax]))*GVp.slope,'r--')
    # plt.annotate(str(np.round(GVp.slope,2)), (1.67, 2),color='r')
    # plt.plot(np.log10(ky11[kymin:kymax]),5+(np.log10(ky11[kymin:kymax]))*GBp.slope,'b--')
    # plt.annotate(str(np.round(GBp.slope,2)), (2, 1),color='blue')
    
    # #     # plt.loglog(AveAm2)
    # # plt.title(r' t = '+str(np.round(tit_time,1)))
    # plt.legend(('ME$_\perp$','KE$_\perp$'), fontsize=8)
    # plt.ylabel('PS'); plt.xlabel('wavenumber'); 
    # plt.show()
    
    

time_mean_Vp = np.mean(time_Ave_Vp,axis=0)
Gt_Vp = stats.linregress(np.log10(ky11[kymin:kymax]),np.log10(time_mean_Vp[kymin:kymax]))
time_mean_Bp = np.mean(time_Ave_Bp,axis=0)
Gt_Bp = stats.linregress(np.log10(ky11[kymin:kymax]),np.log10(time_mean_Bp[kymin:kymax]))
# kymin1 =3; kymax1 = 12;
# Gt1 = stats.linregress(np.log10(ky11[kymin1:kymax1]),np.log10(time_mean[kymin1:kymax1]))
time_mean_stdV = np.mean(time_Ave_stdV,axis=0)
time_mean_stdB = np.mean(time_Ave_stdB,axis=0)
GVp_std = stats.linregress(np.log10(ky11[kymin:kymax]),np.log10((time_mean_Vp+time_mean_stdV)[kymin:kymax]))
GBp_std = stats.linregress(np.log10(ky11[kymin:kymax]),np.log10((time_mean_Bp+time_mean_stdB)[kymin:kymax]))

plt.figure(dpi=300)
#=============== error fill and Ave spectrum =============

plt.plot(np.log10(ky11),np.log10(time_mean_Bp),'b-.')
plt.plot(np.log10(ky11),np.log10(time_mean_Vp),'r')
plt.axvline(x=np.log10(ky11[1]), color='k', linestyle='--',linewidth=0.5)
plt.axvline(x=np.log10(ky11[17]), color='k', linestyle='--',linewidth=0.5)
plt.fill_between(np.log10(ky11), np.log10(AveZVp-AveZ_stdVp), np.log10(AveZVp+AveZ_stdVp), edgecolor='b', facecolor=blended_color, linewidth=0)
plt.fill_between(np.log10(ky11), np.log10(AveZBp-AveZ_stdBp), np.log10(AveZBp+AveZ_stdBp),  edgecolor='b', facecolor=blended_colorM, linewidth=0)
# plt.fill_between(np.log10(ky11), np.log10(time_mean-error[0,:]), np.log10(time_mean+error[0,:]), alpha=1, edgecolor='b', facecolor='#FF7F50', linewidth=0)
# plt.plot(np.log10(ky11),np.log10(time_mean*ky11**(1.56)))

# plt.fill_between(np.log10(ky11), np.log10(time_mean-Gzerr[0,:]), np.log10(time_mean+error[0,:]),
#     alpha=1, edgecolor='b', facecolor='#FF7F50',
#     linewidth=0)
#=========================================================
#================ slope plot =============================
# plt.plot(np.log10(ky11[kymin:kymax]),-4+(np.log10(ky11[kymin:kymax]))*Gt_Vp.slope,'black')
plt.plot(np.log10(ky11[kymin:kymax]),9.5+(np.log10(ky11[kymin:kymax]))*Gt_Vp.slope,'r--')
plt.annotate(str(np.round(Gt_Vp.slope,2))+r'$\pm$'+str(np.round(np.abs(Gt_Vp.slope-GVp_std.slope),2)), (2.2, 4.75),color='r')

plt.plot(np.log10(ky11[kymin:kymax]),10.25+(np.log10(ky11[kymin:kymax]))*Gt_Bp.slope,'b--')
plt.annotate(str(np.round(Gt_Bp.slope,2))+r'$\pm$'+str(np.round(np.abs(Gt_Bp.slope-GBp_std.slope),2)), (2.75, 6),color='blue')

# plt.plot(np.log10(ky11[kymin1:kymax1]),2+(np.log10(ky11[kymin1:kymax1]))*Gt1.slope,'black')

# plt.plot(np.log10(ky11[kymin1:kymax1]),5+(np.log10(ky11[kymin1:kymax1]))*Gt1.slope)
# plt.plot(np.log10(ky11),-1.5+np.log10(time_mean*ky11**(1.7092)))
# plt.plot(np.log10(ky11),-1.5+np.log10(time_mean*ky11**(4)))

#=========================================================
# plt.title('Average in time (2 period) and z (0.25 width)',fontsize=8)
plt.legend(('ME$_\perp$','KE$_\perp$'), fontsize=8)

plt.ylabel('Log$_{10}$(PSD) [code units]',fontsize=8)
plt.xlabel('Log$_{10}$(k$_\perp$) [code units$^{-1}$]',fontsize=8)
plt.show()








