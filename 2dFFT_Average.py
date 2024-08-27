#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:43:35 2021

@author: rajabi
"""



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



def rad_avg_2dfft(data):
    npix = data.shape[0]
    fourier_data = np.fft.fftn(data)
    fourier_amplitudes = np.abs(fourier_data)**2
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic = "mean", bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins



# f = sorted(glob.glob('/volume2/scratch/Rajab/Simulations/sim1_60x60x40/dc5_and_A006/*.dat'))


f = sorted(glob.glob('/volume2/scratch/Rajab/Simulations/sim1_60x60x40/dc2/*.dat'))
# f = sorted(glob.glob('/volume2/scratch/Rajab/Simulations/new_hr/dtfiles/*.dat'))
time_Ave = np.empty((0,120))
tn = 100;time = 0;t = 0
for frame in f[10:60:10]:           # number of datafile​
    power_0 = np.empty((0,120))
    power_1 = np.empty((0,100))
    ds = yt.load(frame, unit_system='code')
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
    #plt.figure()
    tit_time = 0.1*int(str(frame[-6:-4]))
    
    #zn = 40
    for zn in range (100,160,10):
        # print(zn)
        rho = density[:,:,zn]#160 for level 2
        x1 = x[:,:,zn]
        y1 = y[:,:,zn]
        yn = np.size(y1[1,:]);
        m11 = m1[:,:,zn];m22 = m2[:,:,zn];m33 = m3[:,:,zn]
    
        x2 = x1[:,0];    y2 = y1[0,:]
        

        step = 1
        lengt = 0.1 # lenght of y domain
        kymax = yn/(lengt*2)
        LOR = yn/2
        Amax = LOR/kymax    
        rad_avg_2dfft



##------------------------------------------------------------------------------------       
##============================== 2D Fourier transform ================================
##------------------------------------------------------------------------------------
        # ky11 =kymax*np.fft.rfftfreq(y1[120,:].shape[0])
        # ky11 = ky11[range(0,120)]
        ky11, fourier_amplitudes = signal.welch(m11[range(100,140),:], 2000, nfft=121, window='hanning',nperseg=121, scaling='density')
        ky11 = ky11[range(0,120)]
        # fftVx = np.fft.fft2(m11[range(90,150),:])
        # fftV = np.abs(fftVx*fftVx.conjugate())#+fftVy*fftVy.conjugate()+fftVz*fftVz.conjugate()
        # fourier_amplitudes = np.abs(fftVx)**2
        # fourier_amplitudes = fourier_amplitudes[0:240,0:int(len(m11)/2)]
        fourier_amplitudes1 = np.mean(fourier_amplitudes[:,range(0,120)],axis =0)
        # fourier_amplitudes2 = np.mean(fourier_amplitudes,axis =1)
        # ps = fftV[0:int(len(m11)/2),0:int(len(m11)/2)]
        power_0 = np.append(power_0,[fourier_amplitudes1],axis=0)   
        # power_1 = np.append(power_1,[fourier_amplitudes2],axis=0)   
            
        AveAm = np.mean(power_0,axis = 0)
        # AveAm2 = np.mean(power_1,axis = 0)
    # Avehalf = np.empty((len(m11)//2,len(m11)//2))
    # Avehalf = np.append(Avehalf, [AveAm[range(0,200),range(0,200)]])
    # fourier_amplitudes1 = np.mean(fourier_amplitudes,axis =1)
    # plt.plot(ky11, AveAm)
    plt.plot(np.log10(ky11),np.log10(AveAm))
    G = stats.linregress(np.log10(ky11[2:40]),np.log10(AveAm[2:40]))
    plt.plot(np.log10(ky11[2:40]),1+(np.log10(ky11[2:40]))*G.slope)
    # plt.loglog(AveAm2)
    plt.title('2D FFT, Averaged Amplitude in y ('+'t = '+str(np.round(tit_time,3))+')')
    plt.legend(('Ave spectrum','slope = '+str(np.round(G.slope,4)),'slope'), fontsize=13)
    plt.ylabel('Amplitude'); plt.xlabel('wavenumber'); 
    plt.show()
    time_Ave = np.append(time_Ave,[AveAm],axis=0)   
kymin = 2; kymax = 30;
time_mean = np.mean(time_Ave,axis=0)
plt.plot(np.log10(ky11),np.log10(time_mean))
Gt = stats.linregress(np.log10(ky11[kymin:kymax]),np.log10(time_mean[kymin:kymax]))
plt.plot(np.log10(ky11[kymin:kymax]),1+(np.log10(ky11[kymin:kymax]))*Gt.slope)
# plt.loglog(AveAm2)
plt.title('2D FFT, Averaged Amplitude in y ('+'t = '+str(tit_time)+')')
plt.legend(('Ave spectrum','slope = '+str(np.round(Gt.slope,4)),'slope'), fontsize=13)
plt.ylabel('Amplitude'); plt.xlabel('wavenumber'); 
plt.show()













