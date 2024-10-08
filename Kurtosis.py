
import yt
yt.enable_parallelism()
import matplotlib.pyplot as plt
import glob
from scipy.stats import kurtosis

import numpy as np
import matplotlib
print(matplotlib.get_configdir())
plt.style.use('science')


def inc_vec(vec,lag):
    
    i_delta = np.zeros(len(vec)-lag)
    for i in range(len(vec)-lag):
        i_delta[i] = vec[i + lag] - vec[i]
        
    return i_delta
'''
    vec - the input vector for which differences will be computed
    lag - how far ahead the difference is calculated

The function creates an array of zeros called i_delta with a length of 'len(vec) - lag'. 

Then, it iterates over vec with a for loop, and for each iteration, it calculates the difference between 
The current value and the value lag a step ahead and are assigned to the corresponding index in i_delta.

Example:

vec = [1, 2, 3, 4, 5]
lag = 2
inc_vec(vec, lag)

# Output: [2.0, 2.0, 2.0]

Here, the function calculates the difference between each consecutive pair of elements with a lag of 2. 
The first element of the returned array is the difference between the elements at index 2 and index 0, 
the second element is the difference between the elements at index 3 and index 1, and so on.
'''


f = sorted(glob.glob('/path/*.dat'))



CL = 200; Wn = 100; Kn = 100; t=10

variables = ['Vx', 'Vy', 'Vz', 'Bx', 'By', 'Bz', 'V']
delta_aveZ = {f'delta{var}_aveZ': np.empty((0, Kn)) for var in variables}
# Each array is associated with a meaningful key ('deltaVx_aveZ', etc.)

for frame in f[0:t:3]:        # number of datafile​

    ds = yt.load(frame, unit_system='code')
    ad = ds.all_data()
    level = 3
    ad_grid = ds.covering_grid(level, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions * ds.refine_by**level)
    density = ad_grid['rho'].value
    
    b1 = ad_grid['b1'].value;             b2 = ad_grid['b2'].value;             b3 = ad_grid['b3'].value
    m1 = ad_grid['velocity_x'].value;     m2 = ad_grid['velocity_y'].value;     m3 = ad_grid['velocity_z'].value;
    
    pr = ad_grid['e'].value
    
    # Define the variables you want to initialize
    variables = ['Vx', 'Vy', 'Vz', 'Bx', 'By', 'Bz', 'V']
    
    # Initialize empty arrays for kurtosis statistics and results
    kur_mean = {var: np.empty((0)) for var in variables}
    delta = {var: np.empty((0)) for var in variables}
    
    # Define Zcl and Zn -------------- total iteration in the z-axis is 160
    Zcl, Zn = 107, 27 # Zcl - central point; Zn - distance from the center, as subdomain in 3D
    zn_range = range(Zcl - Zn, Zcl + Zn, 5)
    Zn_iterations = len(zn_range)
    
    for zn in zn_range:
            
        rho = density[:, :, zn]  # iteration in z-direction ---- the energy propagates along z-axis
       
        m = [m1[:, :, zn], m2[:, :, zn], m3[:, :, zn]]
        b = [b1[:, :, zn], b2[:, :, zn], b3[:, :, zn]]
        
        # Initialize the kurtosis mean arrays for this iteration
        kur_mean = {var: np.empty((0)) for var in variables}
        
        #--------------- Kurtosis statistics --------------------------------------
        for n in np.arange(0, Kn, 1):
            for xn in np.arange(CL-Wn, CL+Wn, 1):
                # Compute and store kurtosis values for each variable
                for i, var in enumerate(variables[:3]):
                    vecV = inc_vec(m[i][xn, :] - np.mean(m[i][xn, :]), n)
                    kurV = kurtosis(vecV, fisher=True)
                    kur_mean[var] = np.append(kur_mean[var], [kurV], axis=0)
                    
                for i, var in enumerate(variables[3:6]):
                    vecB = inc_vec(b[i][xn, :] - np.mean(b[i][xn, :]), n)
                    kurB = kurtosis(vecB, fisher=True)
                    kur_mean[var] = np.append(kur_mean[var], [kurB], axis=0)
                    
                MagnitudeV_perp = inc_vec(np.sqrt((m[0][xn, :] - np.mean(m[0][xn, :]))**2 + (m[1][xn, :] - np.mean(m[1][xn, :]))**2), n)
                kurV = kurtosis(MagnitudeV_perp, fisher=True)
                kur_mean['V'] = np.append(kur_mean['V'], [kurV], axis=0)
            
            # Calculate mean kurtosis for each variable
            for var in variables:
                mean_Kur = np.mean(kur_mean[var])
                delta[var] = np.append(delta[var], [mean_Kur], axis=0)
        
        # Average kurtosis over Zn_iterations for each variable and store the results
        delta_aveZ = {}
        for var in variables:
            mean_Kur_inz = np.mean(delta[var].reshape(Zn_iterations, Kn), axis=0)
            delta_aveZ[var] = np.append(delta_aveZ.get(var, np.empty((0))), [mean_Kur_inz], axis=0)


def imshow_grid(data_list):
    # Set up the figure and axes
    fig, axes = plt.subplots(3, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.2, 'hspace': 0.3})
    
    # Display each image on its own axis
    for i, data in enumerate(data_list):
        row = i // 2
        col = i % 2
        im = axes[row, col].imshow(data, extent=[0, 100, 0, 30], origin='lower', interpolation='lanczos', cmap='seismic', vmax=3, vmin=-3, aspect='auto')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        
        axes[row, col].set_title([r'V$_x$', r'B$_x$', r'V$_y$', r'B$_y$',  r'V$_z$',r'B$_z$'][i], fontsize=15)
        if row == 2:
            axes[row, col].set_xlabel(r'$\ell$', fontsize=14)
            if col == 0:
                axes[row, col].set_xticks([0, 25, 50, 75, 100])
            if col == 1:
                axes[row, col].set_xticks([0, 25, 50, 75, 100])
        if col == 0:
            # axes[row, col].set_ylabel('time', fontsize=14)
            if row == 0:
                axes[row, col].xaxis.set_ticks_position('top')
            if row == 0:
                axes[row, col].set_yticks([0, 10, 20, 30])
                # axes[row, col].set_ylabel('time', fontsize=14)
            if row == 1:
                axes[row, col].set_yticks([0, 10, 20, 30])
                axes[row, col].set_ylabel('time', fontsize=14)
            if row == 2:
                 axes[row, col].set_yticks([0, 10, 20, 30])
                 # axes[row, col].set_ylabel('time', fontsize=14)
    
    # Set the same x and y limits for all axes
    for ax in axes.flat:
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 30])
    
    # Remove the top and right spines
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, extend='both',pad=0.05)
    cbar.ax.tick_params(labelsize=15)
    # Show the plot
    plt.show()


tn = 1 

newarrV = {var: delta_aveZ[f'delta{var}_aveZ'].reshape(tn, Kn) for var in ['Vx', 'Vy', 'Vz']}
newarrB = {var: delta_aveZ[f'delta{var}_aveZ'].reshape(tn, Kn) for var in ['Bx', 'By', 'Bz']}

newarrVx, newarrVy, newarrVz = newarrV['Vx'], newarrV['Vy'], newarrV['Vz']
newarrBx, newarrBy, newarrBz = newarrB['Bx'], newarrB['By'], newarrB['Bz']

aa = [newarrVx,newarrBx,newarrVy,newarrBy,newarrBz,newarrVz]
plt.figure(dpi=400)
imshow_grid(aa)



t = np.linspace(0.0004,0.04,len(newarrBx[0,:])-1)
plt.figure(dpi=300)
plt.loglog(t,newarrBx[0,range(1,100)]+3,'--',label=r'$B_x$')
plt.loglog(t,newarrBy[0,range(1,100)]+3,'-.',label=r'$B_y$')
plt.loglog(t,newarrBz[0,range(1,100)]+3,label=r'$B_z$')
plt.xlabel(r'$\ell$');plt.ylabel(r'$K$',rotation=0);
plt.yticks([3, 5, 10], [r'$3 \times 10^0$', r'$5 \times 10^0$', r'$10^1$'])

# Turn off minor ticks
plt.minorticks_off()
plt.ylim([0,12])
plt.axhline(y=3, color='k', linestyle='--',linewidth=0.5)
plt.legend()

plt.show()



'''
# plt.axhline(y=0)
plt.imshow(newarrVx, extent=[0,100,0,30],origin= 'lower', interpolation='lanczos', cmap='seismic',vmax=3,vmin=-3)
plt.colorbar(extend='both',orientation="horizontal");
# plt.plot(newarr[120,:])
plt.xlabel(r'k$_y$ (V$_\perp$)');plt.ylabel('time');
# plt.legend()
plt.show()
'''

# dt = np.column_stack(([newarrVx]))
# outfile = open('text files/kurtosis_Vx.txt','a')
# np.savetxt(outfile,dt,fmt='%e')
# outfile.close()







