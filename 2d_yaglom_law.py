import yt
yt.enable_parallelism()
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy import stats
import matplotlib
print(matplotlib.get_configdir())
plt.style.use('science')


def custom_roll(A, x_lag, y_lag):
    
    """
    Custom roll function to shift the 2D array A by x_lag and y_lag.
    
    Parameters:
    A : ndarray
        Input 2D array to be rolled.
    x_lag, y_lag : int
        Number of positions to shift along the x-axis and y-axis.
    
    Returns:
    A_roll_xy : ndarray
        Rolled array with the specified shifts in x and y directions.
    """
    
    # Roll along y-axis (axis=1)
    A_roll_y = np.roll(A, y_lag, axis=1)

    # Roll along x-axis (axis=0)
    A_roll_xy = np.roll(A_roll_y, x_lag, axis=0)

    # Fill first two rows with a copy of the rolled values from the first two rows
    A_roll_xy[:x_lag] = A_roll_y[:x_lag].copy()

    return A_roll_xy


def yaglom2D(varp, varm, max_lag): # varp and varm are the Z^+ and Z^- elsasser variables respectively
    
    """
    Calculates the 3rd order structure function based on Yaglom's law for 2D projection ()
    
    Parameters:
    varp, varm : dict
        Dictionary containing 'x' and 'y' components of Z^+ and Z^- Elsasser variables as 2D arrays.
  
    max_lag : int
        Maximum lag distance to compute the structure function.
    
    Returns:
    sf_values : ndarray
        Array of structure function values for each lag distance.
    """
    
    assert varp['x'].shape == varm['x'].shape, "Both input arrays must have the same shape."
    
    # Initialize arrays to store structure function values and counts of lags
    sf_values = np.zeros(max_lag)
    num_lags = np.zeros(max_lag)

    # Loop over lag distances in both x and y directions
    for lag_x in range(max_lag):
        for lag_y in range(max_lag):
            vector_distance = int(np.sqrt(lag_x**2 + lag_y**2))
            if vector_distance >= max_lag:
                continue # Skip if vector distance exceeds max_lag
            
            # Shift the arrays by (lag_x, lag_y) using custom_roll function
            rolled_varp_x = custom_roll(varp['x'], lag_x, lag_y)
            rolled_varp_y = custom_roll(varp['y'], lag_x, lag_y)
            rolled_varm_x = custom_roll(varm['x'], lag_x, lag_y)
            rolled_varm_y = custom_roll(varm['y'], lag_x, lag_y)

            # Calculate the differences between the original and rolled arrays
            diff_varp_x = varp['x'] - rolled_varp_x
            diff_varp_y = varp['y'] - rolled_varp_y
            diff_varm_x = varm['x'] - rolled_varm_x
            diff_varm_y = varm['y'] - rolled_varm_y

            # Compute the magnitude of the lag vector
            magnitude_rl = np.sqrt(lag_x**2 + lag_y**2)
            
            # Project the difference of varp onto the lag vector
            proj_diff_varp = (lag_x * diff_varp_x + lag_y * diff_varp_y) / magnitude_rl**2

            # Calculate the structure function components in x and y directions
            sf_value_x = np.mean(proj_diff_varp * diff_varm_x**2)
            sf_value_y = np.mean(proj_diff_varp * diff_varm_y**2)
            
            # Average the structure function components
            sf_value = (sf_value_x + sf_value_y) / 2
           
            # Accumulate the structure function value and the count of lags
            sf_values[vector_distance] += sf_value
            num_lags[vector_distance] += 1
            
    # Normalize the structure function values by the number of lags
    sf_values /= num_lags
    return sf_values



f = sorted(glob.glob('/path/*.dat'))

Zcl = 80; Zn = 79

time_list = [] # To store the time values from each data file

tn = 200 
frame_time_step = tn  # Set the frame time step
num_lags = 120 // frame_time_step  # Set the number of lags you want to consider
time_lags = np.arange(0, num_lags * frame_time_step, frame_time_step)
for frame in f[100:180:1]:
    
    ds = yt.load(frame, unit_system='code')   
    
    time_list.append(ds.current_time.value)
    
    ad = ds.all_data()
    level = 2
    ad_grid = ds.covering_grid(level, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions * ds.refine_by**level)
    
    density = ad_grid['rho'].value;       pr = ad_grid['e'].value;
    b1 = ad_grid['b1'].value;             b2 = ad_grid['b2'].value;            b3 = ad_grid['b3'].value;
    m1 = ad_grid['velocity_x'].value;     m2 = ad_grid['velocity_y'].value;    m3 = ad_grid['velocity_z'].value;
    
    max_lag=70
    power_Zp = np.empty((0,max_lag))
    power_Zm = np.empty((0,max_lag))
    xmin=100; xmax=340; # to focus where the small scale structures concentrated
    for zn in range(0,160,3): # 160 is the total iteration in the z direction
        
        rho = density[:,:,zn]
        m11 = m1[:,:,zn];m22 = m2[:,:,zn];m33 = m3[:,:,zn]
        b11 = b1[:,:,zn];b22 = b2[:,:,zn];b33 = b3[:,:,zn]
          
        VarV_x = m11[range(xmin,xmax),:]
        VarV_y = m22[range(xmin,xmax),:]
        VarB_x = b11[range(xmin,xmax),:]
        VarB_y = b22[range(xmin,xmax),:]
    
        Zp = {'x': (VarV_x - np.mean(VarV_x)) + (VarB_x - np.mean(VarB_x)) / np.sqrt(rho[range(xmin,xmax),:]),
              'y': (VarV_y - np.mean(VarV_y)) + (VarB_y - np.mean(VarB_y)) / np.sqrt(rho[range(xmin,xmax),:])}
        
        Zm = {'x': (VarV_x - np.mean(VarV_x)) - (VarB_x - np.mean(VarB_x)) / np.sqrt(rho[range(xmin,xmax),:]),
              'y': (VarV_y - np.mean(VarV_y)) - (VarB_y - np.mean(VarB_y)) / np.sqrt(rho[range(xmin,xmax),:])}
    
        SFVm = yaglom2D(Zp,Zm,max_lag)
        SFVp = yaglom2D(Zm,Zp,max_lag)
        power_Zp = np.append(power_Zp,[SFVp],axis=0) 
        power_Zm = np.append(power_Zm,[SFVm],axis=0) 
        
        
    AvezZp = np.mean(power_Zp,axis=0)    
    AvezZm = np.mean(power_Zm,axis=0)    
  
    meanSF = (AvezZp+AvezZm)/2 

    # Separate meanSF into positive and negative contributions
    meanSF_positive = meanSF.copy()
    meanSF_negative = meanSF.copy()
    meanSF_positive[meanSF < 0] = 0
    meanSF_negative[meanSF > 0] = 0
       
    pl = 3/4
        
    Gstr = stats.linregress(np.log10(lagR1[rn:gn]),np.log10(-pl * meanSF[rn:gn] / lagR1[rn:gn]))
    
    
    #dt = np.column_stack(([meanSF]))
    #outfile = open('Files/yaglom_law_2D.txt','a')
    #np.savetxt(outfile,dt,fmt='%e')
    #outfile.close()
   
