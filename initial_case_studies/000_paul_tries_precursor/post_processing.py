import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import netCDF4 as ncdf
import os

def get_group_from_abl_stats(folder,file,groupname):
    """
    Reads requested data from abl_statistics nc-file and puts data into arrays
    This function can handle multiple files, if in the same folder and given as a list
    
    Args in:
        folder (str): folder location of the input file(s).
        file (str or lst): input .nc-file(s).
        variable (str): the name of the group.
    
    Args out:
        data (class 'netCDF4._netCDF4.Group'): the requested group(s)
    """
    
    if isinstance(file,list):
        Nfiles = len(file)
        data = np.array([])
        for i in range(Nfiles):
            d = ncdf.Dataset(os.path.join(folder,file[i]))
            groups = list(d.groups.keys())
            data = np.append(data,d.groups[groupname])
    else:
        d = ncdf.Dataset(os.path.join(folder,file))
        groups = list(d.groups.keys())
        data = d.groups[groupname]
            
    return data

#########################

def get_data_from_abl_stats(folder,file,variablename):
    """
    Reads requested data from abl_statistics nc-file and puts data into arrays
    This function can handle multiple files, if in the same folder and given as a list
    
    Args in:
        folder (str): folder location of the input file(s).
        file (str or lst): input .nc-file name(s), as a list if more than one.
        variable (str): the name of the group.
    
    Args out:
        data (class 'numpy.ndarray'): the requested group(s).
    """
    
    if isinstance(file,list):
        Nfiles = len(file)
        data = np.array([])
        for i in range(Nfiles):
            d = ncdf.Dataset(os.path.join(folder,file[i]))
            variables = list(d.variables.keys())
            data = np.append(data,d.variables[variablename])
    else:
        d = ncdf.Dataset(os.path.join(folder,file))
        variables = list(d.variables.keys())
        data = d.variables[variablename]
            
    return data

#########################

def get_data_from_group(
    group,
    variable
):
    """
    Reads requested data from abl_statistics nc-file and puts data into arrays
     
    Args in:
        group (class 'netCDF4._netCDF4.Group' or class 'numpy.ndarray'): 
            group or array of groups as obtained from .nc-file(s).
        variable (str): the required variable to be extracted from group.
    
    Args out:
        data (class 'numpy.ndarray'): the requested variable
    """
    if type(group)==np.ndarray:
        Ngroups = len(group)
        
        for i in range(Ngroups):
            variablenames = list(group[i].variables.keys())

            if variable in variablenames:
                if i == 0:
                    data = np.array(np.array(group[i].variables[variable]))
                else:
                    data = np.append(data,np.array(group[i].variables[variable]),axis=0)
            else:
                raise ValueError(f'The specified variable was not found in the given group. \n Available variables: {variablenames} \n Requested variable: {variable}')
    
    else:
        variablenames = list(group.variables.keys())
        
        if variable in variablenames:
            data = np.array(np.array(group.variables[variable]))
        else:
            raise ValueError(f'The specified variable was not found in the given group. \n Available variables: {variablenames} \n Requested variable: {variable}')
                
    return data

#########################

def average_vel_data(
    u,
    time,
    timespan
    
):
    """
    Function that determines the average wind speed over one or more given timespans.
    
    Args in:
        u (arr): wind velocities in [m/s] in size [Nt x Nh], where Nt is the number of
            time steps, and Nh the number of vertical datapoints.
        time (arr): time array in [s], in size [Nt], where Nt is the number of time steps.
        timespan (lst): timespan to average over in [s], in size [Ns], format [t1, ..., tend].
            Make sure all values in "timespan" are in "time".
            
    Args out:
        uavg (class 'numpy.ndarray'): average wind velocities over given timespans in [m/s], 
            in size [Nh,Ns-1]. If Ns > 2, a matrix is created with averages over [t1, t2], 
            [t2, t3], .., [tend-1, tend].
    """
        
    N = len(timespan)-1
    uavg = np.zeros((u.shape[1],N))
    for i in range(N):
        t_begin = np.squeeze(np.where(time == timespan[i]))
        t_end = np.squeeze(np.where(time == timespan[i+1]))
        if not t_begin.size > 0:
            raise ValueError(f'Time value {timespan[i]} not found in time array, t_max = {time[-1]}.')
        elif not t_end.size > 0:
            raise ValueError(f'Time value {timespan[i+1]} not found in time array, t_max = {time[-1]}.')
        else:
            uavg[:,i] = np.average(u[t_begin:t_end],axis=0)
    
    return uavg
    
#########################

def plot_vert_vel_profiles(
    uavg,
    z,
    ax=None

):
    """
    Function that plots the vertical velocity profile of a precursor simulation
    based on the velocity uavg and the height z.
    
    Args:
        uavg (arr): average velocity in [m/s] for different heights.
        z (lst): heights [m] of velocity datapoints.
        ax (:py:class:'matplotlib.pyplot.axes', optional):
            figure axes. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(uavg,z)
    ax.set_xlabel("U m/s")
    ax.set_ylabel("Height [m]")
    xmax = (np.max(uavg)+1)
    ax.set_xlim([0, xmax])
    ax.grid(True)
    
#########################

def get_ws_magnitude(
    u,
    v
):
    """
    Function that determines the magnitude of the wind speed vector based on
    wind speed measurements in x- and y-direction.
    
    Args in:
        u (class 'numpy.ndarray'): velocity in x-direction [m/s].
        v (class 'numpy.ndarray'): velocity in y-direction [m/s].
        
    Arg out:
        umag (class 'numpy.ndarray'): magnitude of velocity [m/s].
    """
    umag = abs(np.sqrt(u**2+v**2))
    #def signwind(x):
    #    return 0 if abs(x) == 0 else x / abs(x)
    #phi = atan(u/v)+pi-(signwind(v)-1)*pi/2*signwind(u)
    
    return umag
    