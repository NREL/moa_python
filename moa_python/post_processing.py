import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as ncdf
import pandas as pd
import os


def plot_vert_vel_profile(
    file,
    timespan,
    ax=None
):
    """
    Function that plots the vertical velocity profile of a precursor simulation
    based on the abl_statistics nc-file data.
    Needs packages Numpy as np and netCDF4 as ncdf?
    
    Args:
        file (str): Name of input file
        timespan (list, tuple, np.array): Timespan to average over format [t_begin, t_end]
        ax (:py:class:`matplotlib.pyplot.axes` optional):
            figure axes. Defaults to None.
    """
   
    d = ncdf.Dataset(file)
    g = d.groups["mean_profiles"]
    u = g.variables["u"]
    time = np.array(d.variables["time"])
   
    z = np.array(g.variables["h"])
   
    #TODO: What are units of time span?
    t_begin = np.squeeze(np.where(time == timespan[0]))
    t_end = np.squeeze(np.where(time == timespan[1]))
    uavg = np.average(u[t_begin:t_end],axis=0)
       
    #TODO Do we want this inside or outside the function?
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(uavg,z)
    ax.set_xlabel("U m/s")
    ax.set_ylabel("Height [m]")
    ax.set_xlim([0, 12])
    ax.grid(True)