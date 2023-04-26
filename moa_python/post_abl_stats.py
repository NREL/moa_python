
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as ncdf


class Post_abl_stats:
    
    def __init__(self, filenames):
        
        # If only one filename passed in
        # Convert to list for consistency
        if not isinstance(filenames, (list, tuple, np.ndarray)):
            filenames = [filenames]
        
        self.Nfiles = len(filenames)
        
        # Load the datasets
        self.dataset_list = []
        for filename in filenames:
            self.dataset_list.append(ncdf.Dataset(filename))
        
        # Read time
        self.time = self.get_variable_from_abl_stats('time')
        self.num_time_steps = len(self.time)
        
        # Save the mean profiles
        self.mean_profiles = self.get_group_from_abl_stats('mean_profiles')
        
        # Save the z-levels
        self.z = self.get_data_from_mean_profiles('h')
        
        # Print a quick summar
        self.summary()
        
    def summary(self):
        """
        Print out a brief summary of abl_stats file
        """
        
        print(f"Object is composed of {self.Nfiles} and time runs from {self.time[0]} to {self.time[-1]}")
        print(self.dataset_list[0])
        
    def get_variable_from_abl_stats(self, variablename):
        """
        Reads requested variable from dataset_list and puts data into arrays
        
        Args in:
            variable (str): the name of the group.

        Args out:
            data (class 'netCDF4._netCDF4.Group'): the requested group(s)
        """

        data = np.array([])
        for i in range(self.Nfiles):
            data = np.append(data,self.dataset_list[i].variables[variablename])

        return data
        
    def get_group_from_abl_stats(self, groupname):
        """
        Reads requested data from dataset_list and puts data into arrays
        
        Args in:
            variable (str): the name of the group.

        Args out:
            data (class 'netCDF4._netCDF4.Group'): the requested group(s)
        """

        data = np.array([])
        for i in range(self.Nfiles):
            data = np.append(data,self.dataset_list[i].groups[groupname])

        return data
    
    
    def get_data_from_group(self, 
        group,
        variable
    ):
        """
        Reads requested data from within a group and returns array

        Args in:
            group (class 'netCDF4._netCDF4.Group' or class 'numpy.ndarray'): 
                group or array of groups as obtained from .nc-file(s).
            variable (str): the required variable to be extracted from group.

        Args out:
            data (class 'numpy.ndarray'): the requested variable
        """

        for i in range(self.Nfiles):
            
            variablenames = list(group[i].variables.keys())

            if variable in variablenames:
                if i == 0:
                    data = np.array(np.array(group[i].variables[variable]))
                else:
                    data = np.append(data,np.array(group[i].variables[variable]),axis=0)
            else:
                raise ValueError(f'The specified variable was not found in the given group. \n Available variables: {variablenames} \n Requested variable: {variable}')


        return data
    
    def get_data_from_mean_profiles(self, variable
    ):
        
        """
        Reads requested data from mean profile and returns array

        Args in:
            group (class 'netCDF4._netCDF4.Group' or class 'numpy.ndarray'): 
                group or array of groups as obtained from .nc-file(s).
            variable (str): the required variable to be extracted from group.

        Args out:
            data (class 'numpy.ndarray'): the requested variable
        """
        return self.get_data_from_group(self.mean_profiles, variable)
    
    def time_average_data(self, x, t_min=None, t_max=None):
        
        """
        Averages the data (x) over time period [t_min, t_max)

        Args in:
            x (class 'numpy.ndarray'): 
                an np.array of length = self.num_time_steps
            t_min (float): time to start averaging (inclusive)
                if None, defaults to self.time[0]
            t_max (float): time to stop averaging (non-inclusive)
                if None, defaults to self.time[-1]

        Args out:
            data (class 'numpy.ndarray'): time averaged data
        """
        
        # Set defaults
        if t_min is None:
            t_min = self.time[0]
        if t_max is None:
            t_max = self.time[-1]
            
        # Check for out of bounds
        if t_min < self.time[0]:
            raise ValueError(f'T_min ({t_min}) is less than the minimum time ({self.time[0]})')
        if t_max > self.time[-1]:
            raise ValueError(f'T_max ({t_max}) is greater than the maximum time ({self.time[-1]})')
            
        # Find time indices within time
        t_min_idx = np.argmax(self.time >= t_min)
        t_max_idx = np.argmax(self.time >= t_max)
        
        # Perform the average and return
        return np.mean(x[t_min_idx:t_max_idx],axis=0)

    def get_mean_wind_direction_at_heights(self, t_min=None, t_max=None):
        """ 
        Get the wind direction at every height over averaging window [t_min, t_max]

        Args in:
            t_min (float): time to start averaging (inclusive)
            t_max (float): time to stop averaging (non-inclusive)
        """

        u = self.get_data_from_mean_profiles('u')
        u_avg = self.time_average_data(u, t_min, t_max)

        v = self.get_data_from_mean_profiles('v')
        v_avg = self.time_average_data(v, t_min, t_max)

        self.wd_rad = np.arctan2(v_avg, u_avg) # Defined so 0 positive along x-axis
        self.wd_deg = (270.0 - np.degrees(self.wd_rad)) % 360. # Compass

    
    def plot_vertical_vel_profile(self, t_min=None, t_max=None, ax=None):
        """
        Plot the vertical velocity profile over an averaging
        period of [t_min, t_max]

        Args in:
            t_min (float): time to start averaging (inclusive)
            t_max (float): time to stop averaging (non-inclusive)
            ax (:py:class:'matplotlib.pyplot.axes', optional):
                figure axes. Defaults to None.
        """
        if ax is None:
            fig, ax = plt.subplots()
            
        u = self.get_data_from_mean_profiles('u')
        u_avg = self.time_average_data(u, t_min, t_max)
        
        ax.plot(u_avg, self.z)
        ax.set_xlabel("U m/s")
        ax.set_ylabel("Height [m]")
        xmax = (np.max(u_avg)+1)
        ax.set_xlim([0, xmax])
        ax.grid(True)
        
    def plot_vertical_temp_profile(self, t_min=None, t_max=None, ax=None):
        """
        Plot the vertical temperature profile over an averaging
        period of [t_min, t_max]

        Args in:
            t_min (float): time to start averaging (inclusive)
            t_max (float): time to stop averaging (non-inclusive)
            ax (:py:class:'matplotlib.pyplot.axes', optional):
                figure axes. Defaults to None.
        """
        if ax is None:
            fig, ax = plt.subplots()
            
        u = self.get_data_from_mean_profiles('theta')
        u_avg = self.time_average_data(u, t_min, t_max)
        
        ax.plot(u_avg, self.z)
        ax.set_xlabel("Temp (K)")
        ax.set_ylabel("Height [m]")
        # xmax = (np.max(u_avg)+1)
        # ax.set_xlim([0, xmax])
        ax.grid(True)

    def plot_wind_measurements_at_height(self, height, axarr=None, settling_time=None, label='_nolegend_'):
        """
        Plot ws, wd and the simple cartesian variables of varaiance u'u'_r, v'v'_r, w'w'_r and wind speed

        Args in:
            height (float): the height to extract, if not a value of self.z, will use nearest
            axarr (list?): an array of axis
            settling_time (float): An option value that indicates a proposed setting time

        Args out:
            axarr (list?): an array of axis
        """

        if axarr is None:
            fig, axarr = plt.subplots(5,1,figsize=(15,10), sharex=True)

        ax = axarr[0]
        data = self.get_wind_speed_time_series_at_height(height)
        ax.plot(self.time, data, label=label)
        ax.set_title('Wind Speed')
        ax.grid(True)
        if not label =='_nolegend_':
            ax.legend()

        ax = axarr[1]
        data = self.get_wind_direction_time_series_at_height(height)
        ax.plot(self.time, data)
        ax.set_title('Wind Direction')
        ax.grid(True)

        for sig, ax in zip(["u'u'_r","v'v'_r","w'w'_r"], axarr[2:]):

            data = self.get_time_series_at_height(sig, height)
            ax.plot(self.time, data)
            ax.set_title(sig)
            
            ax.grid(True)

        axarr[-1].set_xlabel('Time (s)')

        if settling_time is not None:
            for ax in axarr:
                ax.axvline(settling_time, color='r', ls='--')

    def get_time_series_at_height(self, variable, height):
        """
        Return the values of a variable within the mean_profiles for a specific height

        Args in:
            variable (str): the required variable to be extracted from group.
            height (float): the height to extract, if not a value of self.z, will use nearest

        Args out:
            data (class 'numpy.ndarray'): the requested variable
        """
        
        # Identify nearest height
        h_idx = np.argmin(np.abs(self.z - height))
        print(f'Nearest height to {height} is {self.z[h_idx]}')
        
        # Get the data
        x = self.get_data_from_mean_profiles(variable)
        
        # Return at height
        return np.squeeze(x[:,h_idx])
    

    def get_wind_speed_time_series_at_height(self, height):
        """
        Return the magnitude wind speed as a time series

        Args in:
            height (float): the height to extract, if not a value of self.z, will use nearest

        Args out:
            u (class 'numpy.ndarray'): wind speed magnitude over time
        """

        u = self.get_time_series_at_height('u', height)
        v = self.get_time_series_at_height('v', height)

        return np.sqrt(u**2 + v**2)

    def get_wind_direction_time_series_at_height(self, height):
        """
        Return the wind direction (compass) as a time series

        Args in:
            height (float): the height to extract, if not a value of self.z, will use nearest

        Args out:
            wd_deg (class 'numpy.ndarray'): compass wind direction in degrees
        """

        u = self.get_time_series_at_height('u', height)
        v = self.get_time_series_at_height('v', height)

        wd_rad = np.arctan2(v, u) # Defined so 0 positive along x-axis
        wd_deg = (270.0 - np.degrees(wd_rad)) % 360. # Compass

        return wd_deg

    def get_turbulence_intensity_at_height(self, height, t_min=None, t_max=None):

        """ 
        Get the turbulence intensity at prescribed height over averaging window [t_min, t_max]

        Args in:
            height (float): domain height (or nearest domain cell value to height) at which the TI is calculated
            t_min  (float): time to start averaging (inclusive)
            t_max  (float): time to stop averaging (non-inclusive)
        """

        u = self.get_time_series_at_height('u', height)
        u_avg = self.time_average_data(u, t_min, t_max)

        v = self.get_time_series_at_height('v', height)
        v_avg = self.time_average_data(v, t_min, t_max)

        w = self.get_time_series_at_height('w', height)
        w_avg = self.time_average_data(w, t_min, t_max)

        vel_mag_avg = np.sqrt(u_avg**2 + v_avg**2 + w_avg**2)

        uu = self.get_time_series_at_height("u'u'_r", height)
        uu_avg = self.time_average_data(uu, t_min, t_max)

        vv = self.get_time_series_at_height("v'v'_r", height)
        vv_avg = self.time_average_data(vv, t_min, t_max)

        ww = self.get_time_series_at_height("w'w'_r", height)
        ww_avg = self.time_average_data(ww, t_min, t_max)

        tke = 0.5*(uu_avg + vv_avg + ww_avg)
        TI = np.sqrt(2.0/3.0*tke)/vel_mag_avg


        return TI*100

    def plot_turbulence_intensity_profile(self, t_min=None, t_max=None, ax=None):

        """
        Plot the turbulence intensity profile over an averaging
        period of [t_min, t_max]

        Args in:
            t_min (float): time to start averaging (inclusive)
            t_max (float): time to stop averaging (non-inclusive)
            ax (:py:class:'matplotlib.pyplot.axes', optional):
                figure axes. Defaults to None.
        """
        if ax is None:
            fig, ax = plt.subplots()
            
        Nh = len(self.z)
        TI = np.zeros((Nh))

        for i in range(0,Nh):
            TI[i] = self.get_turbulence_intensity_at_height(self.z[i])

        ax.plot(TI, self.z)
        ax.set_xlabel("TI %")
        ax.set_ylabel("Height [m]")
        xmax = (np.max(TI)+1)
        ax.set_xlim([0, xmax])
        ax.grid(True)
            
        
