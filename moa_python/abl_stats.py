
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as ncdf


class Abl_stats:
    
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
    
    def plot_vert_vel_profile(self, t_min=None, t_max=None, ax=None):
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
    
        
        

    
        