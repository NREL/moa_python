
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as ncdf


class Post_hor_plane:
    
    def __init__(self, filename):
        
        # Save the filename
        self.filename = filename

        # Load the data
        self.dataset = ncdf.Dataset(filename)
        
        # Read time
        self.time = np.array(self.dataset.variables['time'])
        self.num_time_steps = len(self.time)
        
        # Save the x and y dimensions
        self.x_N = self.dataset.groups['z_plane'].ijk_dims[0]
        self.y_N = self.dataset.groups['z_plane'].ijk_dims[1]
        self.x_max = self.dataset.groups['z_plane'].axis1[0]
        self.y_max = self.dataset.groups['z_plane'].axis2[1]
        self.x = np.linspace(0, self.x_max, self.x_N)
        self.y = np.linspace(0, self.y_max, self.y_N)

        # Save the number of planes 
        self.z = self.dataset.groups['z_plane'].offsets
        self.z_N = len(self.z)

        # Save the velocity planes
        self.vel_planes = dict()
        self.vel_planes['x'] = np.array(self.dataset.groups['z_plane'].variables['velocityx'])
        self.vel_planes['y']  = np.array(self.dataset.groups['z_plane'].variables['velocityy'])
        self.vel_planes['z'] = np.array(self.dataset.groups['z_plane'].variables['velocityz'])
        self.vel_planes['u'] = np.sqrt(self.vel_planes['x']**2 + self.vel_planes['y']**2)
        
        # Print a quick summar
        self.summary()
        
    def summary(self):
        """
        Print out a brief summary of the horizontal plane
        """
        
        print(f"Hor_plane has {self.z_N} horizontal planes in {self.num_time_steps} time steps from {self.time[0]} to {self.time[-1]}")
        print(f"Hor_plane levels: {self.z}")

    def get_height_index(self, height):
        """
        Return the nearest index to a height
        
        Args in:
            height (float): the height to find

        Args out:
            z_idx (float): nearest index
        """
        # Identify nearest height
        z_idx = np.argmin(np.abs(self.z - height))
        print(f'Nearest height to {height} is {self.z[z_idx]}')

        return z_idx

    def get_time_index(self, time):
        """
        Return the nearest time index to a time
        
        Args in:
            time (float): the time to find

        Args out:
            t_idx (float): nearest index
        """
        # Identify nearest height
        t_idx = np.argmin(np.abs(self.time - time))
        print(f'Nearest time to {time} is {self.time[t_idx]}')

        return t_idx


    def get_plane(self, height, time, component = 'u'):
        """
        Get a plane at particular height and time
        
        Args in:
            height (float): the height to get
            time (float): the time to get
            component (str): the component to get

        Args out:
            data (not sure): the horizontal plane
        """

        z_idx = self.get_height_index(height)
        t_idx = self.get_time_index(time)

        print(f"Returning {component} velocity hor_plane for height {self.z[z_idx]} at time {self.time[t_idx]}")

        return self.vel_planes[component][t_idx, :][z_idx*self.x_N*self.y_N:(z_idx+1)*self.x_N*self.y_N].reshape(self.x_N,self.y_N)

    def get_mean_plane(self, height, component = 'u'):
        """
        Get the mean plane at a particular height
        
        Args in:
            height (float): the height to get
            component (str): the component to get

        Args out:
            data (not sure): the horizontal plane
        """

        z_idx = self.get_height_index(height)

        print(f"Returning {component} mean velocity hor_plane for height {self.z[z_idx]}")

        mean_plane = np.mean(self.vel_planes[component], axis=0)
        return mean_plane[z_idx*self.x_N*self.y_N:(z_idx+1)*self.x_N*self.y_N].reshape(self.x_N,self.y_N)

    def plot_plane(self, height, time, component = 'u', ax = None, vmin=None, vmax=None):
        """
        Plot a horizontal plane at a particular height and time
        
        Args in:
            height (float): the height to plot
            time (float): the time to plot
            component (str): the component to plot
            ax (:py:class:'matplotlib.pyplot.axes', optional):
                figure axes. Defaults to None.
            vmin (float, optional) minimum value in colorbar
            vmax (float, optional) maximum value in colorbar
        """

        print(f"Plotting {component} velocity hor_plane for height {height} at time {time}")

        plane = self.get_plane(height, time, component = 'u')

        if ax is None:
            fig, ax = plt.subplots()
        im = ax.pcolor(self.x, self.y, plane, vmin=vmin, vmax=vmax)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')

        fig.colorbar(im, ax=ax)

    def plot_mean_plane(self, height, component = 'u', ax = None, vmin=None, vmax=None):
        """
        Plot the mean horizontal plane at a particular height and time
        
        Args in:
            height (float): the height to plot
            component (str): the component to plot
            ax (:py:class:'matplotlib.pyplot.axes', optional):
                figure axes. Defaults to None.
            vmin (float, optional) minimum value in colorbar
            vmax (float, optional) maximum value in colorbar
        """

        print(f"Plotting {component} mean velocity hor_plane for height {height}")

        plane = self.get_mean_plane(height, component = 'u')

        if ax is None:
            fig, ax = plt.subplots()
        im = ax.pcolor(self.x, self.y, plane, vmin=vmin, vmax=vmax)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')

        fig.colorbar(im, ax=ax)








        