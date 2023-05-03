
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as ncdf


class Post_plane:
    """
    At the moment, this class only works for datasets containing one group of planes. 
    To do for future: make it compatible for different plane groups.
    """
    
    def __init__(self, filename):
        
        # Save the filename
        self.filename = filename

        # Load the data
        self.dataset = ncdf.Dataset(filename)
        
        # Read time
        self.time = np.array(self.dataset.variables['time'])
        self.num_time_steps = len(self.time)
        
        # Save the x and y dimensions
        self.plane = list(self.dataset.groups.keys())[0]
        self.x_N = self.dataset.groups[self.plane].ijk_dims[0]
        self.y_N = self.dataset.groups[self.plane].ijk_dims[1]
        self.x_max = np.sqrt(sum(entry**2 for entry in self.dataset.groups[self.plane].axis1))
        self.y_max = np.sqrt(sum(entry**2 for entry in self.dataset.groups[self.plane].axis2))
        self.x = np.linspace(0, self.x_max, self.x_N)
        self.y = np.linspace(0, self.y_max, self.y_N)

        # Save the number of planes 
        self.z = self.dataset.groups[self.plane].offsets
        if not isinstance(self.z,np.ndarray): self.z = np.array([self.z])
        self.z_N = len(self.z)

        # Save the velocity planes
        self.vel_planes = dict()
        self.vel_planes['x'] = np.array(self.dataset.groups[self.plane].variables['velocityx'])
        self.vel_planes['y'] = np.array(self.dataset.groups[self.plane].variables['velocityy'])
        self.vel_planes['z'] = np.array(self.dataset.groups[self.plane].variables['velocityz'])
        self.vel_planes['u'] = np.sqrt(self.vel_planes['x']**2 + self.vel_planes['y']**2)
        
        # Print a quick summary
        self.summary()
        
    def summary(self):
        """
        Print out a brief summary of the plane
        """
        
        print(f"Plane has {self.z_N} plane(s) in {self.num_time_steps} time steps from {self.time[0]} to {self.time[-1]}")
        print(f"Plane levels: {self.z}")

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
            component (str): the component to get (default: 'u')
            axes (str): the axes of the plane to get (default: 'xy')

        Args out:
            data (not sure): the plane
        """

        z_idx = self.get_height_index(height)
        t_idx = self.get_time_index(time)

        print(f"Returning {component} velocity plane for slice at {self.z[z_idx]} at time {self.time[t_idx]}")

        return self.vel_planes[component][t_idx, :][z_idx*self.x_N*self.y_N:(z_idx+1)*self.x_N*self.y_N].reshape(self.y_N,self.x_N)

    def get_mean_plane(self, height, component = 'u', axes = 'xy'):
        """
        Get the mean plane at a particular height
        
        Args in:
            height (float): the height to get
            component (str): the component to get

        Args out:
            data (not sure): the plane
        """

        z_idx = self.get_height_index(height)
        x = getattr(self,axes[1]+'_N')
        y = getattr(self,axes[0]+'_N')

        print(f"Returning {component} mean velocity plane for slice at {self.z[z_idx]}")

        mean_plane = np.mean(self.vel_planes[component], axis=0)
        return mean_plane[z_idx*self.x_N*self.y_N:(z_idx+1)*self.x_N*self.y_N].reshape(self.y_N,self.x_N)
    
    def set_origin(self, **kwargs):
        """
        Set plane origin to something else than left bottom corner. 
        Defaults to center around x-axis.

        Args in:
            center (str) (default):
                'x': center x-axis
                'y': center y-axis
                'xy': center both axes
            x (float): x coordinate of origin
            y (float): y coordinate of origin
            frame (str): 
                'python': relative to left bottom corner (default)
                'amr-wind': AMR-Wind coordinate frame
        """

        if not 'kwargs' in locals():
            self.x = self.x - (np.max(self.x)-np.min(self.x))/2
        elif 'center' in kwargs.keys():
            if 'x' in kwargs['center']:
                self.x = self.x - np.min(self.x) - (np.max(self.x)-np.min(self.x))/2
            if 'y' in kwargs['center']:
                self.y = self.y - np.min(self.y) - (np.max(self.y)-np.min(self.y))/2
        elif not 'frame' in kwargs.keys() or kwargs['frame'] == 'python':
            if 'frame' in kwargs.keys(): kwargs.pop('frame')
            for key,value in kwargs.items():
                setattr(self, key, getattr(self,key)-np.min(getattr(self,key))-value)
        elif 'frame' in kwargs.keys() and kwargs['frame'] == 'amr-wind':
            if not 'x' or not 'y' in kwargs.keys():
                raise KeyError('For frame amr-wind, both x and y coordinates of desired origin must be given')
            origin = self.dataset.groups[self.plane].origin
            angle_field = np.arctan(self.dataset.groups[self.plane].axis1[1]/self.dataset.groups[self.plane].axis1[0])
            x_dist = kwargs['x']-origin[0]
            y_dist = kwargs['y']-origin[1]
            angle = np.arctan(y_dist/x_dist)
            dist = np.sqrt(x_dist**2 + y_dist**2)
            new_origin = [dist*np.cos(angle-angle_field)+np.min(self.x), dist*np.sin(angle-angle_field)+np.min(self.y)]
            self.x = self.x - new_origin[0]
            self.y = self.y - new_origin[1]

    def plot_plane(self, z, time, component = 'u', ax = None, vmin=None, vmax=None):
        """
        Plot a plane at a particular slice and time
        
        Args in:
            z (float): the z-coordinate of the plane to plot
            time (float): the time to plot
            component (str): the component to plot
            ax (:py:class:'matplotlib.pyplot.axes', optional):
                figure axes. Defaults to None.
            vmin (float, optional) minimum value in colorbar
            vmax (float, optional) maximum value in colorbar
        """

        print(f"Plotting {component} velocity for plane at location {z} at time {time}")

        plane = self.get_plane(z,time,component)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            plt.sca(ax)
            fig = plt.gcf()
        im = ax.pcolor(self.x, self.y, plane, vmin=vmin, vmax=vmax)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')

        fig.colorbar(im,ax=ax,location='bottom')

        return ax

    def plot_mean_plane(self, z, component = 'u', fig = None, ax = None, vmin=None, vmax=None):
        """
        Plot the mean plane at a particular height and time
        
        Args in:
            z (float): the z-coordinate of the plane to plot
            component (str): the component to plot
            ax (:py:class:'matplotlib.pyplot.axes', optional):
                figure axes. Defaults to None.
            vmin (float, optional) minimum value in colorbar
            vmax (float, optional) maximum value in colorbar
        """

        print(f"Plotting {component} mean velocity for plane at location {z}")

        plane = self.get_mean_plane(z, component)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            plt.sca(ax)
            fig = plt.gcf()
        im = ax.pcolor(self.x, self.y, plane, vmin=vmin, vmax=vmax)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')

        fig.colorbar(im, ax=ax,location='bottom')

        return ax








        