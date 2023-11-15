
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as ncdf
from scipy.optimize import curve_fit

class Post_plane:
    """
    At the moment, this class only works for datasets containing one group of planes. 
    To do for future: make it compatible for different plane groups.
    """
    
    def __init__(self, filename, freq = 1, verbose = 1, origin = None, flip = False):
        
        # Save the filename
        self.filename = filename

        # Load the data
        self.dataset = ncdf.Dataset(filename)
        
        # Read time
        indices = np.arange(0,self.dataset.dimensions['num_time_steps'].size,freq)
        self.time = np.array(self.dataset.variables['time'][indices])
        self.num_time_steps = len(self.time)

        # Save the x and y dimensions
        self.plane = list(self.dataset.groups.keys())[0]
        self.x_N = self.dataset.groups[self.plane].ijk_dims[0]
        self.y_N = self.dataset.groups[self.plane].ijk_dims[1]
        self.x_max = np.sqrt(sum(entry**2 for entry in self.dataset.groups[self.plane].axis1))
        self.y_max = np.sqrt(sum(entry**2 for entry in self.dataset.groups[self.plane].axis2))
        self.x_dir = self.dataset.groups[self.plane].axis1/self.x_max
        self.y_dir = self.dataset.groups[self.plane].axis2/self.y_max
        self.z_dir = self.dataset.groups[self.plane].axis3
        self.x = np.linspace(0, self.x_max, self.x_N)
        self.y = np.linspace(0, self.y_max, self.y_N)
        self.unit = 'm'

        # Save the number of planes 
        self.z = self.get_plane_location(origin)
        if not isinstance(self.z,np.ndarray): self.z = np.array([self.z])
        self.z_N = len(self.z)

        # Save the velocity planes
        self.vel_planes = dict()
        self.vel_planes['x'] = np.array(self.dataset.groups[self.plane].variables['velocityx'][indices,:])
        self.vel_planes['y'] = np.array(self.dataset.groups[self.plane].variables['velocityy'][indices,:])
        self.vel_planes['z'] = np.array(self.dataset.groups[self.plane].variables['velocityz'][indices,:])
        self.vel_planes['u'] = np.sqrt(self.vel_planes['x']**2 + self.vel_planes['y']**2)
        
        # Print a quick summary
        if verbose: self.summary()
        
    def summary(self):
        """
        Print out a brief summary of the plane
        """
        
        print(f"Plane has {self.z_N} plane(s) in {self.num_time_steps} time steps from {self.time[0]} to {self.time[-1]}")
        print(f"Plane offsets: {self.z}")


    def get_plane_location(self,reference = None):
        """
        Return the location of a plane in the third dimension of the defined coordinate system.

        Args in:
            reference (list??): x, y and z-coordinates in AMR-Wind grid w.r.t. which to calculate plane location, default: [0, 0, 0]
        """

        if reference is None: 
            reference = 0
        else: 
            reference = sum((self.dataset.groups[self.plane].origin-reference)*self.z_dir)
        self.z = reference + self.dataset.groups[self.plane].offsets

        return self.z


    def get_plane_index(self, z, plane = 'z', verbose = 1):
        """
        Return the nearest index to a position
        
        Args in:
            z (float): the z location to find

        Args out:
            z_idx (float): nearest index
        """
        # Identify nearest z
        try:
            z_idx = []
            for zi in z:
                z_idx.append(np.argmin(np.abs(getattr(self,plane) - zi)))
        except:
            z_idx = np.argmin(np.abs(getattr(self,plane) - z))
        if verbose: print(f'Nearest point to {z} is {getattr(self,plane)[z_idx]}')

        return z_idx


    def get_time_index(self, time, verbose = 1):
        """
        Return the nearest time index to a time
        
        Args in:
            time (float): the time to find

        Args out:
            t_idx (float): nearest index
        """
        # Identify nearest time
        try:
            t_idx = []
            for t in time:
                t_idx.append(np.argmin(np.abs(self.time - t)))
        except:
            t_idx = np.argmin(np.abs(self.time - time))
        if verbose: print(f'Nearest time to {time} is {self.time[t_idx]}')

        return t_idx


    def get_plane(self, plane, time, component = 'u', verbose = 1):
        """
        Get a plane at particular location and time
        
        Args in:
            plane (float): the plane location to get
            time (float): the time to get
            component (str): the component to get (default: 'u')
            axes (str): the axes of the plane to get (default: 'xy')

        Args out:
            data (not sure): the plane
        """

        z_idx = self.get_plane_index(plane)
        t_idx = self.get_time_index(time)

        if verbose: print(f"Returning {component} velocity plane for slice at {self.z[z_idx]} at time {self.time[t_idx]}")

        return self.vel_planes[component][t_idx, :][z_idx*self.x_N*self.y_N:(z_idx+1)*self.x_N*self.y_N].reshape(self.y_N,self.x_N)


    def get_mean_plane(self, plane, component = 'u', timespan = None, verbose = 1):
        """
        Get the mean plane at a particular location
        
        Args in:
            plane (float): the plane location to get
            component (str): the component to get
            axes (str): the plane axes over which to average
            timespan (list, np.ndarray, tuple): [t0, tend], timespan over which to average

        Args out:
            data (not sure): the plane
        """

        z_idx = self.get_plane_index(plane, verbose = verbose)

        if verbose: print(f"Returning {component} mean velocity plane for slice at {self.z[z_idx]}")

        if timespan is None:
            timespan = [self.time[0], self.time[-1]]
        i0 = self.get_time_index(timespan[0], verbose = verbose)
        iend = self.get_time_index(timespan[1], verbose = verbose)
        mean_plane = np.mean(self.vel_planes[component][i0:iend,:], axis=0)
        return mean_plane[z_idx*self.x_N*self.y_N:(z_idx+1)*self.x_N*self.y_N].reshape(self.y_N,self.x_N)
    

    def get_line_from_plane(self, y, time = None, z = 0, axis = 'x', component = 'u', verbose = 1):
        """
        Outputs the velocity over a line
        Args in:
            y (float): position on the off-axis to take the line at
            axis (str): axis over which the line is moving (default: 'x')
            component (str): velocity component ('u', 'v', or 'w') to extract (default: 'u')
            plane (float): desired plane location (default: 0)
        
        Args out:
            line (array): velocity over the defined line
        """
        if len(np.shape(y)) == 0: y = [y]

        if axis == 'x': 
            idx_x = np.arange(0,self.x_N)
            idx_y = self.get_plane_index(y, axis, verbose)
        if axis == 'y':
            idx_x = self.get_plane_index(y, axis, verbose)
            idx_y = np.arange(0,self.y_N)
        
        idx_z = self.get_plane_index(z, verbose = verbose)
        if time is None:
            t_idx = np.arange(0, self.num_time_steps)
        else:
            if np.size(time) == 1: t_idx = [self.get_time_index(time, verbose = verbose)]
            else: t_idx = np.arange(self.get_time_index(time[0], verbose = verbose), self.get_time_index(time[-1], verbose = verbose))

        return np.squeeze(self.vel_planes[component][t_idx, idx_z*self.x_N*self.y_N:(idx_z+1)*self.x_N*self.y_N]\
                    .reshape(np.size(t_idx),self.y_N,self.x_N)[np.ix_(np.arange(np.size(t_idx)),idx_y,idx_x)])


    def mean_vel_in_circle(self, origin, radius, z = None, time = None, component = 'u', verbose = 1):
        """
        Outputs the mean velocity over an area of the flow field
        Args in:
            origin (array): origin location of the area, in format [x0, y0]
            x (float): 
        """

        if time is None:
            time = self.time
            t_idx = np.arange(0, self.num_time_steps)
        elif np.size(time) == 1: 
            t_idx = np.arange(self.get_time_index(time, verbose = verbose), self.num_time_steps)
        else: t_idx = np.arange(self.get_time_index(time[0], verbose = verbose), self.get_time_index(time[-1], verbose = verbose))

        if z is None: z_idx = 0
        else: z_idx = self.get_plane_index(z, verbose = verbose)

        x_coor, y_coor = np.meshgrid(self.x, self.y)
        idx = np.squeeze(np.where((np.reshape(x_coor,-1)-origin[0])**2 + (np.reshape(y_coor,-1)-origin[1])**2 < radius**2))
 
        return np.average(self.vel_planes[component][np.ix_(t_idx, z_idx*self.x_N*self.y_N+idx)], axis=1)


    def get_mean_line_from_plane(self, y, timespan = None, z = 0, axis = 'x', component = 'u', verbose = 1):
        """
        Outputs the mean velocity over a line
        Args in:
            y (float): position on the off-axis to take the line at
            axis (str): axis over which the line is moving (default: 'x')
            component (str): velocity component ('u', 'v', or 'w') to extract (default: 'u')
            plane (float): desired plane location (default: 0)
        
        Args out:
            line (array): velocity over the defined line
        """

        if timespan is None:
            timespan = [self.time[0], self.time[-1]]
        elif np.size(timespan) == 1:
            timespan = [timespan, self.time[-1]]

        line = self.get_line_from_plane(y, timespan, z, axis, component, verbose)

        return np.mean(line,axis=0)


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
                'amr-wind': AMR-Wind coordinate frame (for horizontal planes)
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
        elif kwargs['frame'] == 'amr-wind':
            if not 'x' or not 'y' in kwargs.keys():
                raise KeyError('For frame amr-wind, both x and y coordinates of desired origin must be given')
            if self.unit == 'D':
                print('WARNING: "set_origin" is meant to be used before using "scale_to_rot_diam". Expect undesired results!')
            if not hasattr(self,'amr_origin'): self.amr_origin = self.dataset.groups[self.plane].origin[0:2]
            shift_in_origin = np.array([kwargs['x'], kwargs['y']]) - self.amr_origin
            self.amr_origin = np.array([kwargs['x'], kwargs['y']])
            normal = self.dataset.groups[self.plane].axis1/np.sqrt(self.dataset.groups[self.plane].axis1.dot(self.dataset.groups[self.plane].axis1))
            self.x = self.x - (normal[0]*shift_in_origin[0]+normal[1]*shift_in_origin[1])
            self.y = self.y - (normal[0]*shift_in_origin[1]-normal[1]*shift_in_origin[0])


    def scale_to_rot_diam(self,rot_diam):
        """
        Scales all axes to rotor diameter. 

        Args in:
            rot_diam: rotor diameter in m
        """

        if self.unit == 'm':
            self.x = self.x/rot_diam
            self.y = self.y/rot_diam
            self.z = self.z/rot_diam
            self.unit = 'D'
        else:
            print('WARNING: Field already scaled to rotor diameter. Nothing happened.')


    def plot_plane(self, z, time, component = 'u', ax = None, vmin = None, vmax = None, verbose = 1):
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

        if verbose: print(f"Plotting {component} velocity for plane at location {z} at time {time}")

        plane = self.get_plane(z,time,component, verbose = verbose)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            plt.sca(ax)
            fig = plt.gcf()
        im = ax.pcolor(self.x, self.y, plane, vmin=vmin, vmax=vmax)
        ax.set_xlabel(f'X [{self.unit}]')
        ax.set_ylabel(f'Y [{self.unit}]')
        ax.set_aspect('equal')

        fig.colorbar(im,ax=ax,location='bottom')

        return ax


    def plot_mean_plane(self, z, component = 'u', fig = None, ax = None, vmin=None, vmax=None, timespan=None, verbose = 1):
        """
        Plot the mean plane at a particular height and time
        
        Args in:
            z (float): the z-coordinate of the plane to plot
            component (str): the component to plot
            ax (:py:class:'matplotlib.pyplot.axes', optional):
                figure axis. Defaults to None.
            vmin (float, optional) minimum value in colorbar
            vmax (float, optional) maximum value in colorbar
        """

        if verbose: print(f"Plotting {component} mean velocity for plane at location {z}")

        plane = self.get_mean_plane(z, component, timespan = timespan, verbose = verbose)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            plt.sca(ax)
            fig = plt.gcf()
        im = ax.pcolor(self.x, self.y, plane, vmin=vmin, vmax=vmax)
        ax.set_xlabel(f'X [{self.unit}]')
        ax.set_ylabel(f'Y [{self.unit}]')
        ax.set_aspect('equal')

        fig.colorbar(im, ax=ax,location='bottom')

        return ax


    def plot_line(self, y, time, z = 0, axis = 'x', component = 'u', ax = None, verbose = 1):
        """
        Plot a line at a particular slice, position and time
        
        Args in:
            z (float): the z-coordinate of the plane to plot
            time (float): the time to plot
            component (str): the component to plot
            ax (:py:class:'matplotlib.pyplot.axes', optional):
                figure axes. Defaults to None.
            vmin (float, optional) minimum value in colorbar
            vmax (float, optional) maximum value in colorbar
        """

        if verbose: print(f"Plotting {component} velocity for plane at location ({y}, {z}) at time {time}")

        line = self.get_line_from_plane(y, time, z, axis, component, verbose)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            plt.sca(ax)
            fig = plt.gcf()
        im = ax.plot(getattr(self, axis), line)
        ax.set_xlabel(f'X [{self.unit}]')
        ax.set_ylabel(f'Y [{self.unit}]')
        ax.grid(True)

        return ax


    def plot_mean_line(self, y, timespan = None, z = 0, axis = 'x', component = 'u', ax = None, verbose = 1):
        """
        Plot a line at a particular slice, position and time
        
        Args in:
            z (float): the z-coordinate of the plane to plot
            time (float): the time to plot
            component (str): the component to plot
            ax (:py:class:'matplotlib.pyplot.axes', optional):
                figure axes. Defaults to None.
            vmin (float, optional) minimum value in colorbar
            vmax (float, optional) maximum value in colorbar
        """

        if verbose: print(f"Plotting {component} velocity for plane at location ({y}, {z}) at time {timespan}")

        line = self.get_mean_line_from_plane(y, timespan, z, axis, component, verbose)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            plt.sca(ax)
            fig = plt.gcf()
        im = ax.plot(getattr(self, axis), line)
        ax.set_xlabel(f'X [{self.unit}]')
        ax.set_ylabel(f'Y [{self.unit}]')
        ax.grid(True)

        return ax

    
    def plot_turbine(self,hub_height=150/240,rot_diam=1,turb_loc=[0,0],ax=None,plane='xy'):
        """
        Plot the turbine location in the flow field

        Args in:
            hub_height (float): turbine hub height, defaults to 240/150
            rot_diam (float): rotor diameter, defaults to 1
            turb_loc (list, tuple, np.ndarray): x and y turbine location in flow field
                Defaults to [0, 0]
            ax (:py:class:'matplotlib.pyplot.axes', optional): 
                figure axis. Defaults to None.
            plane (string): plane in which to plot turbine ('xy', 'yz', or 'xz')
        """

        if ax is None: ax=plt.gca()

        if plane == 'yz':
            ax.plot(rot_diam/2*np.sin(np.linspace(0, 2*np.pi,200)),\
                    hub_height+rot_diam/2*np.cos(np.linspace(0, 2*np.pi,200)),'k-',linewidth=1)
        elif plane == 'xy':
            ax.plot([turb_loc[0],turb_loc[0]],\
                    [turb_loc[1]-rot_diam/2,turb_loc[1]+rot_diam/2],'k',linewidth=1.5)
        elif plane == 'xz':
            ax.plot([turb_loc[0],turb_loc[0]],\
                    hub_height+[turb_loc[1]-rot_diam/2,turb_loc[1]+rot_diam/2],'k',linewidth=1.5)


    def vel_in_wake(self, radius, turb_loc = None, z = None, time = None, axis = 'x', component = 'u', verbose = 1):
        """
        Calculates velocity in the wake of a turbine
        
        Args in:
            hub_heigth (float): turbine hub height in m
            rot_diam (float): turbine rotor diameter in m
            turb_loc (list): x- and y- location of the turbine in m (default: [0, 0])
            dir (str): direction of the plane, 
                    either 'x' (default), 'y', (streamwise in x- or y-direction, respectively) or 'z' (i.e., cut-through of the flow)
        
        Args out:
            Utube (np.array): average wind speed in wake (dimensions: squeeze(num_time_steps, num_x_coor, num_cases) )
        """

        if turb_loc is None: turb_loc = [np.average(self.x), np.average(self.y), self.z[0]]
        if z is None: z = self.z[0]

        if axis == 'x':
            xyrange = self.y[np.where((self.y-turb_loc[1])**2 + (self.z-z)**2 < radius**2)]
            return np.average(self.get_line_from_plane(xyrange, time, z, axis, component, verbose), axis=1)
        elif axis == 'y':
            xyrange = self.x[np.where((self.x-turb_loc[0])**2 + (self.z-z)**2 < radius**2)]
            return np.average(self.get_line_from_plane(xyrange, time, z, axis, component, verbose), axis=1)
        else:
            return self.mean_vel_in_circle(turb_loc, radius, z, time, component, verbose)


    def mean_vel_in_wake(self, radius, turb_loc = None, z = None, timespan = None, axis = 'x', component = 'u', verbose = 1):
        """
        Calculates average velocity in the wake of a turbine
        
        Args in:
            hub_heigth (float): turbine hub height in m
            rot_diam (float): turbine rotor diameter in m
            turb_loc (list): x- and y- location of the turbine in m (default: [0, 0])
            dir (str): direction of the plane, either 'streamwise' (default) or 'slice' (i.e., cut-through of the flow)
        
        Args out:
            Utube (np.array): average wind speed in wake (dimensions: squeeze(num_time_steps, num_x_coor, num_cases) )
        """

        if z is None: z = self.z 
        if np.size(z) > 1:
            mean_vel = []
            for zi in z:
                mean_vel.append(np.average(self.vel_in_wake(radius, turb_loc, zi, timespan, axis, component, verbose),axis=0))
        else: 
            mean_vel = np.average(self.vel_in_wake(radius, turb_loc, z, timespan, axis, component, verbose),axis=0)

        return mean_vel


    def plot_vel_in_wake(self, radius, turb_loc = [0,0,0], z = None, timespan = None, axis = 'x', component = 'u', ax = None, linestyle = '-', verbose = 0):
        """
        Plots average velocity in the wake using mean_vel_in_wake

        Additional arg in:
            ax: axis to plot on, default: None (creates new figure)
        """

        line = self.mean_vel_in_wake(radius, turb_loc, z, timespan, axis, component, verbose)

        if verbose: print(f"Plotting average {component} wake velocity for plane ({axis})")

        if ax is None:
            fig, ax = plt.subplots()
        else:
            plt.sca(ax)
            fig = plt.gcf()
        im = ax.plot(getattr(self, axis)-turb_loc['xyz'.find(axis)], line, linestyle)
        ax.set_xlabel(f'X [{self.unit}]')
        ax.set_ylabel(f'Wind speed [m/s]')
        ax.grid(True)

        return ax
    

    def get_vorticity(self, plane, time, orientation = 'xy'):
        """
        Calculates vorticity in plane at given time

        Args in:
            plane (float): plane location
            time (float): time to calculate vorticity at
            orientation (str): orientation of the plane in xyz coordinates, default: xy
        """

        u = self.get_plane(plane, time, component = orientation[0])
        v = self.get_plane(plane, time, component = orientation[1])
        
        return (np.diff(v,axis=1)/np.diff(self.x))[0:-1,:] - (np.diff(u,axis=0).T/np.diff(self.y)).T[:,0:-1]


    def periodic_averaging(self, signal, num_bins, period, poi=None, amplitude=None, offset=0):
        """
        
        """

        
        if amplitude is not None:
            bins = np.linspace(-amplitude+offset, amplitude+offset, num_bins)[1:-1]     
            bin_indices = np.digitize(signal,bins)
        else:
            bins = np.linspace(min(signal), max(signal),num_bins)[1:-1]
            bin_indices = np.digitize(signal,bins)

        return signal_in_bins, bins


    def plot_vorticity(self, plane, time, orientation='xy', ax=None, vmin=None, vmax=None, verbose=1):
        """
        Plot vorticity over a plane at a particular slice and time
        
        Args in:
            plane (float): the z-coordinate of the plane to plot
            time (float): the time to plot
            orientation (str): orientation of the plane in xyz coordinates, default: xy
            ax (:py:class:'matplotlib.pyplot.axes', optional):
                figure axes. Defaults to None.
            vmin (float, optional) minimum value in colorbar
            vmax (float, optional) maximum value in colorbar
        """

        if verbose: print(f"Plotting vorticity for plane at location {plane} at time {time}")

        vorticity = self.get_vorticity(plane,time,orientation)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            plt.sca(ax)
            fig = plt.gcf()
        im = ax.pcolor(self.x[0:-1], self.y[0:-1], vorticity, vmin=vmin, vmax=vmax)
        ax.set_xlabel(f'X [{self.unit}]')
        ax.set_ylabel(f'Y [{self.unit}]')
        ax.set_aspect('equal')

        fig.colorbar(im,ax=ax,location='bottom')

        return ax


    def fit_gauss_to_wake_profile(self, x, y = None, p0 = None, fit = 'double', time = None, z = 0, axis = 'y', component = 'u', ax = None, verbose = 0):
        """
        Fit a Gauss curve to signal y:
            ym = v0 - a*exp(-(x-x0))^2 / (2*sigma^2)

        Args in:
            x (array-like): x vector
            y (array-like): signal to fit the Gauss curve to
            p0 (array-like): initial guess for the fit, in shape [v0, a, x0, sigma]
                                (defaults to standard wake values as intial guess)
            fit (str): type of Gaussian fit ('single' or 'double' (default))
            ...
            ax (mpl.axis or bool): if defined, plots the signal and the fit

        Args out: 
            ym (array): the best Gaussian fit to y
            popt (array): optimal parameters for p
        """

        if time is None: time = self.time[-1]
        if y is None:
            y = x
            x = getattr(self, axis)
        if np.size(y) == 1: 
            y = self.get_line_from_plane(y, time, z, axis, component, verbose)

        if p0 is None:
            v0 = np.max(self.vel_planes['x'])
            a = np.max(self.vel_planes['x']) - np.min(self.vel_planes['x'])
            x0 = np.average(x)
            if self.unit == 'm': sigma = 120
            else: sigma = 1
            if fit == 'double': p0 = [v0, a, x0, sigma/2, sigma/2]
            else: p0 = [v0, a, x0, sigma]

        try: 
            popt, pcov = curve_fit(gauss_func, x, y, p0)
        except:
            print(f'Warning! Optimal parameters not found at time {time}. Reverted to initial guess.')
            popt = p0

        if ax:
            ax = self.plot_gauss_fit(x, y, popt, ax)

        return popt


    def fit_gauss_to_mean_wake_profile(self, x, y = None, p0 = None, fit = 'double', timespan = None, z = 0, axis = 'y', component = 'u', ax = None, verbose = 0):
        """
        Fit a Gauss curve to signal y:
            ym = v0 - a*exp(-(x-x0))^2 / (2*sigma^2)

        Args in:
            x (array-like): x position vector (optional, if not defined, x = self.y)
            y (array-like): wind speed signal to fit the Gauss curve to
            p0 (array-like): initial guess for the fit, in shape [v0, a, x0, sigma]
                                (defaults to standard wake values as intial guess)
            fit (str): type of Gaussian fit ('single' or 'double' (default))

        Args out: 
            ym (array): the best Gaussian fit to y
            popt (array): optimal parameters for p
        """

        if y is None:
            y = x
            x = getattr(self, axis)
        
        if len(np.shape(y)) > 1:
            y = np.mean(y, axis=0)
        elif np.size(y) == 1:
            y = self.get_mean_line_from_plane(y, timespan, z, axis, component, verbose)
        
        if p0 is None:
            v0 = np.max(self.vel_planes['x'])
            a = np.max(self.vel_planes['x']) - np.min(self.vel_planes['x'])
            x0 = np.average(x)
            if self.unit == 'm': sigma = 120
            else: sigma = 1
            if fit == 'double': p0 = [v0, a, x0, sigma/2, sigma/2]
            else: p0 = [v0, a, x0, sigma]

        try: 
            popt, pcov = curve_fit(gauss_func, x, y, p0)
        except:
            print('Warning! Optimal parameters not found. Reverted to initial guess.')
            popt = p0
        
        if ax:
            ax = self.plot_gauss_fit(x, y, popt, ax)

        return popt


    def plot_gauss_fit(self, x, y, popt, ax = None):
        """
        Plots the wake profile and Gaussian fit
        """

        if not hasattr(ax, 'plot'): fig, ax = plt.subplots()

        ax.plot(x, y)
        ax.plot(x, gauss_func(x,*popt))
        ax.grid()
        ylim = ax.get_ylim()
        ax.plot([popt[2], popt[2]],ylim,'--', color='0.7')
        ax.set_ylim(ylim)
        ax.set_xlabel(f'Position [{self.unit}]')
        ax.set_ylabel(f'Wind speed [m/s]')
        ax.set_title('Wake profile')

        return ax


    def fit_gauss_to_time_series(self, x, y = None, p0 = None, fit = 'double', time = None, z = 0, axis = 'y', component = 'u', verbose = 0):
        """
        Fit a Gauss curve to signal y over a time series. 
        Returns optimal fit parameters popt as function of time
        """

        if time is None: time = self.time
        elif np.size(time) == 1: time = self.time[self.time >= time]
        elif np.size(time) == 2: time = self.time[np.where((self.time >= time[0]) & (self.time <= time[1]))]
        if p0 is None:
            v0 = np.average(self.vel_planes['x'])
            a = 2/3*v0
            if y is None: x0 = np.average(getattr(self, axis))            
            else: x0 = np.average(x)
            if not hasattr(self,'unit') or self.unit == 'm':
                sigma = 120
            else:
                sigma = 1
            if fit == 'double': p0 = [v0, a, x0, sigma, sigma]
            else: p0 = [v0, a, x0, sigma]

        popt = p0
        parr = []
        for n in range(len(time)):
            popt = self.fit_gauss_to_wake_profile(x, y, popt, fit, time[n], z, axis, component, verbose)
            parr.append(list(popt))

        return np.array(parr)


    def track_wake_center_over_time(self, x, y = None, p0 = None, fit = 'double', timespan = None, z = 0, axis = 'y', component = 'u', ax = None, verbose = 0):
        """
        Tracks the wake centerline over time,
        based on the gaussian fit obtained with fit_gauss_to_time_series.
        """

        if timespan is None: timespan = self.time
        elif np.size(timespan) == 1: timespan = self.time[self.time >= timespan]
        elif np.size(timespan) == 2: timespan = self.time[np.where((self.time >= timespan[0]) & (self.time <= timespan[1]))]
        popt = self.fit_gauss_to_time_series(x, y, p0, fit, timespan, z, axis, component, verbose)

        if ax:
            if not hasattr(ax, 'plot'): fig, ax = plt.subplots()
            ax.plot(timespan, popt[:,2])
            ax.grid()
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(f'Location [{self.unit}]')
            ax.set_title('Wake centerline position')

            return ax
        else:
            return popt[:,2]

######################

def fit_sin(t, y, p0 = None):
    t = np.array(t)
    y = np.array(y)
    ff = np.fft.fftfreq(len(t), (t[1]-t[0]))
    Fyy = abs(np.fft.fft(y))
    if p0 is None:
        guess_freq = abs(ff[np.argmax(Fyy[1:])+1])
        guess_amp = np.std(y) * 2.**0.5
        guess_offset = np.mean(y)
        p0 = np.array([guess_amp, guess_freq, 0., guess_offset])

    def sin_fun(t,A,f,ph,offset):
        return A * np.sin(2*np.pi*f*t+ph) + offset
    
    popt, pcov = curve_fit(sin_fun,t,y,p0)
    A,f,ph,offset = popt
    print(f'\
    Amplitude:    {A}\n\
    frequency:    {f}\n\
    offset:       {offset}')
    curve = sin_fun(t,*popt)
    
    return popt, curve


def gauss_func(x, v0, a, x0, sigma, w = None):
    """
    Simple or double gaussian function:
    Single:
        y = v0 - a*exp(-(x-x0))^2 / (2*sigma^2)
    Double:
        v0 - a*exp(-(x-x0-w))^2 / (2*sigma^2) - a*exp(-(x-x0+w))^2 / (2*sigma^2)

    Args in:
        x (array-like): x vector
        p (array-like): in shape [v0, a, x0, sigma, w]
            if w is not provided, single gaussian fit is applied

    Args out: 
        y (array): Gauss curve
    """
    
    if w is None:
        return v0 - a*np.exp(-(x-x0)**2/(2*sigma**2))
    else: 
        return v0 - a*np.exp(-(x-x0-w)**2/(2*sigma**2)) - a*np.exp(-(x-x0+w)**2/(2*sigma**2))


# def gauss_func_2d(x, y, )





        