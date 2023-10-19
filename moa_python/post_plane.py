
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as ncdf
from scipy.optimize import curve_fit

class Post_plane:
    """
    At the moment, this class only works for datasets containing one group of planes. 
    To do for future: make it compatible for different plane groups.
    """
    
    def __init__(self, filename, freq = 1, verbose = 1, amr_origin = [0,0,0], flip = False):
        
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

        # Save the number of planes 
        self.z = self.get_plane_location(amr_origin)
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

    def get_plane_location(self,reference = [0,0,0]):
        """
        Return the location of a plane in the third dimension of the defined coordinate system.

        Args in:
            reference (list??): x, y and z-coordinates in AMR-Wind grid w.r.t. which to calculate plane location, default: [0, 0, 0]
        """

        if not isinstance(reference,list): reference = list(reference)
        origin = sum((self.dataset.groups[self.plane].origin-reference)*self.z_dir)
        self.z = origin + self.dataset.groups[self.plane].offsets

        return self.z

    def get_plane_index(self, z, verbose = 1):
        """
        Return the nearest index to a position
        
        Args in:
            z (float): the z location to find

        Args out:
            z_idx (float): nearest index
        """
        # Identify nearest z
        z_idx = np.argmin(np.abs(self.z - z))
        if verbose: print(f'Nearest plane to {z} is {self.z[z_idx]}')

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

        z_idx = self.get_plane_index(plane)

        if verbose: print(f"Returning {component} mean velocity plane for slice at {self.z[z_idx]}")

        if timespan is None:
            timespan = [self.time[0], self.time[-1]]
        i0 = self.get_time_index(timespan[0])
        iend = self.get_time_index(timespan[1])
        mean_plane = np.mean(self.vel_planes[component][i0:iend,:], axis=0)
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
            if hasattr(self,'unit') and self.unit == 'D':
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

        if not hasattr(self,'unit') or self.unit == 'm':
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

        plane = self.get_plane(z,time,component)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            print(ax)
            plt.sca(ax)
            fig = plt.gcf()
        im = ax.pcolor(self.x, self.y, plane, vmin=vmin, vmax=vmax)
        if not hasattr(self,'unit'): self.unit = 'm'
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

        plane = self.get_mean_plane(z, component, timespan=timespan)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            plt.sca(ax)
            fig = plt.gcf()
        im = ax.pcolor(self.x, self.y, plane, vmin=vmin, vmax=vmax)
        if not hasattr(self,'unit'): self.unit = 'm'
        ax.set_xlabel(f'X [{self.unit}]')
        ax.set_ylabel(f'Y [{self.unit}]')
        ax.set_aspect('equal')

        fig.colorbar(im, ax=ax,location='bottom')

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

    def vel_in_wake(self,hub_height,rot_diam,turb_loc=[0, 0],timespan=None):
        """
        Calculates average velocity up/downstream of a turbine
        
        Args in:
            hub_heigth (float): turbine hub height in m (default: 90)
            rot_diam (float): turbine rotor diameter in m (default: 126)
            turb_loc (list): x- and y- location of the turbine in m (default: [0, 0])
        
        Args out:
            Utube (np.array): average wind speed in wake (dimensions: squeeze(num_time_steps, num_x_coor, num_cases) )
        """
        idx = np.where((self.y-turb_loc[1])**2 + (self.z-hub_height)**2 < (rot_diam/2)**2)
        self.Uwake = self.get_mean_plane(hub_height, timespan=timespan)
        self.Uwake_avg = np.average(self.Uwake[idx],axis=0)
                
        return self.Uwake_avg


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


    def fit_sin(t,y):
        t = np.array(t)
        y = np.array(y)
        ff = np.fft.fftfreq(len(t), (t[1]-t[0]))
        Fyy = abs(np.fft.fft(y))
        guess_freq = abs(ff[np.argmax(Fyy[1:])+1])
        guess_amp = np.std(y) * 2.**0.5
        guess_offset = np.mean(y)
        guess = np.array([guess_amp, guess_freq, 0., guess_offset])

        def sin_fun(t,A,f,ph,offset):
            return A * np.sin(2*np.pi*f*t+ph) + offset
        
        popt, pcov = curve_fit(sin_fun,t,y,p0=guess)
        A,f,ph,offset = popt
        print(f'\
        Amplitude:    {A}\n\
        offset:       {offset}')
        curve = sin_fun(t,*popt)
        
        return popt, curve


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
        if not hasattr(self,'unit'): self.unit = 'm'
        ax.set_xlabel(f'X [{self.unit}]')
        ax.set_ylabel(f'Y [{self.unit}]')
        ax.set_aspect('equal')

        fig.colorbar(im,ax=ax,location='bottom')

        return ax











        