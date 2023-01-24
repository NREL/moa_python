
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
        
        print(f"Hor_plane has {self.z_N} horizontal planes in {self.num_time_steps} time steps from {self.time[0]} to {self.time[-1]}")
        print(f"Hor_plane levels: {self.z}")


    def get_plane(self, z_idx, t_idx, component = 'u'):

        print(f"Returning {component} velocity hor_plane for height {self.z[z_idx]} at time {self.time[t_idx]}")

        return self.vel_planes[component][t_idx, :][z_idx*self.x_N*self.y_N:(z_idx+1)*self.x_N*self.y_N].reshape(self.x_N,self.y_N)

    def get_mean_plane(self, z_idx, component = 'u'):

        print(f"Returning {component} mean velocity hor_plane for height {self.z[z_idx]}")

        mean_plane = np.mean(self.vel_planes[component], axis=0)
        return mean_plane[z_idx*self.x_N*self.y_N:(z_idx+1)*self.x_N*self.y_N].reshape(self.x_N,self.y_N)

    def plot_plane(self, z_idx, t_idx, component = 'u', ax = None):

        print(f"Plotting {component} velocity hor_plane for height {self.z[z_idx]} at time {self.time[t_idx]}")

        plane = self.get_plane(z_idx, t_idx, component = 'u')

        if ax is None:
            fig, ax = plt.subplots()
        im = ax.pcolor(self.x, self.y, plane)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')

        fig.colorbar(im, ax=ax)

    def plot_mean_plane(self, z_idx, component = 'u', ax = None):

        print(f"Plotting {component} mean velocity hor_plane for height {self.z[z_idx]}")

        plane = self.get_mean_plane(z_idx, component = 'u')

        if ax is None:
            fig, ax = plt.subplots()
        im = ax.pcolor(self.x, self.y, plane)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')

        fig.colorbar(im, ax=ax)








        