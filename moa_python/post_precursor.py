import numpy as np
import sys
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
import netCDF4 as ncdf
from moa_python.post_abl_stats import Post_abl_stats
import os
import shutil

class Post_precursor:
    """
    Functions to evaluate precursor and restarted precursor simulations,
    and set up input files for restart.
    Large parts of this code are based on the "calc_inflowoutflow_stats.py" file
    developed by Ganesh Vijayakumar and modified by Alex Rybchuk.
    """

    def __init__(self, precursor_path, stats_file, t_start, t_end):
        """
        Loads abl_statistics dataset
        Args:
            stats_file (path): Name or path of the file containing statistics
                               during time of interest
            t_start (float): Start time of inflow/outflow simulation
            t_end (float): End time of inflow/outflow simulation
        """

        self.precursor_path = precursor_path
        if os.path.exists(stats_file): self.stats_file = stats_file
        elif os.path.exists(os.path.join(self.precursor_path,stats_file)): self.stats_file = os.path.join(self.precursor_path,stats_file)
        else: sys.exit(f"ABL statistics file {stats_file} not found.")
        self.dset = ncdf.Dataset(self.stats_file)
        self.t_start = t_start
        self.t_end = t_end
        print(f'ABL statistics file {stats_file} loaded succesfully.')
    
    def calc_runtime_stats(self,restart_path=os.getcwd()):
        """
        Examine statistics file and calculate relevant quantities for inflow/outflow simulation
        Args:
            restart_path (path): Path to save restart file to, defaults to current directory
        """

        time = self.dset['time'][:]
        t_filter = (time > self.t_start) & (time < self.t_end)

        if (not t_filter.any()):
            dset_tstart = time[0]
            dset_tend = time[-1]
            sys.exit(textwrap.dedent(f"""
            Statistics file time window does not cover desired run time of inflow/outflow simulation

            Desired start time   = {str(self.t_start)}
            Desired end time     = {str(self.t_end)}

            Time window in statistics file = {str(dset_tstart)} - {str(dset_tend)}
            """))

        self.ioparams = {'ABL.wall_shear_stress_type': '"local"','ABL.inflow_outflow_mode': 'true',
                         'BoussinesqBuoyancy.read_temperature_profile': 'true',
                         'BoussinesqBuoyancy.tprofile_filename': 'avg_theta.dat'}
        mean_profiles = self.dset['mean_profiles']
        self.ioparams['ABL.wf_vmag'] = np.average(mean_profiles['hvelmag'][t_filter, 0])
        u_mean = np.average(mean_profiles['u'][t_filter, 0])
        v_mean = np.average(mean_profiles['v'][t_filter, 0])
        self.ioparams['ABL.wf_velocity'] = [u_mean, v_mean, 0]
        self.ioparams['ABL.wf_theta'] = np.average(mean_profiles['theta'][t_filter, 0])
        abl_force_x = np.average(self.dset['abl_forcing_x'][t_filter])
        abl_force_y = np.average(self.dset['abl_forcing_y'][t_filter])
        self.ioparams['BodyForce.magnitude'] = [abl_force_x, abl_force_y, 0]

        ## TO DO: change writing avg_theta to optional/seperate function? Also: does restart_path need to be in self?
        if not hasattr(self, "restart_path"): self.restart_path = restart_path
        h = self.dset['mean_profiles']['h'][:]
        avg_theta = np.average(mean_profiles['theta'][t_filter, :], 0)
        with open(os.path.join(restart_path,'avg_theta.dat'),'w') as f:
            f.write('{} \n'.format(avg_theta.size))
            for i,t in enumerate(avg_theta):
                f.write('{} {} \n'.format(h[i], t))
        print(f"Temperature file succesfully written to '{self.restart_path}/avg_theta.dat'.")

    
    def general_precursor_to_restart(self, **kwargs):
        """
        Function to change the following variables in the restart simulation input file:
            time.fixed_dt           [optional]      input as: fixed_dt = (float)
            ABL.bndry_file          [optional]      input as: bndry_file = (str)
            ABL.bndry_io_mode
            io.restart_file         [optional]      input as: restart_file = (str)
            geometry.is_periodic    [optional]      input as: periodicity = (list) -> 
                                                        automatically sets xlo/ylo/xhi/yhi
            ICNS.source_terms
        Args in:
            **kwargs (str): 
        """
        self.restart_params = {"ABL.bndry_io_mode": 1,
                               "ICNS.source_terms": "BoussinesqBuoyancy CoriolisForcing BodyForce ABLMeanBoussinesq"}
        for variable, value in kwargs.items():
            if variable == "periodicity":
                self.restart_params["geometry.is_periodic"] = value
                from re import findall
                temp_line = [idx for idx, curr_line in enumerate(self.input_file) if "ABL.reference_temperature" in curr_line][0]
                density_line = [idx for idx, curr_line in enumerate(self.input_file) if "incflo.density" in curr_line][0]
                ref_temp = findall(r'\d+\.\d+',self.input_file[temp_line])
                ref_density = findall(r'\d+\.\d+',self.input_file[density_line])
                if value[0] == 0:
                    self.restart_params["xlo.type"] = '"mass_inflow"'
                    self.restart_params["xlo.density"] = ref_density
                    self.restart_params["xlo.temperature"] = ref_temp
                    self.restart_params["xlo.tke"] = '0.0'
                    self.restart_params["xhi.type"] = '"pressure_outflow"'
                if value[1] == 0:
                    self.restart_params["ylo.type"] = '"mass_inflow"'
                    self.restart_params["ylo.density"] = ref_density
                    self.restart_params["ylo.temperature"] = ref_temp
                    self.restart_params["ylo.tke"] = '0.0'
                    self.restart_params["yhi.type"] = '"mass_inflow"'
            if variable == "fixed_dt": self.restart_params["time.fixed_dt"] = value
            if variable == "restart_file": self.restart_params["io.restart_file"] = value
            if variable == "bndry_file": self.restart_params["ABL.bndry_file"] = value

        line_xlo = [idx for idx, curr_line in enumerate(self.input_file) if "Boundary conditions" in curr_line][0]+1
        self.alter_input_variables("restart_params", line_xlo)


    def read_input_file(self, restart_filename=None, restart_path=os.getcwd(), precursor_filename=None):
        """
        Reads the current restart simulation input file
        Args:
            restart_filename (str, optional): name of existing restart file to alter
            restart_path (path, optional): directory were restart file lives or should live, 
                                                defaults to current directory
            precursor_filename (str, optional): name of the precursor file to copy (should live in self.precursor_path),
                                                defaults to None (assumes restart input file already exists)
        """

        ## Determine location of input file to read
        self.restart_path = restart_path
        if (restart_filename is not None) and (os.path.exists(os.path.join(self.restart_path,restart_filename))):
            self.restart_input_dir = os.path.join(self.restart_path,restart_filename)
            print(f"Editing existing restart input file {self.restart_input_dir}.")
        elif precursor_filename is not None: 
            self.restart_input_dir = os.path.join(self.precursor_path,precursor_filename)
            print(f"Creating restart input file based on precursor input file {self.restart_input_dir}.")
        else:
            sys.exit(f"No restart input file found. Please provide either an existing restart input file, or a precursor input file to copy.")

        ## Read input file
        with open(self.restart_input_dir) as file:
            self.input_file = file.readlines()


    def alter_input_variables(self, dict_name, line_idx=None):
        """
        Add or alter a variable in the restart simulation input file
        Args in:
            variable (str): variable to add or alter
            value (str, float, or list): value of said variable
            line_idx (int): position to add variable in input file,
                                ignored if variable is already in input file;
                                defaults to bottom of file
        """

        ## Loop over variables in dictionary
        for variable, value in getattr(self,dict_name).items():
            ## Create string to add to file
            if isinstance(value,list):
                valuestr = ''
                for n in range(len(value)):
                    valuestr = valuestr+str(value[n])+' '
            else:
                valuestr = str(value)
            altered_variable = variable.ljust(40)+' = '+valuestr+'\n'

            ## Find place to add/alter variable
            line = [idx for idx, curr_line in enumerate(self.input_file) if variable in curr_line]
            if len(line) > 1:
                print(f"Warning: {variable} existed multiple times within {self.restart_input_dir}\nDuplicates are removed.")
                for idx_dup in reversed(line[1:]):
                    del self.input_file[idx_dup]
                self.input_file[line[0]] = altered_variable
            elif len(line) == 1:
                self.input_file[line[0]] = altered_variable
            else:
                if line_idx is None: line_idx = len(self.input_file)+1
                if (isinstance(line_idx,list)) and (len(line_idx) == 0): line_idx = len(self.input_file)+1
                self.input_file.insert(line_idx,altered_variable)
                line_idx += 1


    def define_flowfield_geo(self, xbndry, ybndry, ref_position=[0,0,0], rot_diam=240, hub_height=150, angle = 0, unit = 'D', plot_field=True):
        """
        Initialize refinement and sampling by defining the flow field geometry
        """
        
        # Maybe combine in one dict?
        self.turb_loc = np.array(ref_position)
        self.hub_height = hub_height
        self.plot_field = plot_field
        self.angle = angle
        self.xbndry = xbndry
        self.ybndry = ybndry
        self.refinements = {}
        if unit == 'D': 
            self.turb_loc = self.turb_loc/rot_diam
            self.rot_diam = rot_diam
        else: 
            self.rot_diam = 1

        if plot_field:
            self.fig, self.ax = plt.subplots()
            self.ax.plot(ref_position[0],ref_position[1],'ob')
            # self.ax.plot([ref_position[0]+rot_diam/2*np.sin(angle/180*np.pi), \
            #                  ref_position[0]-rot_diam/2*np.sin(angle/180*np.pi)],\
            #                 [ref_position[1]-rot_diam/2*np.cos(angle/180*np.pi),
            #                  ref_position[1]+rot_diam/2*np.sin(angle/180*np.pi)],'b')
            self.ax.set_xlim(xbndry)
            self.ax.set_ylim(ybndry)
            self.ax.set_aspect('equal')
            self.ax.grid(True)

    
    def add_box_refinement(self, label_name, x_coor, y_coor, z_coor, level=0, shape_name = None, plot_field = True):
        """
        Add a box mesh refinement to the refinements dictionary.
        """

        if shape_name is None: shape_name = label_name
        origin, xaxis, yaxis, zaxis = self.check_box_coordinates(x_coor, y_coor, z_coor, plot_field)

        ## Search if there is already a refinement defined somewhere
        if not hasattr(self.refinements,label_name):
            self.refinements[label_name] = {"type": "GeometryRefinement",
                                            "shapes": shape_name,
                                            "level": level}
        elif self.refinements[label_name]["level"] != level:
            print(f"Warning: second refinement added to '{label_name}' at a different refinement level!")
        self.refinements[label_name][shape_name] = {"type": "box",
                                                    "origin": origin,
                                                    "xaxis": xaxis,
                                                    "yaxis": yaxis,
                                                    "zaxis": zaxis}
        

    def check_box_coordinates(self, x_coor, y_coor, z_coor, plot_field = True):
        """
        Shows the origin and axes for a certain desired box refinement.
        Inputs:
            x_coor, y_coor (list):  Start and end point in x- and y-direction w.r.t. turbine location
                                        (can be defined in rotor diameter (default) or in meters)
            z_coor (list):          Start and end point in z-direction w.r.t. ground [in meters, not D!]
            rot_diam (float):       Turbine rotor diameter (default: 240 (=IEA 15 MW))
            ref_position (list):    Reference position w.r.t. which x, y, and z-coordinates are taken (default: [0,0,0])
            angle (float, opt):     Inflow wind speed angle in degrees (default: 0 deg)
            unit (str, opt):        'D' (rotor diameter, default) or 'm' (meters), unit of x and y coordinates
        """
            
        # Add function that checks whether define_flowfield_geo has already run?
        if self.angle == 30:
            sin = 0.5
            cos = np.sqrt(3)/2
        else:
            sin = np.sin(self.angle*np.pi/180)
            cos = np.cos(self.angle*np.pi/180)
        origin = np.array([self.turb_loc[0] + cos*x_coor[0] - sin*y_coor[0], \
                self.turb_loc[1] + sin*x_coor[0] + cos*y_coor[0], \
                z_coor[0]/self.rot_diam]) * self.rot_diam
        xaxis = np.array([(x_coor[1]-x_coor[0])*cos, (x_coor[1]-x_coor[0])*sin, 0]) * self.rot_diam
        yaxis = np.array([-(y_coor[1]-y_coor[0])*sin, (y_coor[1]-y_coor[0])*cos, 0]) * self.rot_diam
        zaxis = np.array([0, 0, (z_coor[1]-z_coor[0])])

        if self.plot_field and plot_field: self.plot_hor_plane(origin,xaxis,yaxis,'grey','dashed')
        elif self.plot_field: sys.exit("Plane not plotted, because 'plot_field' was initially set to False.")

        return origin, xaxis, yaxis, zaxis

    
    def plot_hor_plane(self,origin,axis1,axis2,color='r',linestyle='solid'):
        """
        Auxiliary function that plots horizontal planes/boxes.
        """

        if self.plot_field:
            self.ax.plot([origin[0], origin[0]+axis1[0]],[origin[1],origin[1]+axis1[1]],color,linestyle=linestyle)
            self.ax.plot([origin[0], origin[0]+axis2[0]],[origin[1],origin[1]+axis2[1]],color,linestyle=linestyle)
            self.ax.plot([origin[0]+axis1[0], origin[0]+axis1[0]+axis2[0]],[origin[1]+axis1[1],origin[1]+axis1[1]+axis2[1]],color,linestyle=linestyle)
            self.ax.plot([origin[0]+axis2[0], origin[0]+axis1[0]+axis2[0]],[origin[1]+axis2[1],origin[1]+axis1[1]+axis2[1]],color,linestyle=linestyle)
        else:
            sys.exit("Plane not plotted, because 'plot_field' was initially set to False.")


    def plot_ver_plane(self,origin,axis1,offsets,normal,color='g',linestyle='solid'):
        """
        Auxiliary function that plots vertical planes.
        """

        if self.plot_field:
            for offset in offsets:
                self.ax.plot([origin[0], origin[0]+axis1[0]]+offset*normal[0],\
                            [origin[1], origin[1]+axis1[1]]+offset*normal[1],'g')
        else:
            sys.exit("Plane not plotted, because 'plot_field' was initially set to False.")


    def clear_refinements(self):
        """
        Clears all defined refinements.
        """                

        self.refinements = {}
        if self.plot_field:
            self.ax.clear()
            self.ax.plot(self.turb_loc[0]*self.rot_diam,self.turb_loc[1]*self.rot_diam,'ob')
            # self.ax.plot([ref_position[0]+rot_diam/2*np.sin(angle/180*np.pi), \
            #                  ref_position[0]-rot_diam/2*np.sin(angle/180*np.pi)],\
            #                 [ref_position[1]-rot_diam/2*np.cos(angle/180*np.pi),
            #                  ref_position[1]+rot_diam/2*np.sin(angle/180*np.pi)],'b')
            self.ax.set_xlim(self.xbndry)
            self.ax.set_ylim(self.ybndry)
            self.ax.set_aspect('equal')
            self.ax.grid(True)


    def add_runtime_stats(self, restart_filename=None, restart_path=os.getcwd(), precursor_filename=None, write_file=False):
        """
        Writes the runtime stats from calc_runtime_stats to the restart input file.
        Args in:
            restart_filename (str, optional): name of the restart file to alter, 
                                                defaults to None (using precursor file)
                                                restart_path (path, optional): directory were restart file lives or should live, 
                                                defaults to current directory
            precursor_filename (str, optional): name of the precursor file to copy (should live in self.precursor_path),
                                                defaults to None (assumes restart input file already exists)
            write_file (bool, optional): write changes to the restart input file,
                                                defaults to False 
        """

        ## Calculate runtime stats
        if not hasattr(self,"ioparams"):
            self.calc_runtime_stats(restart_path)  # add verbose?

        ## Read input file
        if not hasattr(self,"input_file"):
            self.read_input_file(restart_filename, restart_path, precursor_filename)
        
        ## Add i/o parameters
        line_idx = [idx for idx, curr_line in enumerate(self.input_file) if "incflo.velocity" in curr_line]
        self.alter_input_variables("ioparams", line_idx)

        if write_file and (restart_filename is not None): self.write_input_file(restart_filename, restart_path)
        elif write_file: print("Warning: no restart input file name defined. Runtime stats not written to file. Use 'write_input_file' to create or alter restart input file.")


    def add_refinements(self, restart_filename=None, restart_path=os.getcwd(), precursor_filename=None, write_file=False):
        """
        Adds the refinements to the input file log.
        """

        ## Check if refinements exist already
        if (not hasattr(self,"refinements")) or (len(self.refinements) == 0):
            sys.exit(f"No refinements found. First run a refinement function.")

        ## Read input file
        if not hasattr(self,"input_file"):
            self.read_input_file(restart_filename, restart_path, precursor_filename)

        ## Add i/o parameters
        line_idx = [idx for idx, curr_line in enumerate(self.input_file) if "tagging" in curr_line]
        # If tagging exists: remove existing refinements
        # Then add new refinements
        # Maybe also add a function that defines the output file?
        self.alter_input_variables("refinements", line_idx)       


    def generate_restart_input_file(self, restart_filename=None, restart_path=os.getcwd(), precursor_filename=None, write_file=True, **kwargs):
        """
        Write AMR-Wind input file based on precursor file
        Args:
            restart_filename: filename of restart input file (on path self.restart_path)
        """

        ## ABL STATS FUNCTION
        self.add_runtime_stats(restart_filename, restart_path, precursor_filename, write_file=False)

        ## ADD TURBINE FUNCTION

        ## CREATE MESH FUNCTION

        ## ADD SAMPLING FUNCTION
        self.add_refinements(restart_filename, restart_path, precursor_filename, write_file=False)

        ## OTHER CHANGES FUNCTION
        self.general_precursor_to_restart(kwargs)

        ## Write restart input file
        self.write_input_file(restart_filename,restart_path)


    def write_input_file(self, restart_filename, restart_path=os.getcwd()):
        """
        Writes changes to restart input to the actual file
        """

        if not hasattr(self,"input_file"):
            sys.exit(f"No input file to write. First read input file using read_input_file.")

        restart_input_filename = os.path.join(restart_path,restart_filename)
        if os.path.exists(restart_input_filename):
            print(f"Input file {restart_input_filename} already exists. File will be overwritten.")
        f = open(restart_input_filename, "w")
        for i,t in enumerate(self.input_file):
            f.write(self.input_file[i])
        f.close()