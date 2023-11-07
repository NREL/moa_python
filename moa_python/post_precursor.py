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

    def __init__(self, precursor_path=os.getcwd, precursor_input_file=None, restart_path=os.getcwd(), restart_input_file=None):
        """
        Loads abl_statistics dataset
        Args:
            stats_file (path): Name or path of the file containing statistics
                               during time of interest
            t_start (float): Start time of inflow/outflow simulation
            t_end (float): End time of inflow/outflow simulation
        """

        self.precursor_path = precursor_path
        self.precursor_input_file = precursor_input_file
        self.restart_path = restart_path
        self.restart_input_file = restart_input_file
 

    def read_input_file(self,file_base=None):
        """
        Reads the current restart simulation input file.
        Args in:
            file_base (str): file to use as basis for restart file, either 'precursor' or 'restart'
                Defaults to seeing if the defined restart file already exists. If not, it uses the precursor file.
        """

        ## Determine which input file to read
        if (file_base != 'precursor') and (self.restart_input_file is not None) and (os.path.exists(os.path.join(self.restart_path,self.restart_input_file))):
            self.restart_input_dir = os.path.join(self.restart_path,self.restart_input_file)
            print(f"Editing existing restart input file {self.restart_input_dir}.")
        elif self.precursor_input_file is not None: 
            self.restart_input_dir = os.path.join(self.precursor_path,self.precursor_input_file)
            print(f"Creating restart input file based on precursor input file {self.restart_input_dir}.")
        else:
            sys.exit(f"No restart input file found. \nPlease provide either an existing restart input file, or a precursor input file to copy while initializing 'Post_precursor'.")

        ## Read input file
        with open(self.restart_input_dir) as file:
            self.input_file = file.readlines()

   
    def calc_runtime_stats(self, stats_file, t_start=None, t_end=None):
        """
        Examine statistics file and calculate relevant quantities for inflow/outflow simulation
        Args:
            restart_path (path): Path to save restart file to, defaults to current directory
        """

        if os.path.exists(stats_file): self.stats_file = stats_file
        elif os.path.exists(os.path.join(self.precursor_path,stats_file)): self.stats_file = os.path.join(self.precursor_path,stats_file)
        else: sys.exit(f"ABL statistics file {stats_file} not found.")
        self.dset = ncdf.Dataset(self.stats_file)
        time = self.dset['time'][:]
        if t_start is not None: self.t_start = t_start
        else: t_start = time[0]
        if t_start is not None: self.t_end = t_end
        else: t_end = time[-0]
        print(f'ABL statistics file {stats_file} loaded succesfully.')

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
        h = self.dset['mean_profiles']['h'][:]
        avg_theta = np.average(mean_profiles['theta'][t_filter, :], 0)
        if not os.path.exists(self.restart_path): os.mkdir(self.restart_path)
        with open(os.path.join(self.restart_path,'avg_theta.dat'),'w') as f:
            f.write('{} \n'.format(avg_theta.size))
            for i,t in enumerate(avg_theta):
                f.write('{} {} \n'.format(h[i], t))
        print(f"Temperature file succesfully written to '{self.restart_path}/avg_theta.dat'.")

 
    def add_runtime_stats(self, stats_file=None, t_start=None, t_end=None, write_file=False):
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
            if stats_file is None: sys.exit(f"Please provide abl_stats filename to gather runtime stats from.")
            else: self.calc_runtime_stats(stats_file, t_start, t_end)  # add verbose?

        ## Read input file
        if not hasattr(self,"input_file"):
            self.read_input_file(self.restart_input_file, self.restart_path, self.precursor_input_filename)
        
        ## Add i/o parameters
        line_idx = self.find_line("incflo.velocity")
        self.alter_dict_variables("ioparams", line_idx)

        if write_file and (self.restart_input_file is not None): self.write_input_file(self.restart_input_file, self.restart_path)
        elif write_file: print("Warning: no restart input file name defined. Runtime stats not written to file. Use 'write_input_file' to create or alter restart input file.")

   
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
            elif variable == "fixed_dt": self.restart_params["time.fixed_dt"] = value
            elif variable == "restart_file": self.restart_params["io.restart_file"] = value
            elif variable == "bndry_file": self.restart_params["ABL.bndry_file"] = value
            else: self.restart_params[variable] = value

        line_xlo = [idx for idx, curr_line in enumerate(self.input_file) if "Boundary conditions" in curr_line][0]+1
        self.alter_dict_variables("restart_params", line_xlo)


    def alter_dict_variables(self, dict_name, line_idx=None):
        """
        Add or alter a variable in the restart simulation input file
        Args in:
            dict_name (str or dict): dictionary or dict name to add or alter
            line_idx (int): position to add variable in input file,
                                ignored if variable is already in input file;
                                defaults to bottom of file
        """

        ## Get dictionary
        if isinstance(dict_name, str):
            dict_name = getattr(self,dict_name)

        ## Loop over variables in dictionary
        for variable, value in dict_name.items():
            self.find_line(variable, value, line_idx)
            line_idx = line_idx + 1


    def define_flowfield_geo(self, xbndry, ybndry, ref_position=[0,0,0], rot_diam=240, hub_height=150, angle = 0, unit = 'D', plot_field=True):
        """
        Initialize refinement and sampling by defining the flow field geometry
        """
        
        # Maybe combine in one dict?
        if len(ref_position)!=3: sys.exit(f"'Variable 'ref_position' has length {len(ref_position)}, but should have length 3.")
        self.turb_loc = np.array(ref_position)
        self.hub_height = hub_height
        self.plot_field = plot_field
        self.angle = angle
        self.xbndry = xbndry
        self.ybndry = ybndry
        self.refinements = {}
        self.sampling = {}
        self.max_refinement_level = 0
        self.unit = unit
        self.abs_rot_diam = rot_diam
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

    
    ## TO DO: Make functions (or change this function) for different types of refinements
    def apply_box_refinement(self, label_name, x_coor, y_coor, z_coor, level=0, shape_name = None, plot_field=True):
        """
        Add a box mesh refinement to the refinements dictionary.
        """

        if shape_name is None: shape_name = label_name
        origin, xaxis, yaxis, zaxis = self.check_box_coordinates(x_coor, y_coor, z_coor, plot_field)

        ## Search if there is already a refinement defined somewhere
        if not hasattr(self.refinements,label_name):
            self.refinements[label_name] = {"type": "GeometryRefinement",
                                            "shapes": shape_name,
                                            "level ": level}
        elif self.refinements[label_name]["level"] != level:
            sys.exit(f"A second refinement cannot added to '{label_name}' at a different refinement level!\n\
                     Current refinement level: {self.refinements[label_name]['level']}. Set refinement level: {level}")
        elif shape_name in self.refinements[label_name]["shapes"]:
            sys.exit(f"Shape name {shape_name} already present in refinement label {label_name}. Please pick a different name.")
        else:
            self.refinements[label_name]["shapes"] = self.refinements[label_name]["shapes"]+' '+shape_name
        self.refinements[label_name][shape_name] = {"type": "box",
                                                    "origin": origin,
                                                    "xaxis": xaxis,
                                                    "yaxis": yaxis,
                                                    "zaxis": zaxis}
        
        self.max_refinement_level = max(self.max_refinement_level,level+1)


    def check_box_coordinates(self, x_coor, y_coor, z_coor, plot_field=True):
        """
        Shows the origin and axes for a certain desired box refinement.
        Args in:
            x_coor, y_coor (list):  Start and end point in x- and y-direction w.r.t. turbine location
                                        (can be defined in rotor diameter (default) or in meters, as set in set_flowfield_geo)
            z_coor (list):          Start and end point in z-direction w.r.t. ground [in meters, not D!]
            plot_field (bool):      Plot the box refinement in a figure. Default: True
        """
            
        if not hasattr(self,"refinements"): sys.exit("No flow field geometry defined yet.\nRun 'define_flowfield_geo' first.")
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

        ## Perhaps add possibility to show the origin and axes

        return origin, xaxis, yaxis, zaxis
    

    def check_sampling_plane(self, plane_start, plane_end, dx=10, dy=None, offsets=0, angle=None, origin=None, normal=None, unit=None, plot_field=True):
        """
        Shows the origin and axes for a certain desired sampling plane.
        Args in:
            x_coor, y_coor (list):      Start and end point in x- and y-direction w.r.t. origin location defined in set_flowfield_geo
                                            (can be defined in rotor diameter (default) or in meters, as set in set_flowfield_geo)
            offsets (list or float):    List of offsets to be applied to the plane, default: 0.
            axis1, axis2 (str or list): Direction of the sampling plane axes, default: 'x' and 'y', respectively. 
                                        Can take negative directions (e.g.: '-x')
                                            Note: direction is aligned with the flow direction. 'xy' therefore means that the
                                            x-coordinate direction is aligned with the filow direction. 
                                        Alternatively: list with directional vector: [x_x, x_y, x_z]
        """

        if not hasattr(self,"sampling"): sys.exit("No flow field geometry defined yet.\nRun 'define_flowfield_geo' first.")
        if angle is None: angle = self.angle
        if dy is None: dy = dx
        if angle == 30:
            sin = 0.5
            cos = np.sqrt(3)/2
        else:
            sin = np.sin(self.angle*np.pi/180)
            cos = np.cos(self.angle*np.pi/180)
        if origin is None: origin = self.turb_loc
        if unit == 'm' and self.unit != 'm': 
            self.turb_loc = self.turb_loc*self.abs_rot_diam
            self.rot_diam = 1
        elif unit == 'D' and self.unit != 'D': 
            self.turb_loc = self.turb_loc/self.abs_rot_diam
            self.rot_diam = self.abs_rot_diam
        transform = np.matrix([[cos, sin, 0],[-sin, cos, 0],[0, 0, 1]])

        origin = (np.array(origin) + np.squeeze(np.array(plane_start*transform)))*self.rot_diam
        axes = np.identity(3)*(np.array(plane_end)-np.array(plane_start))*transform
        axis1 = np.squeeze(np.array(axes[np.where(~np.all((axes==0), axis=1))[0][0]]))*self.rot_diam
        axis2 = np.squeeze(np.array(axes[np.where(~np.all((axes==0), axis=1))[0][1]]))*self.rot_diam
        if normal is None: 
            normal = np.cross(axis2, axis1)/np.linalg.norm(np.cross(axis1, axis2))
            if np.max(normal) <= 0: normal = -normal
        offsets = np.array(offsets) * self.rot_diam
        num_points = np.array([int(np.round(np.linalg.norm(axis1))/dx+1), int(np.round(np.linalg.norm(axis2))/dy+1)])

        if plot_field:
            if isinstance(offsets,(list,np.ndarray,tuple)):
                for offset in offsets: 
                    self.plot_hor_plane(origin+offset*normal, axis1, axis2)
            else:
                self.plot_hor_plane(origin+offsets*normal, axis1, axis2)

        return origin, axis1, axis2, normal, offsets, num_points


    def apply_sampling_plane(self, sampling_name, plane_start, plane_end, output_freq=1, fields="velocity", \
                                dx=10, dy=None, offsets=0, angle=None, origin=None, normal=None, unit=None, plane_name=None, plot_field=True):
        """
        Add a sampling plane to the sampling dictionary.
        """

        if plane_name is None: plane_name = "plane"
        origin, axis1, axis2, normal, offsets, num_points = self.check_sampling_plane(plane_start, plane_end, dx, dy, \
                                                                                        offsets, angle, origin, normal, unit, plot_field)

        ## Search if there is already a sampling defined somewhere
        if not hasattr(self.sampling,sampling_name):
            self.sampling[sampling_name] = {"labels": plane_name,
                                            "fields": fields,
                                            "output_frequency": output_freq}
        elif plane_name in self.sampling[sampling_name]["fields"]:
            sys.exit(f"Plane name {plane_name} already present in sampling label {sampling_name}. Please pick a different name.")
        elif self.sampling[sampling_name]["output_frequency"] != output_freq:
            print(f"Warning: second plane added to '{sampling_name}' with a different output frequency!\n\
                    Original output frequency: {self.sampling[sampling_name]['output_frequency']} (used). Requested output frequency: {output_freq}")
        else: 
            self.sampling[sampling_name]["fields"] = self.sampling[sampling_name]["fields"]+' '+plane_name
        self.sampling[sampling_name][plane_name] = {"type": "PlaneSampler",
                                                    "origin": origin,
                                                    "axis1": axis1,
                                                    "axis2": axis2,
                                                    "num_points": num_points,
                                                    "normal": normal,
                                                    "offsets": offsets}
        
    
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
            sys.exit("Plane not plotted, because 'plot_field' was set to False in 'define_flowfield_geo'.")


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
            self.ax.set_xlim(self.xbndry)
            self.ax.set_ylim(self.ybndry)
            self.ax.set_aspect('equal')
            self.ax.grid(True)

    
    def find_line(self, variable_name, variable_value=None, line_idx=None, delete=False):
        """
        Function that looks if variable already exists in the input file.
        If a variable value is defined, given variable is added to or altered in the input file.
            Optional: define line_idx to set where to add variable. Otherwise it will be added to EoF.
        If delete is set to True, given variable will be deleted from the file.
        """

        line = [idx for idx, curr_line in enumerate(self.input_file) if variable_name in curr_line]
        ## Variable already exists multiple times
        if delete:
            if len(line) > 0:
                for n in reversed(line):
                    del self.input_file[n]
                return line
            elif len(line) == 1:
                del self.input_file[line[0]]
                return line
            else:
                return line
        elif variable_value is None:
            if len(line) == 0: return line
            else: return line[-1]
        else:
            altered_variable = format_variable_to_string(variable_name, variable_value)

            if len(line) > 1:
                print(f"Warning: {variable_name} existed multiple times within {self.restart_input_dir}\nDuplicates are removed.")
                for idx_dup in reversed(line[1:]):
                    del self.input_file[idx_dup]
                self.input_file[line[0]] = altered_variable
            # Variable already exists: overwrite
            elif len(line) == 1:
                self.input_file[line[0]] = altered_variable
            # Variable doesn't exist yet: add at desired location
            else:
                if line_idx is None: line_idx = len(self.input_file)+1
                if (isinstance(line_idx,list)): 
                    if (len(line_idx) == 0): line_idx = len(self.input_file)+1
                    elif (len(line_idx) == 1): line_idx = line_idx[0]
                    else: 
                        print(f"Warning: {variable_name} existed multiple times within {self.restart_input_dir}\nDuplicates are removed.")
                        for idx_dup in reversed(line_idx[1:]):
                            del self.input_file[idx_dup]
                self.input_file.insert(line_idx,altered_variable)
                line_idx += 1


    def add_sampling(self, write_file=False):
        """
        Adds the sampling to the input file log.
        """

        ## Check whether sampling are defined
        if (not hasattr(self,"sampling")) or (len(self.sampling) == 0):
            sys.exit(f"No sampling found. First run a sampling function.")

        ## Read input file
        if not hasattr(self,"input_file"):
            self.read_input_file()

        ## Delete existing sampling
        line_idx = self.find_line("incflo.post_processing")
        if isinstance(line_idx, int):
            print("Warning: Existing sampling found. These will be removed from file!")
            sampling_labels = self.input_file[line_idx].split()[2:]
            for sampling_label in sampling_labels:
                self.find_line(sampling_label,delete=True)
        else:
            line_idx = len(self.input_file)+1

        ## Add sampling text box
        line_idx = self.find_line("SAMPLING")
        if isinstance(line_idx, int):
            line_idx = line_idx + 2
        else:
            self.input_file.insert(len(self.input_file)+1, \
            "#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#\n#                SAMPLING               #\n#.......................................#\n")
            line_idx = len(self.input_file)+3

        ## Add sampling tag names
        self.find_line("incflo.post_processing", list(self.sampling.keys()), line_idx)
        self.input_file.insert(line_idx+1,'')
        line_idx += 2

        ## Add sampling
        for sampling_name, sampling_dict in self.sampling.items():
            for plane_name, plane_dict in sampling_dict.items():
                if isinstance(plane_dict,dict):
                    for var_name,var_val in plane_dict.items():
                        self.find_line(sampling_name+"."+plane_name+"."+var_name,var_val, line_idx)
                        line_idx += 1
                else:  #"tagging."+label_name+"."+shape_name+"."+var_name
                    self.find_line(sampling_name+"."+plane_name, plane_dict, line_idx)     
                    line_idx += 1
            self.input_file.insert(line_idx,'')
            line_idx += 1

        if write_file and (self.restart_input_filename is not None): self.write_input_file()
        elif write_file: print("Warning: no restart input file name defined. Runtime stats not written to file. Use 'write_input_file' to create or alter restart input file.")


    def add_refinements(self, write_file=False):
        """
        Adds the refinements to the input file log.
        """

        ## Check whether refinements are defined
        if (not hasattr(self,"refinements")) or (len(self.refinements) == 0):
            sys.exit(f"No refinements found. First run a refinements function.")

        ## Read input file
        if not hasattr(self,"input_file"):
            self.read_input_file()

        ## Delete existing refinements
        line_idx = [idx for idx, curr_line in enumerate(self.input_file) if "refinements" in curr_line]
        if len(line_idx) > 0:
            print("Warning: Existing refinements found. These will be overwritten!")
            for n in line_idx:
                del self.input_file[n]

        ## Add refinement text box
        line_idx = self.find_line(" REFINEMENT ")
        if isinstance(line_idx, int):
            line_idx = line_idx + 2
        else:
            self.input_file.insert(len(self.input_file)+1,\
            "#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#\n#        ADAPTIVE MESH REFINEMENT       #\n#.......................................#\n")
            line_idx = len(self.input_file)+3

        ## Add max refinement level
        level_line = self.find_line("amr.max_level")
        if isinstance(level_line, list):
            for n in level_line: del self.input_file[n]
        self.find_line("amr.max_level", self.max_refinement_level, line_idx)
        line_idx += 1

        ## Add refinement tag names
        self.find_line("tagging.labels", list(self.refinements.keys()), line_idx)
        self.input_file.insert(line_idx+1,'')
        line_idx += 2

        ## Add refinements
        for label_name, label_dict in self.refinements.items():
            for shape_name, shape_dict in label_dict.items():
                if isinstance(shape_dict,dict):
                    for var_name,var_val in shape_dict.items():
                        self.find_line("tagging."+label_name+"."+shape_name+"."+var_name,var_val, line_idx)
                        line_idx += 1
                else:
                    self.find_line("tagging."+label_name+"."+shape_name, shape_dict, line_idx)     
                    line_idx += 1
            self.input_file.insert(line_idx,'')
            line_idx += 1

        if write_file and (self.restart_input_filename is not None): self.write_input_file()
        elif write_file: print("Warning: no restart input file name defined. Runtime stats not written to file. Use 'write_input_file' to create or alter restart input file.")


    def apply_actuator(self, label, openfast_input_file=None, position=None, act_type="TurbineFastLine", **kwargs):
        """
        Add an actuator to the input file.
        Args in:
            label (str)
            type (str)
            **kwargs
        """

        if not hasattr(self,'act_labels'): self.act_labels = [label]
        elif label in self.act_labels: 
            print(f"Warning: Actuator label {label} already used. Existing actuators will be overwritten.")
            self.actuators = {}
        else: self.act_labels.append(label)
        self.actuators = {"labels": self.act_labels,
                         "type": act_type}
        if position is None: position = np.round(self.turb_loc*self.rot_diam,1)
        self.actuators[label] = {"base_position": position}
        if openfast_input_file is not None: self.actuators[label]["openfast_input_file"] = openfast_input_file
        ## TO DO: AUTOMIZE THE INPUTS NEEDED IN AMR-WIND BY FINDING THEM FROM OPENFAST INPUT FILES
        if not hasattr(self.actuators,act_type): self.actuators[act_type] = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    self.actuators[act_type][key2] = value2
            else:
                self.actuators[act_type][key] = value


    def add_actuators(self, write_file=False):

        ## Check whether actuators are defined
        if (not hasattr(self,"actuators")) or (len(self.actuators) == 0):
            sys.exit(f"No actuators found. First run an actuators function first.")

        ## Read input file
        if not hasattr(self,"input_file"):
            self.read_input_file()

        # Add actuator physics
        phys_line = self.find_line("incflo.physics")
        if phys_line is None:
            sys.exit("No 'incflo.physics' found in precursor file. Check if precursor file is correct.")
        elif not "Actuator" in self.input_file[phys_line]:
            self.input_file[phys_line] = self.input_file[phys_line] + ' ' + 'Actuator'
        
        # Add ActuatorForcing force term
        source_line = self.find_line("ICNS.source_terms")
        if source_line is None:
            sys.exit("No 'ICNS.source_terms' found in precursor file. Check if precursor file is correct.")
        elif not "ActuatorForcing" in self.input_file[source_line]:
            self.input_file[source_line] = self.input_file[source_line] + ' ' + 'ActuatorForcing'

        # Delete existing actuator lines
        check = self.find_line("Actuator.", delete=True)
        if len(check)>0:
            print("Warning: Existing actuator(s) found. These lines will be overwritten!")
            line_idx = check
        else:
            line_idx = self.find_line(" ACTUATORS ")

        ## Add actuators text box
        if len(line_idx) > 0:
            line_idx = line_idx + 2
        else:
            self.input_file.insert(len(self.input_file)+1,\
            "#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#\n#               ACTUATORS               #\n#.......................................#\n")
            line_idx = len(self.input_file)+3

        ## Add actuator type and label
        if hasattr(self,"act_labels"):
            self.act_labels = self.actuators['labels']
        else:
            self.act_labels = self.act_labels.append(self.actuators['labels'])
        self.input_file.insert(line_idx+1,format_variable_to_string("Actuator.labels", self.act_labels))

        ## Add actuators to file
        for act_name, act_dict in self.actuators.items():
            if not isinstance(act_dict,dict):
                self.find_line("Actuator."+act_name, act_dict, line_idx)
            else:
                for sub_name, sub_dict in act_dict.items():
                    if isinstance(sub_dict,dict):
                        self.alter_dict_variables(sub_dict, line_idx)
                    else:
                        self.find_line("Actuator."+act_name+"."+sub_name, sub_dict, line_idx)     
                        line_idx += 1
            self.input_file.insert(line_idx,'')
            line_idx += 1
        self.input_file.insert(line_idx,'')

        if write_file and (self.restart_input_filename is not None): self.write_input_file()
        elif write_file: print("Warning: no restart input file name defined. Runtime stats not written to file. Use 'write_input_file' to create or alter restart input file.")
                

    def copy_actuator_input(self,act_input_file):
        """
        WIP: Allow for an actuator to be copied from an existing AMR input file.
        """


    def generate_restart_input_file(self, write_file=True, **kwargs):
        """
        Write AMR-Wind input file based on precursor file
        Args:
            restart_filename: filename of restart input file (on path self.restart_path)
        """

        ## ADD ABL STATS
        self.add_runtime_stats(write_file=False)

        ## ADD ACTUATORS (IF DEFINED) - NOT WORKING YET
        if (hasattr(self,"actuators")) and (len(self.actuators) > 0):
            self.add_actuators(write_file=False)

        ## ADD SAMPLING (IF DEFINED)
        if (hasattr(self,"sampling")) and (len(self.sampling) > 0):
            self.add_sampling(write_file=False)

        ## ADD REFINEMENTS (IF DEFINED)
        if (hasattr(self,"refinements")) and (len(self.refinements) > 0):
            self.add_refinements(write_file=False)

        ## OTHER CHANGES FUNCTION
        if not hasattr(self,"restart_params"):
            self.general_precursor_to_restart(kwargs, write_file=False)

        ## Write restart input file
        if write_file: self.write_input_file()


    def write_input_file(self):
        """
        Writes changes to restart input to the actual file
        """

        if not hasattr(self,"input_file"):
            sys.exit(f"No input file to write. First read input file using read_input_file.")

        restart_input_filename = os.path.join(self.restart_path,self.restart_input_file)
        if os.path.exists(restart_input_filename):
            print(f"Input file {restart_input_filename} already exists.\nFile will be overwritten.")
        f = open(restart_input_filename, "w")
        for i,t in enumerate(self.input_file):
            f.write(self.input_file[i])
        f.close()

def format_variable_to_string(variable_name, variable_value):
        """
        Auxiliary function to format lines for AMR input files.
        """

        if isinstance(variable_value,(list,np.ndarray,tuple)):
            valuestr = ''
            for n in range(len(variable_value)):
                valuestr = valuestr+str(variable_value[n])+' '
        else:
            valuestr = str(variable_value)
        return variable_name.ljust(40)+' = '+valuestr+'\n'