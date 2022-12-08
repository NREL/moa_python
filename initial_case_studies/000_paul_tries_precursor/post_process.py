import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import netCDF4 as ncdf
import pandas as pd

def plot_stuff(field):
	d1 = ncdf.Dataset("your_directory/abl_statistics00000.nc")

	g1 = d1.groups["mean_profiles"]

	# float64 h(nlevels), float64 u(num_time_steps, nlevels), float64 v(num_time_steps, nlevels), float64 w(num_time_steps, nlevels), float64 hvelmag(num_time_steps, nlevels), float64 theta(num_time_steps, nlevels), float64 mueff(num_time_steps, nlevels), float64 u'theta'_r(num_time_steps, nlevels), float64 v'theta'_r(num_time_steps, nlevels), float64 w'theta'_r(num_time_steps, nlevels), float64 u'u'_r(num_time_steps, nlevels), float64 u'v'_r(num_time_steps, nlevels), float64 u'w'_r(num_time_steps, nlevels), float64 v'v'_r(num_time_steps, nlevels), float64 v'w'_r(num_time_steps, nlevels), float64 w'w'_r(num_time_steps, nlevels), float64 u'u'u'_r(num_time_steps, nlevels), float64 v'v'v'_r(num_time_steps, nlevels), float64 w'w'w'_r(num_time_steps, nlevels), float64 u'theta'_sfs(num_time_steps, nlevels), float64 v'theta'_sfs(num_time_steps, nlevels), float64 w'theta'_sfs(num_time_steps, nlevels), float64 u'v'_sfs(num_time_steps, nlevels), float64 u'w'_sfs(num_time_steps, nlevels), float64 v'w'_sfs(num_time_steps, nlevels)

	print(field)
	u1 = g1.variables[field]
	
	# avg 5-6 hrs, 5hrs -> 3600*5*2 = 36000 timesteps
  # todo, there is a time variable in netcdf file that could be used to find the correct time steps
	uavg1 = np.average(u1[35999:43199],axis=0)

	z = g1.variables["h"]

	plt.figure()
	plt.clf()
	plt.plot(uavg1,z,label="your_label")
	plt.legend()
	plt.xlabel(field)
	plt.ylabel("Height [m]")
	plt.savefig(field+"avg.pdf")
	plt.savefig(field+"avg.png")

if __name__ == "__main__":
	plot_stuff("u")
	plot_stuff("v")
	plot_stuff("theta")
	plot_stuff("u'u'_r")
	plot_stuff("u'v'_r")
	plot_stuff("u'w'_r")
	plot_stuff("v'w'_r")
	plot_stuff("v'v'_r")
	plot_stuff("w'w'_r")
	plot_stuff("w'theta'_r")
