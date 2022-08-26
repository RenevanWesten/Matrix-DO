#Program plots the analytical and DO Burgers solutions

from pylab import *
import netCDF4 as netcdf
import numpy as np

#Directory for the output
directory = '../Data/Burgers/'

def PeriodicBoundaries(x, y):
	"""Make boundaries periodic as y(x=0) = y(x=1)
	Copy y(x=0) to y(x=1)"""
	x_new		= np.zeros(len(x) + 1)
	y_new		= np.zeros(len(x_new))
	x_new[:-1]	= x
	x_new[-1]	= 1
	y_new[:-1]	= y
	y_new[-1]	= y[0]

	return x_new, y_new

def DistributionY(Y_data, y_min = -8, y_max = 8):
	"""Reshape data to make distribution"""
	#Rescale data
	bar_width	= 0.25
	bars		= np.arange(y_min - bar_width / 2.0, y_max + bar_width, bar_width)
	bars_height	= np.zeros(len(bars) - 1)

	for bar_i in range(len(bars) - 1):
		#Generate histogram, loop over all relevant bins	
		index			= np.where((Y_data >= bars[bar_i]) & (Y_data < bars[bar_i + 1]))[0]
		bars_height[bar_i]	= len(index)

	#Normalise the historgram
	bars_height	= bars_height / sum(bars_height)

	#Take the center point of each bar
	bars		= 0.5 * (bars[1:] + bars[:-1])

	return bars, bars_height

#-----------------------------------------------------------------------------
#---------------------------MAIN SCRIPT STARTS HERE---------------------------
#-----------------------------------------------------------------------------	

nx          = 128   #Number of grid points
nV          = 32     #Number of DO modes
mu          = 0.005

#-----------------------------------------------------------------------------

fh = netcdf.Dataset(directory+'BE_model_nx_'+str(nx)+'_mu_'+str(int(mu))+'_'+str(mu)[2:]+'_nV_'+str(nV)+'.nc', 'r')

time		= fh.variables['time'][:]
x		    = fh.variables['x'][:] 
mean		= fh.variables['Mean'][-1] 
V_all		= fh.variables['V'][-1] 	
Y_all		= fh.variables['Y'][-1] 		
norm_mean	= fh.variables['norm_mean'][:]
norm_V		= fh.variables['norm_V'][:]

fh.close()

#-----------------------------------------------------------------------------

fh = netcdf.Dataset(directory+'BE_exact_end_solution_nx_'+str(nx)+'_mu_'+str(int(mu))+'_'+str(mu)[2:]+'.nc', 'r') 

exact_sol = fh.variables['BE_end'][:]

fh.close()

#-----------------------------------------------------------------------------

#Generate periodic boundaries
x_plot, exact_sol_mean	= PeriodicBoundaries(x, np.mean(exact_sol, axis = 1))
x_plot, DO_mean		    = PeriodicBoundaries(x, mean)

fig, ax	= subplots()


graph_1	= ax.plot(x_plot, exact_sol_mean, '-r', linewidth = 2.0, label = 'Exact')
graph_2	= ax.plot(x_plot, DO_mean, '-k', linewidth = 2.0, label = 'DO')

ax.grid()
ax.set_xlabel('$x$')
ax.set_ylabel('Amplitude')
ax.set_xlim(0, 1)
ax.set_ylim(-0.2, 0.6)  

ax2 = ax.twinx()

#Determine the variance
x_plot, exact_var	= PeriodicBoundaries(x, np.var(exact_sol, axis = 1))
x_plot, DO_var		= PeriodicBoundaries(x, np.var(np.matmul(np.array(V_all), np.array(Y_all)), axis = 1))

graph_3	= ax2.plot(x_plot, exact_var * 100, '--r', linewidth = 2.0)
graph_4	= ax2.plot(x_plot, DO_var * 100, '--k', linewidth = 2.0)

ax2.set_ylabel(r'Variance ($\times 10^{-2}$)')
ax2.set_ylim(0, 8)

graphs		= graph_1 + graph_2
legend_labels 	= [l.get_label() for l in graphs]
legend_1      	= ax.legend(graphs, legend_labels, loc = 'lower left', ncol=1, numpoints = 1)

ax.set_title('a) Exact solution and DO')


#-----------------------------------------------------------------------------
#Generate periodic boundaries
x_plot, DO_mean	= PeriodicBoundaries(x, mean)
x_plot, V_1	= PeriodicBoundaries(x, V_all[:, 0])
x_plot, V_2	= PeriodicBoundaries(x, V_all[:, 1])
x_plot, V_3	= PeriodicBoundaries(x, V_all[:, 2])
x_plot, V_4	= PeriodicBoundaries(x, V_all[:, 3])

fig, ax	= subplots()

graph_1     = ax.plot(x_plot, DO_mean, '-k', linewidth = 2.0, label = '$\overline{v}$')
graph_2     = ax.plot(x_plot, V_1, linestyle = '-', color = 'red', linewidth = 1.5, label = '$v_1$')
graph_3     = ax.plot(x_plot, V_2, linestyle = '-', color = 'blue', linewidth = 1.5, label = '$v_2$')
graph_4     = ax.plot(x_plot, V_3, linestyle = '-', color = 'firebrick', linewidth = 1.5, label = '$v_3$')
graph_5     = ax.plot(x_plot, V_4, linestyle = '-', color = 'cyan', linewidth = 1.5, label = '$v_4$')

ax.grid()
ax.set_xlabel('$x$')
ax.set_ylabel('Amplitude')
ax.set_xlim(0, 1)
ax.set_ylim(-0.2, 0.4)  
    
graphs		= graph_1 + graph_2 + graph_3 + graph_4 + graph_5
legend_labels 	= [l.get_label() for l in graphs]
legend_1      	= ax.legend(graphs, legend_labels, loc = 'upper right', ncol=1, numpoints = 1)

ax.set_title('b) DO mean and DO modes')
#-----------------------------------------------------------------------------
fig, ax     = subplots()

graph_1	= ax.plot(time, norm_mean, '-k', linewidth = 2.0, label = '$\overline{v}$')
graph_2	= ax.plot(time, norm_V[:,0], linestyle = '-', color = 'red', linewidth = 1.5, label = '$v_1$')
graph_3	= ax.plot(time, norm_V[:,1], linestyle = '-', color = 'blue', linewidth = 1.5, label = '$v_2$')
graph_4	= ax.plot(time, norm_V[:,2], linestyle = '-', color = 'firebrick',linewidth = 1.5, label = '$v_3$')
graph_5	= ax.plot(time, norm_V[:,3], linestyle = '-', color = 'cyan', linewidth = 1.5, label = '$v_4$')

ax.set_yscale('log')
ax.set_ylim(10**(-3.0), 10**1)
ax.set_xlim(0, 0.8)
ax.set_xticks(np.arange(0, 0.81, 0.2))
ax.set_xlabel('Time')
ax.set_ylabel('Variance')
ax.grid()

graphs		= graph_1 + graph_2 + graph_3 + graph_4 + graph_5
legend_labels 	= [l.get_label() for l in graphs]
legend      	= ax.legend(graphs, legend_labels, loc = 'lower right', ncol=1, numpoints = 1)

ax.set_title('c) Variance of DO modes')

#-----------------------------------------------------------------------------
bars, Y_1_dist	= DistributionY(Y_all[0])
bars, Y_2_dist	= DistributionY(Y_all[1])
bars, Y_3_dist	= DistributionY(Y_all[2])
bars, Y_4_dist	= DistributionY(Y_all[3])

fig, ax     = subplots()

graph_4	= ax.plot(bars, Y_4_dist, linestyle = '-', color = 'cyan', linewidth = 1.5, label = '$Y_4$')
graph_3	= ax.plot(bars, Y_3_dist, linestyle = '-', color = 'firebrick',linewidth = 1.5, label = '$Y_3$')
graph_2	= ax.plot(bars, Y_2_dist, linestyle = '-', color = 'blue', linewidth = 1.5, label = '$Y_2$')
graph_1	= ax.plot(bars, Y_1_dist, linestyle = '-', color = 'red', linewidth = 1.5, label = '$Y_1$')

ax.set_ylim(-0.015, 0.35)
ax.set_xlim(-7, 7)
ax.set_xticks(np.arange(-6, 6.1, 2))
ax.set_xlabel('$Y$')
ax.set_ylabel('Probability distribution function')
ax.grid()

ax.set_title('d) Probability distribution functions')

graphs		= graph_1 + graph_2 + graph_3 + graph_4
legend_labels 	= [l.get_label() for l in graphs]
legend      	= ax.legend(graphs, legend_labels, loc = 'upper right', ncol=1, numpoints = 1)

show()
