#Program conducts Dynamically Orthogonal (DO) field equations for stochastic dynamical systems
#Original (Matlab) script received from Fred Wubs
#This script contains the main DO frame work

from pylab import *
import numpy as np
import netCDF4 as netcdf
from matplotlib import colors, ticker, cm

#Directory for the output
directory = '../Data/QG/'

def DistributionYtime(time, Y_data, y_min = -8000, y_max = 8000):
	"""Reshape data to make distribution"""
	#Rescale data
	bar_width	= 10
	bars		= np.arange(y_min - bar_width / 2.0, y_max + bar_width, bar_width)
	bars_height	= np.zeros((len(time), len(bars) - 1))

	for bar_i in range(len(bars) - 1):
		#Generate histogram, loop over all relevant bins	
		index			= np.where((Y_data >= bars[bar_i]) & (Y_data < bars[bar_i + 1]))[0]
		
		if len(index) == 0:
			continue

		for time_i in np.unique(index):
			#Loop over all the relevant time indices and save the counts
			bars_height[time_i, bar_i]	= len(index[index == time_i])

	#Normalise the historgram
	for time_i in range(len(time)):
		bars_height[time_i]	= bars_height[time_i] / sum(bars_height[time_i])

	#Take the center point of each bar
	bars		= 0.5 * (bars[1:] + bars[:-1])

	return bars, bars_height

#-----------------------------------------------------------------------------
#---------------------------MAIN SCRIPT STARTS HERE---------------------------
#-----------------------------------------------------------------------------	

nx          = 64 #Numbber of zonal dimensions
ny          = 64 #Number of meridional dimensions
nV          = 4 #Number of DO modes
Re	        = 40
delta_t	    = 0.001
stoch_amp   = 10**(0)
#-----------------------------------------------------------------------------

fh = netcdf.Dataset(directory+'QG_sorted_basis_nx_'+str(nx)+'_ny_'+str(ny)+'_dt_'+str(int(delta_t))+'_'+str(delta_t)[2:]+'_nV_'+str(nV)+'_Re_'+str(Re)+'_Stoch_amp_'+str("{:.0e}".format(stoch_amp))+'.nc', 'r')

#Writing data to correct variable	
time 		= fh.variables['time'][:]  	
x		    = fh.variables['x'][:] 			
y		    = fh.variables['y'][:] 		
nY		    = fh.variables['nY'][:-1] 	
stream_mean	= fh.variables['stream_mean'][-1] 
stream_all	= fh.variables['stream_V'][-1]
vor_mean	= fh.variables['vor_mean'][-1] 
vor_all		= fh.variables['vor_V'][-1]
Y_all		= fh.variables['Y'][:]
norm_x		= fh.variables['norm_mean'][:] 
norm_V		= fh.variables['norm_V'][:] 

fh.close()

time_end_plot	= int(round(time[-1],0))

#-----------------------------------------------------------------------------

fig, ax     = subplots()

graph_1	= ax.plot(time, norm_x, '-k', linewidth = 2.0, label = '$\overline{v}$')
graph_2	= ax.plot(time, norm_V[:,0], linestyle = '-', color = 'red', linewidth = 1.5, label = '$v_1$')
graph_3	= ax.plot(time, norm_V[:,1], linestyle = '-', color = 'blue', linewidth = 1.5, label = '$v_2$')
graph_4	= ax.plot(time, norm_V[:,2], linestyle = '-', color = 'firebrick',linewidth = 1.5, label = '$v_3$')
graph_5	= ax.plot(time, norm_V[:,3], linestyle = '-', color = 'cyan', linewidth = 1.5, label = '$v_4$')

ax.set_yscale('log')
ax.set_ylim(10**(-5.0), 10**8.0)
ax.set_xlim(0, 15)
ax.set_xticks(np.arange(0, 15.1, 3))
ax.set_xlabel('Time')
ax.set_ylabel('Variance')
ax.grid()

graphs		= graph_1 + graph_2 + graph_3 + graph_4 + graph_5
legend_labels 	= [l.get_label() for l in graphs]

if stoch_amp == 10**(0):
	x_plot, y_plot	= np.meshgrid(x, y)

	ax2 	= fig.add_axes([0.2, 0.15, 0.21, 0.24])
	CS1  	= ax2.contourf(x_plot, y_plot, vor_all[0] * 10**2.0, levels = np.arange(-5, 5.1, 0.25), extend = 'both', cmap = 'RdBu_r')
	CS2	= ax2.contour(x_plot, y_plot, stream_all[0] * 10**4.0, levels = np.arange(-2, 2.1, 1), colors = 'k')
	CS2_0	= ax2.contour(x_plot, y_plot, stream_all[0] * 10**4.0, levels = [0], colors = 'gray', linewidths = 3.0)

	ax2.set_xticklabels([])
	ax2.set_yticklabels([])
	ax2.set_title('$v_1(t = '+str(time_end_plot)+')$')

legend      	= ax.legend(graphs, legend_labels, loc = 'lower right', ncol=1, numpoints = 1)
ax.set_title(r'a) $Re = 40$, $n_V = 4$, $\sigma = 1$, $n_x = n_y = 64$')
#-----------------------------------------------------------------------------
   
    
fig, ax	= subplots()
x_plot, y_plot	= np.meshgrid(x, y)

CS1  	= ax.contourf(x_plot, y_plot, vor_mean, levels = np.arange(-300, 300.1, 15), extend = 'both', cmap = 'RdBu_r')
CS2	= ax.contour(x_plot, y_plot, stream_mean, levels = np.arange(-2, 2.1, 0.5), colors = 'k')
CS2_0	= ax.contour(x_plot, y_plot, stream_mean, levels = [0], colors = 'gray', linewidths = 3.0)
cbar 	= colorbar(CS1, ticks = np.arange(-300, 300.1, 100))
cbar.set_label(r'Vorticity')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('c) $\overline{v}(t = '+str(time_end_plot)+')$, vorticity and streamfunction')

#-----------------------------------------------------------------------------

Y_1_1_time	= np.zeros(len(time))
Y_2_1_time	= np.zeros(len(time))
Y_3_1_time	= np.zeros(len(time))
Y_4_1_time	= np.zeros(len(time))
Y_1_2_time	= np.zeros(len(time))
Y_2_2_time	= np.zeros(len(time))
Y_3_2_time	= np.zeros(len(time))
Y_4_2_time	= np.zeros(len(time))

for time_i in range(len(time)):
	Y_index_1		= np.where(Y_all[time_i, 0] >= 0.0)[0]
	Y_index_2		= np.where(Y_all[time_i, 0] < 0.0)[0]
	Y_1_1_time[time_i]	= np.mean(Y_all[time_i, 0, Y_index_1])
	Y_2_1_time[time_i]	= np.mean(Y_all[time_i, 1, Y_index_1])
	Y_3_1_time[time_i]	= np.mean(Y_all[time_i, 2, Y_index_1])
	Y_4_1_time[time_i]	= np.mean(Y_all[time_i, 3, Y_index_1])
	Y_1_2_time[time_i]	= np.mean(Y_all[time_i, 0, Y_index_2])
	Y_2_2_time[time_i]	= np.mean(Y_all[time_i, 1, Y_index_2])
	Y_3_2_time[time_i]	= np.mean(Y_all[time_i, 2, Y_index_2])
	Y_4_2_time[time_i]	= np.mean(Y_all[time_i, 3, Y_index_2])

fig, ax     = subplots()

graph_1	= ax.plot(Y_1_1_time, time, color = 'red', linestyle = '-', linewidth = 2.0, label = '$\overline{Y_1^{+}}}$')
graph_2	= ax.plot(Y_2_1_time, time,  color = 'blue', linestyle = '-', linewidth = 2.0, label = '$\overline{Y_2(Y_1^{+})}$')
graph_3	= ax.plot(Y_3_1_time, time,  color = 'firebrick', linestyle = '-', linewidth = 2.0, label = '$\overline{Y_3(Y_1^{+})}$')
graph_4	= ax.plot(Y_4_1_time, time,  color = 'cyan', linestyle = '-', linewidth = 2.0, label = '$\overline{Y_4(Y_1^{+})}$')
graph_5	= ax.plot(Y_1_2_time, time,  color = 'red', linestyle = '--', linewidth = 2.0, label = '$\overline{Y_1^{-}}$')
graph_6	= ax.plot(Y_2_2_time, time,  color = 'blue', linestyle = '--', linewidth = 2.0, label = '$\overline{Y_2(Y_1^{-})}$')
graph_7	= ax.plot(Y_3_2_time, time,  color = 'firebrick', linestyle = '--', linewidth = 2.0, label = '$\overline{Y_3(Y_1^{-})}$')
graph_8	= ax.plot(Y_4_2_time, time,  color = 'cyan', linestyle = '--', linewidth = 2.0, label = '$\overline{Y_4(Y_1^{-})}$')

ax.set_xlim(-6000, 6000)
ax.set_ylim(0, 15)
ax.set_yticks(np.arange(0, 15.1, 3))

ax.set_ylabel('Time')
ax.set_xlabel('$Y$ mean value')
ax.grid()
ax.set_title('f) $Y$ mean value')

graphs		= graph_1 + graph_2 + graph_3 + graph_4
legend_labels 	= [l.get_label() for l in graphs]
legend_1      	= ax.legend(graphs, legend_labels, loc = (0.80, 0.015), ncol=1, numpoints = 1, framealpha=1)

graphs		= graph_5 + graph_6 + graph_7 + graph_8
legend_labels 	= [l.get_label() for l in graphs]
legend_2      	= ax.legend(graphs, legend_labels, loc = (0.0075, 0.015), ncol=1, numpoints = 1, framealpha=1)

ax.add_artist(legend_1)
ax.add_artist(legend_2)

#-----------------------------------------------------------------------------
stream_1	= stream_mean + stream_all[0] * Y_1_1_time[-1] + stream_all[1] * Y_2_1_time[-1] + stream_all[2] * Y_3_1_time[-1] + stream_all[3] * Y_4_1_time[-1] 
stream_2	= stream_mean + stream_all[0] * Y_1_2_time[-1] + stream_all[1] * Y_2_2_time[-1] + stream_all[2] * Y_3_2_time[-1] + stream_all[3] * Y_4_2_time[-1] 
vor_1		= vor_mean + vor_all[0] * Y_1_1_time[-1] + vor_all[1] * Y_2_1_time[-1] + vor_all[2] * Y_3_1_time[-1] + vor_all[3] * Y_4_1_time[-1] 
vor_2		= vor_mean + vor_all[0] * Y_1_2_time[-1] + vor_all[1] * Y_2_2_time[-1] + vor_all[2] * Y_3_2_time[-1] + vor_all[3] * Y_4_2_time[-1] 


fig, ax	= subplots()
x_plot, y_plot	= np.meshgrid(x, y)

CS1  	= ax.contourf(x_plot, y_plot, vor_1, levels = np.arange(-300, 300.1, 15), extend = 'both', cmap = 'RdBu_r')
CS2	= ax.contour(x_plot, y_plot, stream_1, levels = np.arange(-2, 2.1, 0.5), colors = 'k')
CS2_0	= ax.contour(x_plot, y_plot, stream_1, levels = [0], colors = 'gray', linewidths = 3.0)
cbar 	= colorbar(CS1, ticks = np.arange(-300, 300.1, 100))
cbar.set_label(r'Vorticity')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('i) $v^{+}(t = '+str(time_end_plot)+')$, vorticity and streamfunction')
savefig('DO_spin_up_v_plus_Re_'+str(Re)+'_nV_'+str(nV)+'_sigma_1_nx_ny_64.png')		

fig, ax	= subplots()
x_plot, y_plot	= np.meshgrid(x, y)

CS1  	= ax.contourf(x_plot, y_plot, vor_2, levels = np.arange(-300, 300.1, 15), extend = 'both', cmap = 'RdBu_r')
CS2	= ax.contour(x_plot, y_plot, stream_2, levels = np.arange(-2, 2.1, 0.5), colors = 'k')
CS2_0	= ax.contour(x_plot, y_plot, stream_2, levels = [0], colors = 'gray', linewidths = 3.0)
cbar 	= colorbar(CS1, ticks = np.arange(-300, 300.1, 100))
cbar.set_label(r'Vorticity')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('i) $v^{-}(t = '+str(time_end_plot)+')$, vorticity and streamfunction')

#-----------------------------------------------------------------------------

bars, Y_1_dist_all	= DistributionYtime(time, Y_all[:, 0])
bars, Y_2_dist_all	= DistributionYtime(time, Y_all[:, 1])
bars, Y_3_dist_all	= DistributionYtime(time, Y_all[:, 2])
bars, Y_4_dist_all	= DistributionYtime(time, Y_all[:, 3])

#Now set the zeros to very low values for log plot
Y_1_dist_all[Y_1_dist_all == 0] = Y_1_dist_all[Y_1_dist_all == 0] + 10**(-10.0)
Y_2_dist_all[Y_2_dist_all == 0] = Y_2_dist_all[Y_2_dist_all == 0] + 10**(-10.0)
Y_3_dist_all[Y_3_dist_all == 0] = Y_3_dist_all[Y_3_dist_all == 0] + 10**(-10.0)
Y_4_dist_all[Y_4_dist_all == 0] = Y_4_dist_all[Y_4_dist_all == 0] + 10**(-10.0)

#-----------------------------------------------------------------------------

fig, ax	= subplots()

x_plot, y_plot	= np.meshgrid(bars, time)
CS	= pcolor(x_plot, y_plot, Y_1_dist_all, norm=colors.LogNorm(10**(-3), 10**0), cmap = 'Spectral_r', shading = 'auto')
cbar	= colorbar(CS, extend = 'min')
cbar.set_label('Probability distribution function')

ax.set_xlim(-6000, 6000)
ax.set_ylim(0, 15)
ax.set_yticks(np.arange(0, 15.1, 3))

ax.set_xlabel('$Y_1$')
ax.set_ylabel('Time')
ax.set_title('a) Probability distribution function of $Y_1(t)$')

#-----------------------------------------------------------------------------
fig, ax	= subplots()

CS	= pcolor(x_plot, y_plot, Y_2_dist_all, norm=colors.LogNorm(10**(-3), 10**0), cmap = 'Spectral_r', shading = 'auto')
cbar	= colorbar(CS, extend = 'min')
cbar.set_label('Probability distribution function')

ax.set_xlim(-6000, 6000)
ax.set_ylim(0, 15)
ax.set_yticks(np.arange(0, 15.1, 3))

ax.set_xlabel('$Y_2$')
ax.set_ylabel('Time')
ax.set_title('d) Probability distribution function of $Y_2(t)$')

#-----------------------------------------------------------------------------
fig, ax	= subplots()

CS	= pcolor(x_plot, y_plot, Y_3_dist_all, norm=colors.LogNorm(10**(-3), 10**0), cmap = 'Spectral_r', shading = 'auto')
cbar	= colorbar(CS, extend = 'min')
cbar.set_label('Probability distribution function')

ax.set_xlim(-6000, 6000)
ax.set_ylim(0, 15)
ax.set_yticks(np.arange(0, 15.1, 3))

ax.set_xlabel('$Y_3$')
ax.set_ylabel('Time')
ax.set_title('g) Probability distribution function of $Y_3(t)$')

#-----------------------------------------------------------------------------
fig, ax	= subplots()

CS	= pcolor(x_plot, y_plot, Y_4_dist_all, norm=colors.LogNorm(10**(-3), 10**0), cmap = 'Spectral_r', shading = 'auto')
cbar	= colorbar(CS, extend = 'min')
cbar.set_label('Probability distribution function')

ax.set_xlim(-6000, 6000)
ax.set_ylim(0, 15)
ax.set_yticks(np.arange(0, 15.1, 3))

ax.set_xlabel('$Y_4$')
ax.set_ylabel('Time')
ax.set_title('j) Probability distribution function of $Y_4(t)$')

#-----------------------------------------------------------------------------
fig, ax	= subplots()
x_plot, y_plot	= np.meshgrid(x, y)

CS1  	= ax.contourf(x_plot, y_plot, vor_all[0] * 10**2.0, levels = np.arange(-5, 5.1, 0.25), extend = 'both', cmap = 'RdBu_r')
CS2	= ax.contour(x_plot, y_plot, stream_all[0] * 10**4.0, levels = np.arange(-2, 2.1, 1), colors = 'k')
CS2_0	= ax.contour(x_plot, y_plot, stream_all[0] * 10**4.0, levels = [0], colors = 'gray', linewidths = 3.0)
cbar 	= colorbar(CS1, ticks = np.arange(-5, 5.1, 1))
cbar.set_label(r'Vorticity ($\times 10^{-2}$)')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('b) $v_1(t = '+str(time_end_plot)+')$, vorticity and streamfunction')

#-----------------------------------------------------------------------------
fig, ax	= subplots()

CS1  	= ax.contourf(x_plot, y_plot, vor_all[1] * 10**2.0, levels = np.arange(-5, 5.1, 0.25), extend = 'both', cmap = 'RdBu_r')
CS2	= ax.contour(x_plot, y_plot, stream_all[1] * 10**4.0, levels = np.arange(-2, 2.1, 1), colors = 'k')
CS2_0	= ax.contour(x_plot, y_plot, stream_all[1] * 10**4.0, levels = [0], colors = 'gray', linewidths = 3.0)
cbar 	= colorbar(CS1, ticks = np.arange(-5, 5.1, 1))
cbar.set_label(r'Vorticity ($\times 10^{-2}$)')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('e) $v_2(t = '+str(time_end_plot)+')$, vorticity and streamfunction')

#-----------------------------------------------------------------------------
fig, ax	= subplots()

CS1  	= ax.contourf(x_plot, y_plot, vor_all[2] * 10**2.0, levels = np.arange(-5, 5.1, 0.25), extend = 'both', cmap = 'RdBu_r')
CS2	= ax.contour(x_plot, y_plot, stream_all[2] * 10**4.0, levels = np.arange(-2, 2.1, 1), colors = 'k')
CS2_0	= ax.contour(x_plot, y_plot, stream_all[2] * 10**4.0, levels = [0], colors = 'gray', linewidths = 3.0)
cbar 	= colorbar(CS1, ticks = np.arange(-5, 5.1, 1))
cbar.set_label(r'Vorticity ($\times 10^{-2}$)')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('h) $v_3(t = '+str(time_end_plot)+')$, vorticity and streamfunction')

#-----------------------------------------------------------------------------
fig, ax	= subplots()

CS1  	= ax.contourf(x_plot, y_plot, vor_all[3] * 10**2.0, levels = np.arange(-5, 5.1, 0.25), extend = 'both', cmap = 'RdBu_r')
CS2	= ax.contour(x_plot, y_plot, stream_all[3] * 10**4.0, levels = np.arange(-2, 2.1, 1), colors = 'k')
CS2_0	= ax.contour(x_plot, y_plot, stream_all[3] * 10**4.0, levels = [0], colors = 'gray', linewidths = 3.0)
cbar 	= colorbar(CS1, ticks = np.arange(-5, 5.1, 1))
cbar.set_label(r'Vorticity ($\times 10^{-2}$)')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('k) $v_4(t = '+str(time_end_plot)+')$, vorticity and streamfunction')

show()

