#Program plots the Y_i coefficients for various noise levels

from pylab import *
import numpy as np
import netCDF4 as netcdf

#Directory for the output
directory = '../Data/QG/'

#-----------------------------------------------------------------------------
#---------------------------MAIN SCRIPT STARTS HERE---------------------------
#-----------------------------------------------------------------------------	

nx          = 64 #Number of zonal dimensions
ny          = 64 #Number of meridional dimensions
nV          = 4  #Number of DO modes
Re	        = 40
delta_t	    = 0.0005
stoch_amp   = 15

#-----------------------------------------------------------------------------

fh = netcdf.Dataset(directory+'QG_sorted_basis_nx_'+str(nx)+'_ny_'+str(ny)+'_dt_'+str(int(delta_t))+'_'+str(delta_t)[2:]+'_nV_'+str(nV)+'_Re_'+str(Re)+'_Stoch_amp_'+str(stoch_amp)+'_transition.nc', 'r')

time 		= fh.variables['time'][:] 
x		    = fh.variables['x'][:] 			
y		    = fh.variables['y'][:] 		 
stream_all	= fh.variables['stream_V'][:]	
vor_all		= fh.variables['vor_V'][:]			
Y_1		    = fh.variables['Y'][:, 0] 
Y_2		    = fh.variables['Y'][:, 1] 
Y_3		    = fh.variables['Y'][:, 2] 
Y_4		    = fh.variables['Y'][:, 3] 

fh.close()

#-----------------------------------------------------------------------------

time_index	    = (fabs(time - 1)).argmin()
stream_all_1	= stream_all[time_index]
vor_all_1	    = vor_all[time_index]
time_index	    = (fabs(time - 2.5)).argmin()
stream_all_2	= stream_all[time_index]
vor_all_2	    = vor_all[time_index]
time_index	    = (fabs(time - 4)).argmin()
stream_all_3	= stream_all[time_index]
vor_all_3	    = vor_all[time_index]

#-----------------------------------------------------------------------------

#Arrays for mean and 95%-Cl
Y_1_plus	= np.zeros((len(time), 4))
Y_1_min		= np.zeros((len(time), 4))
Y_2_plus	= np.zeros((len(time), 4))
Y_2_min		= np.zeros((len(time), 4))
Y_3_plus	= np.zeros((len(time), 4))
Y_3_min		= np.zeros((len(time), 4))
	
for time_i in range(len(time)):
	print(time_i)

	#Get the two branches
	Y_1_time	    = Y_1[time_i]
	Y_1_time	    = Y_1_time[Y_1_time.mask == False]
	Y_plus_index	= np.where(Y_1_time >= 0)[0]
	Y_min_index	    = np.where(Y_1_time <= 0)[0]
	Y_1_time_plus	= Y_1_time[Y_plus_index]
	Y_1_time_min	= Y_1_time[Y_min_index]
	Y_2_time_plus	= Y_2[time_i, Y_plus_index]
	Y_2_time_min	= Y_2[time_i, Y_min_index]
	Y_3_time_plus	= Y_3[time_i, Y_plus_index]
	Y_3_time_min	= Y_3[time_i, Y_min_index]

	#Get the mean and 95%-Cl
	Y_1_plus[time_i]= np.mean(Y_1_time_plus), np.percentile(Y_1_time_plus, 2.5), np.percentile(Y_1_time_plus, 97.5), np.min(Y_1_time_plus)
	Y_1_min[time_i]	= np.mean(Y_1_time_min), np.percentile(Y_1_time_min, 2.5), np.percentile(Y_1_time_min, 97.5), np.max(Y_1_time_min)
	Y_2_plus[time_i]= np.mean(Y_2_time_plus), np.percentile(Y_2_time_plus, 2.5), np.percentile(Y_2_time_plus, 97.5), np.max(Y_2_time_plus)
	Y_2_min[time_i]	= np.mean(Y_2_time_min), np.percentile(Y_2_time_min, 2.5), np.percentile(Y_2_time_min, 97.5), np.max(Y_2_time_min)
	Y_3_plus[time_i]= np.mean(Y_3_time_plus), np.percentile(Y_3_time_plus, 2.5), np.percentile(Y_3_time_plus, 97.5), np.max(Y_3_time_plus)
	Y_3_min[time_i]	= np.mean(Y_3_time_min), np.percentile(Y_3_time_min, 2.5), np.percentile(Y_3_time_min, 97.5), np.max(Y_3_time_min)

#-----------------------------------------------------------------------------

#Get the initial values of Y at t = 0
time_index	    = (fabs(time - 0.0)).argmin()
Y_1_start	    = Y_1[time_index]
Y_plus_index	= np.where(Y_1_start >= 0)[0]
Y_min_index	    = np.where(Y_1_start <= 0)[0]
Y_plus_start	= Y_1_start[Y_plus_index]
Y_min_start  	= Y_1_start[Y_min_index]

print('Y plus:', len(Y_plus_index))
print('Y min:', len(Y_min_index))

Y_plus_trans_min	= ma.masked_all(len(time))
Y_min_trans_max		= ma.masked_all(len(time))
Y_plus_diff_min		= np.zeros(len(Y_plus_index))
Y_min_diff_min		= np.zeros(len(Y_min_index))

for Y_i in range(len(Y_plus_diff_min)):
	#Get the minimum difference for each Y_1+ trajectory 
	Y_plus_diff_min[Y_i]	= np.min(Y_1[time_index:, Y_plus_index[Y_i]] - Y_1_min[time_index:, 0])

for Y_i in range(len(Y_min_diff_min)):
	#Get the minimum difference for each Y_1- trajectory 
	Y_min_diff_min[Y_i]	= np.min(Y_1_plus[time_index:, 0] - Y_1[time_index:, Y_min_index[Y_i]])


for time_i in range(time_index, len(time)):
	#Select trajectories closest to other branch, using the starting indices
	Y_1_plus_time		= Y_1[time_i, Y_plus_index]
	Y_1_min_time		= Y_1[time_i, Y_min_index]
	Y_2_plus_time		= Y_2[time_i, Y_plus_index]
	Y_2_min_time		= Y_2[time_i, Y_min_index]
	Y_3_plus_time		= Y_3[time_i, Y_plus_index]
	Y_3_min_time		= Y_3[time_i, Y_min_index]

	#Save the maximum and minimum
	Y_1_plus[time_i, 3]	= np.min(Y_1_plus_time)
	Y_1_min[time_i, 3]	= np.max(Y_1_min_time)
	Y_2_plus[time_i, 3]	= np.max(Y_2_plus_time)
	Y_2_min[time_i, 3]	= np.max(Y_2_min_time)
	Y_3_plus[time_i, 3]	= np.max(Y_3_plus_time)
	Y_3_min[time_i, 3]	= np.max(Y_3_min_time)

sorted_plus_index	= np.argsort(Y_plus_diff_min)
sorted_min_index	= np.argsort(Y_min_diff_min)

#Now get the indices for the entire trajectories
sorted_plus_index	= Y_plus_index[sorted_plus_index]
sorted_min_index	= Y_min_index[sorted_min_index]

#-----------------------------------------------------------------------------

fig, ax	= subplots()

ax.fill_between(time, Y_3_plus[:, 1], Y_3_plus[:, 2], facecolor = 'red', alpha = 0.2)
ax.fill_between(time, Y_3_min[:, 1], Y_3_min[:, 2], facecolor = 'blue', alpha = 0.2)

#Now plot the minimum and maximum (of all trajectories)
graph_1	= ax.plot(time, Y_3_plus[:, 0], '-r', linewidth = 2.0, label = r'$\overline{Y_3^{+}}$')
graph_2	= ax.plot(time, Y_3_min[:, 0], '-b', linewidth = 2.0, label = r'$\overline{Y_3^{-}}$')

graph_3	= ax.plot(time, Y_3_plus[:, 3], '--r', linewidth = 1.5, label = r'$Y_{3,\mathrm{max}}^{+}$')
graph_4	= ax.plot(time, Y_3_min[:, 3], '--b', linewidth = 1.5, label = r'$Y_{3,\mathrm{max}}^{-}$')

ax.set_xlim(-0.5, 5)
ax.set_ylim(-9000, 9000)
ax.set_xlabel('Time')
ax.set_ylabel('$Y_3$')
ax.grid()


graphs		= graph_1 + graph_2 + graph_3 + graph_4
legend_labels 	= [l.get_label() for l in graphs]
legend      	= ax.legend(graphs, legend_labels, loc = (0.70, 0.90), ncol=2, numpoints = 1, framealpha=1)

ax.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')
ax.text(-0.2, 4000, 'Spin-up', verticalalignment='center', horizontalalignment='center', color = 'k', fontsize=14, rotation=90)

ax2 	= fig.add_axes([0.21, 0.125, 0.21, 0.24])

x_plot, y_plot	= np.meshgrid(x, y)
CS1  	= ax2.contourf(x_plot, y_plot, vor_all_1[2] * 10**2.0, levels = np.arange(-5, 5.1, 0.25), extend = 'both', cmap = 'RdBu_r')
CS1_a	= ax2.contour(x_plot, y_plot, stream_all_1[2] * 10**4.0, levels = np.arange(-2, 2.1, 1), colors = 'k')
CS1_b	= ax2.contour(x_plot, y_plot, stream_all_1[2] * 10**4.0, levels = [0], colors = 'gray', linewidths = 3.0)

ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_title('$v_3(t = 1)$')

ax3 	= fig.add_axes([0.445, 0.125, 0.21, 0.24])

x_plot, y_plot	= np.meshgrid(x, y)
CS2  	= ax3.contourf(x_plot, y_plot, vor_all_2[2] * 10**2.0, levels = np.arange(-5, 5.1, 0.25), extend = 'both', cmap = 'RdBu_r')
CS2_a	= ax3.contour(x_plot, y_plot, stream_all_2[2] * 10**4.0, levels = np.arange(-2, 2.1, 1), colors = 'k')
CS2_b	= ax3.contour(x_plot, y_plot, stream_all_2[2] * 10**4.0, levels = [0], colors = 'gray', linewidths = 3.0)

ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_title('$v_3(t = 2.5)$')

ax4 	= fig.add_axes([0.68, 0.125, 0.21, 0.24])

x_plot, y_plot	= np.meshgrid(x, y)
CS3  	= ax4.contourf(x_plot, y_plot, vor_all_3[2] * 10**2.0, levels = np.arange(-5, 5.1, 0.25), extend = 'both', cmap = 'RdBu_r')
CS3_a	= ax4.contour(x_plot, y_plot, stream_all_3[2] * 10**4.0, levels = np.arange(-2, 2.1, 1), colors = 'k')
CS3_b	= ax4.contour(x_plot, y_plot, stream_all_3[2] * 10**4.0, levels = [0], colors = 'gray', linewidths = 3.0)

ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_title('$v_3(t = 4)$')

if Re == 40 and nV == 4 and stoch_amp == 1 and nx == 64:

	ax5 = fig.add_axes([0.30, 0.59, 0.55, 0.20])

	ax5.fill_between(time, Y_3_plus[:, 1], Y_3_plus[:, 2], facecolor = 'red', alpha = 0.2)
	ax5.plot(time, Y_3_plus[:, 0], '-r', linewidth = 2.0)
	ax5.plot(time, Y_3_plus[:, 3], '--r', linewidth = 1.5)
	ax5.fill_between(time, Y_3_min[:, 1], Y_3_min[:, 2], facecolor = 'blue', alpha = 0.2)
	ax5.plot(time, Y_3_min[:, 0], '-b', linewidth = 2.0)
	ax5.plot(time, Y_3_min[:, 3], '--b', linewidth = 1.5)

	ax5.set_xlim(-0.5, 5)
	ax5.set_ylim(-150, 150)
	ax5.grid()
	ax5.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')

	ax.set_title('c) $Y_3$, $\sigma = 1$')

if Re == 40 and nV == 4 and stoch_amp == 5 and nx == 64:

	ax5 = fig.add_axes([0.30, 0.59, 0.55, 0.20])

	ax5.fill_between(time, Y_3_plus[:, 1], Y_3_plus[:, 2], facecolor = 'red', alpha = 0.2)
	ax5.plot(time, Y_3_plus[:, 0], '-r', linewidth = 2.0)
	ax5.plot(time, Y_3_plus[:, 3], '--r', linewidth = 1.5)
	ax5.fill_between(time, Y_3_min[:, 1], Y_3_min[:, 2], facecolor = 'blue', alpha = 0.2)
	ax5.plot(time, Y_3_min[:, 0], '-b', linewidth = 2.0)
	ax5.plot(time, Y_3_min[:, 3], '--b', linewidth = 1.5)

	ax5.set_xlim(-0.5, 5)
	ax5.set_ylim(-500, 500)
	ax5.grid()
	ax5.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')

	ax.set_title('f) $Y_3$, $\sigma = 5$')

if Re == 40 and nV == 4 and stoch_amp == 10 and nx == 64:
	ax.set_title('i) $Y_3$, $\sigma = 10$')

if Re == 40 and nV == 4 and stoch_amp == 15 and nx == 64:

	ax.plot(time[time_index:], Y_3[time_index:, sorted_plus_index[1]], linestyle = '-', color = 'firebrick', linewidth = 1.0)
	ax.plot(time[time_index:], Y_3[time_index:, sorted_plus_index[2]], linestyle = '-', color = 'orangered', linewidth = 1.0)

	ax.plot(time[time_index:], Y_3[time_index:, sorted_min_index[1]], linestyle = '-', color = 'deepskyblue', linewidth = 1.0)
	ax.plot(time[time_index:], Y_3[time_index:, sorted_min_index[2]], linestyle = '-', color = 'dodgerblue', linewidth = 1.0)

	ax.set_title('l) $Y_3$, $\sigma = 15$')

#-----------------------------------------------------------------------------
fig, ax	= subplots()

ax.fill_between(time, Y_2_plus[:, 1], Y_2_plus[:, 2], facecolor = 'red', alpha = 0.2)
ax.fill_between(time, Y_2_min[:, 1], Y_2_min[:, 2], facecolor = 'blue', alpha = 0.2)

#Now plot the minimum and maximum (of all trajectories)
graph_1	= ax.plot(time, Y_2_plus[:, 0], '-r', linewidth = 2.0, label = r'$\overline{Y_2^{+}}$')
graph_2	= ax.plot(time, Y_2_min[:, 0], '-b', linewidth = 2.0, label = r'$\overline{Y_2^{-}}$')

graph_3	= ax.plot(time, Y_2_plus[:, 3], '--r', linewidth = 1.5, label = r'$Y_{2,\mathrm{max}}^{+}$')
graph_4	= ax.plot(time, Y_2_min[:, 3], '--b', linewidth = 1.5, label = r'$Y_{2,\mathrm{max}}^{-}$')

ax.set_xlim(-0.5, 5)
ax.set_ylim(-9000, 9000)
ax.set_xlabel('Time')
ax.set_ylabel('$Y_2$')
ax.grid()

graphs		= graph_1 + graph_2 + graph_3 + graph_4
legend_labels 	= [l.get_label() for l in graphs]
legend      	= ax.legend(graphs, legend_labels, loc = (0.01, 0.90), ncol=2, numpoints = 1, framealpha=1)

ax.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')
ax.text(-0.2, 4000, 'Spin-up', verticalalignment='center', horizontalalignment='center', color = 'k', fontsize=14, rotation=90)

ax2 	= fig.add_axes([0.21, 0.125, 0.21, 0.24])

x_plot, y_plot	= np.meshgrid(x, y)
CS1  	= ax2.contourf(x_plot, y_plot, vor_all_1[1] * 10**2.0, levels = np.arange(-5, 5.1, 0.25), extend = 'both', cmap = 'RdBu_r')
CS1_a	= ax2.contour(x_plot, y_plot, stream_all_1[1] * 10**4.0, levels = np.arange(-2, 2.1, 1), colors = 'k')
CS1_b	= ax2.contour(x_plot, y_plot, stream_all_1[1] * 10**4.0, levels = [0], colors = 'gray', linewidths = 3.0)

ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_title('$v_2(t = 1)$')

ax3 	= fig.add_axes([0.445, 0.125, 0.21, 0.24])

x_plot, y_plot	= np.meshgrid(x, y)
CS2  	= ax3.contourf(x_plot, y_plot, vor_all_2[1] * 10**2.0, levels = np.arange(-5, 5.1, 0.25), extend = 'both', cmap = 'RdBu_r')
CS2_a	= ax3.contour(x_plot, y_plot, stream_all_2[1] * 10**4.0, levels = np.arange(-2, 2.1, 1), colors = 'k')
CS2_b	= ax3.contour(x_plot, y_plot, stream_all_2[1] * 10**4.0, levels = [0], colors = 'gray', linewidths = 3.0)

ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_title('$v_2(t = 2.5)$')

ax4 	= fig.add_axes([0.68, 0.125, 0.21, 0.24])

x_plot, y_plot	= np.meshgrid(x, y)
CS3  	= ax4.contourf(x_plot, y_plot, vor_all_3[1] * 10**2.0, levels = np.arange(-5, 5.1, 0.25), extend = 'both', cmap = 'RdBu_r')
CS3_a	= ax4.contour(x_plot, y_plot, stream_all_3[1] * 10**4.0, levels = np.arange(-2, 2.1, 1), colors = 'k')
CS3_b	= ax4.contour(x_plot, y_plot, stream_all_3[1] * 10**4.0, levels = [0], colors = 'gray', linewidths = 3.0)

ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_title('$v_2(t = 4)$')

if Re == 40 and nV == 4 and stoch_amp == 1 and nx == 64:

	ax5 = fig.add_axes([0.30, 0.59, 0.55, 0.20])

	ax5.fill_between(time, Y_2_plus[:, 1], Y_2_plus[:, 2], facecolor = 'red', alpha = 0.2)
	ax5.plot(time, Y_2_plus[:, 0], '-r', linewidth = 2.0)
	ax5.plot(time, Y_2_plus[:, 3], '--r', linewidth = 1.5)
	ax5.fill_between(time, Y_2_min[:, 1], Y_2_min[:, 2], facecolor = 'blue', alpha = 0.2)
	ax5.plot(time, Y_2_min[:, 0], '-b', linewidth = 2.0)
	ax5.plot(time, Y_2_min[:, 3], '--b', linewidth = 1.5)

	ax5.set_xlim(-0.5, 5)
	ax5.set_ylim(-300, 300)
	ax5.grid()
	ax5.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')

	ax.set_title('b) $Y_2$, $\sigma = 1$')

if Re == 40 and nV == 4 and stoch_amp == 5 and nx == 64:
	
	ax5 = fig.add_axes([0.30, 0.59, 0.55, 0.20])

	ax5.fill_between(time, Y_2_plus[:, 1], Y_2_plus[:, 2], facecolor = 'red', alpha = 0.2)
	ax5.plot(time, Y_2_plus[:, 0], '-r', linewidth = 2.0)
	ax5.plot(time, Y_2_plus[:, 3], '--r', linewidth = 1.5)
	ax5.fill_between(time, Y_2_min[:, 1], Y_2_min[:, 2], facecolor = 'blue', alpha = 0.2)
	ax5.plot(time, Y_2_min[:, 0], '-b', linewidth = 2.0)
	ax5.plot(time, Y_2_min[:, 3], '--b', linewidth = 1.5)

	ax5.set_xlim(-0.5, 5)
	ax5.set_ylim(-1500, 1500)
	ax5.grid()
	ax5.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')

	ax.set_title('e) $Y_2$, $\sigma = 5$')

if Re == 40 and nV == 4 and stoch_amp == 10 and nx == 64:
	ax.set_title('h) $Y_2$, $\sigma = 10$')

if Re == 40 and nV == 4 and stoch_amp == 15 and nx == 64:

	ax.plot(time[time_index:], Y_2[time_index:, sorted_plus_index[1]], linestyle = '-', color = 'firebrick', linewidth = 1.0)
	ax.plot(time[time_index:], Y_2[time_index:, sorted_plus_index[2]], linestyle = '-', color = 'orangered', linewidth = 1.0)

	ax.plot(time[time_index:], Y_2[time_index:, sorted_min_index[1]], linestyle = '-', color = 'deepskyblue', linewidth = 1.0)
	ax.plot(time[time_index:], Y_2[time_index:, sorted_min_index[2]], linestyle = '-', color = 'dodgerblue', linewidth = 1.0)

	ax.set_title('k) $Y_2$, $\sigma = 15$')

#-----------------------------------------------------------------------------

fig, ax	= subplots()

ax.fill_between(time, Y_1_plus[:, 1], Y_1_plus[:, 2], facecolor = 'red', alpha = 0.2)
ax.fill_between(time, Y_1_min[:, 1], Y_1_min[:, 2], facecolor = 'blue', alpha = 0.2)

graph_1	= ax.plot(time, Y_1_plus[:, 0], '-r', linewidth = 2.0, label = r'$\overline{Y_1^{+}}$')
graph_2	= ax.plot(time, Y_1_min[:, 0], '-b', linewidth = 2.0, label = r'$\overline{Y_1^{-}}$')

#Now plot the minimum and maximum (of all trajectories)
graph_3	= ax.plot(time, Y_1_plus[:, 3], '--r', linewidth = 1.5, label = r'$Y_{1,\mathrm{min}}^{+}$')
graph_4	= ax.plot(time, Y_1_min[:, 3], '--b', linewidth = 1.5, label = r'$Y_{1,\mathrm{max}}^{-}$')

ax.set_xlim(-0.5, 5)
ax.set_ylim(-6000, 6000)
ax.set_xlabel('Time')
ax.set_ylabel('$Y_1$')
ax.grid()

graphs		= graph_1 + graph_2 + graph_3 + graph_4
legend_labels 	= [l.get_label() for l in graphs]
legend      	= ax.legend(graphs, legend_labels, loc = (0.70, 0.90), ncol=2, numpoints = 1, framealpha=1)

ax.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')
ax.text(-0.2, 0, 'Spin-up', verticalalignment='center', horizontalalignment='center', color = 'k', fontsize=14, rotation=90)

if Re == 40 and nV == 4 and stoch_amp == 1 and nx == 64:

	ax2 = fig.add_axes([0.30, 0.51, 0.55, 0.15])

	ax2.fill_between(time, Y_1_plus[:, 1], Y_1_plus[:, 2], facecolor = 'red', alpha = 0.2)
	ax2.plot(time, Y_1_plus[:, 0], '-r', linewidth = 2.0, label = r'$\overline{Y_1^{+}}$')
	ax2.plot(time, Y_1_plus[:, 3], '--r', linewidth = 1.5, label = r'$Y_{1,\mathrm{min}}^{+}$')

	ax2.set_xlim(-0.5, 5)
	ax2.set_ylim(3240, 3360)
	ax2.grid()
	ax2.set_xticklabels([])
	ax2.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')

	ax3 = fig.add_axes([0.30, 0.33, 0.55, 0.15])

	ax3.fill_between(time, Y_1_min[:, 1], Y_1_min[:, 2], facecolor = 'blue', alpha = 0.2)
	ax3.plot(time, Y_1_min[:, 0], '-b', linewidth = 2.0, label = r'$\overline{Y_1^{-}}$')
	ax3.plot(time, Y_1_min[:, 3], '--b', linewidth = 1.5, label = r'$Y_{1,\mathrm{max}}^{-}$')

	ax3.set_xlim(-0.5, 5)
	ax3.set_ylim(-3660, -3540)
	ax3.grid()
	ax3.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')

	ax.set_title('a) $Y_1$, $\sigma = 1$')

if Re == 40 and nV == 4 and stoch_amp == 5 and nx == 64:

	ax2 = fig.add_axes([0.30, 0.51, 0.55, 0.15])

	ax2.fill_between(time, Y_1_plus[:, 1], Y_1_plus[:, 2], facecolor = 'red', alpha = 0.2)
	ax2.plot(time, Y_1_plus[:, 0], '-r', linewidth = 2.0, label = r'$\overline{Y_1^{+}}$')
	ax2.plot(time, Y_1_plus[:, 3], '--r', linewidth = 1.5, label = r'$Y_{1,\mathrm{min}}^{+}$')

	ax2.set_xlim(-0.5, 5)
	ax2.set_ylim(2900, 3500)
	ax2.grid()
	ax2.set_xticklabels([])
	ax2.set_yticks([2950, 3200, 3450])
	ax2.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')

	ax3 = fig.add_axes([0.30, 0.33, 0.55, 0.15])

	ax3.fill_between(time, Y_1_min[:, 1], Y_1_min[:, 2], facecolor = 'blue', alpha = 0.2)
	ax3.plot(time, Y_1_min[:, 0], '-b', linewidth = 2.0, label = r'$\overline{Y_1^{-}}$')
	ax3.plot(time, Y_1_min[:, 3], '--b', linewidth = 1.5, label = r'$Y_{1,\mathrm{max}}^{-}$')

	ax3.set_xlim(-0.5, 5)
	ax3.set_ylim(-3800, -3200)
	ax3.set_yticks([-3750, -3500, -3250])
	ax3.grid()
	ax3.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')

	ax.set_title('d) $Y_1$, $\sigma = 5$')

if Re == 40 and nV == 4 and stoch_amp == 10 and nx == 64:
	ax.set_title('g) $Y_1$, $\sigma = 10$')

if Re == 40 and nV == 4 and stoch_amp == 15 and nx == 64:

	#Now plot three individual trajectories, closest to the other branch
	ax.plot(time[time_index:], Y_1[time_index:, sorted_plus_index[1]], linestyle = '-', color = 'firebrick', linewidth = 1.0)
	ax.plot(time[time_index:], Y_1[time_index:, sorted_plus_index[2]], linestyle = '-', color = 'orangered', linewidth = 1.0)

	ax.plot(time[time_index:], Y_1[time_index:, sorted_min_index[1]], linestyle = '-', color = 'deepskyblue', linewidth = 1.0)
	ax.plot(time[time_index:], Y_1[time_index:, sorted_min_index[2]], linestyle = '-', color = 'dodgerblue', linewidth = 1.0)

	ax.set_title('i) $Y_1$, $\sigma = 15$')

show()

#-----------------------------------------------------------------------------

