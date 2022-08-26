#Program plots the transition paths
#Tip: Provide the sorted basis with the same temporal resolution as the original file
#to obtain nice smooth transition paths and the correct transition probabilities

from pylab import *
import numpy as np
import netCDF4 as netcdf
from matplotlib.collections import LineCollection

#Directory for the output
directory = '../Data/QG/'

#-----------------------------------------------------------------------------
#---------------------------MAIN SCRIPT STARTS HERE---------------------------
#-----------------------------------------------------------------------------	

nx          = 64 #Numbber of zonal dimensions
ny          = 64 #Number of meridional dimensions
nV          = 4  #Number of DO modes
Re	        = 40
delta_t	    = 0.0005
stoch_amp   = 15

#-----------------------------------------------------------------------------

fh = netcdf.Dataset(directory+'QG_sorted_basis_nx_'+str(nx)+'_ny_'+str(ny)+'_dt_'+str(int(delta_t))+'_'+str(delta_t)[2:]+'_nV_'+str(nV)+'_Re_'+str(Re)+'_Stoch_amp_'+str(stoch_amp)+'_transition.nc', 'r')

time 		= fh.variables['time'][:]     	
Y_all		= fh.variables['Y'][:]

fh.close()

#Start from t = 0.0
time_start_index = (np.abs(time - 0.0)).argmin()
time             = time[time_start_index:]
Y_all            = Y_all[time_start_index:]

#-----------------------------------------------------------------------------

Y_1_plus	= np.zeros((len(time), 3))
Y_1_min	    = np.zeros((len(time), 3))

for time_i in range(len(time)):
	#Get the two branches
	Y_1_time	    = Y_all[time_i, 0]
	Y_plus_index	= np.where(Y_1_time >= 0)[0]
	Y_min_index	    = np.where(Y_1_time <= 0)[0]
	Y_1_time_plus	= Y_1_time[Y_plus_index]
	Y_1_time_min	= Y_1_time[Y_min_index]

	#Get the mean and 95%-Cl
	Y_1_plus[time_i]= np.mean(Y_1_time_plus), np.percentile(Y_1_time_plus, 2.5), np.percentile(Y_1_time_plus, 97.5)
	Y_1_min[time_i]	= np.mean(Y_1_time_min), np.percentile(Y_1_time_min, 2.5), np.percentile(Y_1_time_min, 97.5)

#-----------------------------------------------------------------------------

Y_plus_to_min	= np.zeros((1, 2))
Y_min_to_plus	= np.zeros((1, 2))

for Y_i in range(len(Y_all[0, 0])):
	#Loop over each realisation
	Y_1		= Y_all[:, 0, Y_i]
	trans_index	= np.argwhere(np.diff(np.sign(Y_1))).flatten()

	if len(trans_index) == 0: 
		continue

	#Now get the first point in time where it switches sign
	time_index	= trans_index[0]

	if Y_1[0] >= 0:
		#Positive Y-branch
		if np.all(Y_1 >= Y_1_min[:, 2]):
			#No transition to 95%-Cl branch, skip realisation
			continue

		#Get the first time where it makes the transition to the 95%-Cl
		time_index	= np.where(Y_1 <= Y_1_min[:, 2])[0][0]

		if np.any(Y_1[time_index:] >= Y_1_plus[time_index:, 1]):
			#Ends up in the original branch
			Y_plus_to_min    	= np.concatenate((Y_plus_to_min, [[Y_i, 2]]), axis = 0)

		else:
			#Check whether it ends up in the other branch
			#Trajectory makes a full transition, becomes part of the new branch
			if np.any(Y_1[time_index:] <= Y_1_min[time_index:, 0]):
				Y_plus_to_min    	= np.concatenate((Y_plus_to_min, [[Y_i, 1]]), axis = 0)

			else:
				#Trajectory is wandering around (inside or outside the 95%-Cl)
				Y_plus_to_min    	= np.concatenate((Y_plus_to_min, [[Y_i, 3]]), axis = 0)


	if Y_1[0] <= 0:
		#Negative Y-branch
		if np.all(Y_1 <= Y_1_plus[:, 1]):
			#No transition to 95%-Cl branch, skip realisation
			continue

		#Get the first time where it makes the transition to the 95%-Cl
		time_index	= np.where(Y_1 >= Y_1_plus[:, 1])[0][0]

		if np.any(Y_1[time_index:] <= Y_1_min[time_index:, 2]):
			#Ends up in the original branch
			Y_min_to_plus    	= np.concatenate((Y_min_to_plus, [[Y_i, 2]]), axis = 0)

		else:
			#Check whether it ends up in the other branch
			#Trajectory makes a full transition, becomes part of the new branch
			if np.any(Y_1[time_index:] >= Y_1_plus[time_index:, 0]):
				Y_min_to_plus    	= np.concatenate((Y_min_to_plus, [[Y_i, 1]]), axis = 0)

			else:
				#Trajectory is wandering around (inside or outside the 95%-Cl)
				Y_min_to_plus    	= np.concatenate((Y_min_to_plus, [[Y_i, 3]]), axis = 0)

#Remove the first empty input
Y_plus_to_min	= Y_plus_to_min[1:]
Y_min_to_plus	= Y_min_to_plus[1:]

#-----------------------------------------------------------------------------

fig, ax1	= subplots()

graph_1	= ax1.plot(time, Y_1_plus[:, 0], '-r', linewidth = 2.0, label = '$\overline{Y_1^{+}}$')
graph_2	= ax1.plot(time, Y_1_min[:, 0], '-b', linewidth = 2.0, label = '$\overline{Y_1^{-}}$')

ax1.fill_between(time, Y_1_plus[:, 1], Y_1_plus[:, 2], facecolor = 'red', alpha = 0.2)
ax1.fill_between(time, Y_1_min[:, 1], Y_1_min[:, 2], facecolor = 'blue', alpha = 0.2)

counter_1, counter_2, counter_3 = 0, 0, 0

for Y_i in range(len(Y_plus_to_min)):
    #Get the relevant realisations
	Y_i, trans_number	= Y_plus_to_min[Y_i]
	Y_1		            = Y_all[:, 0, int(Y_i)]

	if trans_number == 1:
		#Transition to other branch
		ax1.plot(time, Y_1, linewidth = 0.5, linestyle = '-', color = 'k')
		counter_1 += 1

	if trans_number == 2:
		#Returns to the original branch
		ax1.plot(time, Y_1, linewidth = 0.5, linestyle = '-', color = 'c')
		counter_2 += 1

	if trans_number == 3:
		#Transition, no sufficient information
		counter_3 += 1

ax1.set_xlim(0, 5)
ax1.set_ylim(-6000, 6000)
ax1.set_xlabel('Time')
ax1.set_ylabel('$Y_1$')
ax1.grid()

graph_3		= ax1.plot([-10000, -10000], [-10000, -10000], '-k', linewidth = 2.0, label = r'$Y_1^{+} \rightarrow Y_1^{-}$')
graph_4		= ax1.plot([-10000, -10000], [-10000, -10000], '-c', linewidth = 2.0, label = r'$Y_1^{+} \rightarrow Y_1^{-} \rightarrow Y_1^{+}$')

graphs		    = graph_1 + graph_2 + graph_3 + graph_4
legend_labels 	= [l.get_label() for l in graphs]
legend      	= ax1.legend(graphs, legend_labels, loc = (0.65, 0.90), ncol=2, numpoints = 1, framealpha=1)

ax1.text(4.85, -5000, str(counter_1), verticalalignment='center', horizontalalignment='right', color = 'k', fontsize=14)
ax1.text(4.85, -5600, str(counter_2), verticalalignment='center', horizontalalignment='right', color = 'c', fontsize=14)

ax1.set_title(r'a) $Y_1^{+} \rightarrow Y_1^{-}$')

if counter_3 != 0:
	print('Y+ to Y-:', counter_3, 'transitions are not defined well, consider running DO further')

#-----------------------------------------------------------------------------

fig, ax1	= subplots()

graph_1	= ax1.plot(time, Y_1_plus[:, 0], '-r', linewidth = 2.0, label = '$\overline{Y_1^{+}}$')
graph_2	= ax1.plot(time, Y_1_min[:, 0], '-b', linewidth = 2.0, label = '$\overline{Y_1^{-}}$')

ax1.fill_between(time, Y_1_plus[:, 1], Y_1_plus[:, 2], facecolor = 'red', alpha = 0.2)
ax1.fill_between(time, Y_1_min[:, 1], Y_1_min[:, 2], facecolor = 'blue', alpha = 0.2)

counter_1, counter_2, counter_3 = 0, 0, 0

for Y_i in range(len(Y_min_to_plus)):

    #Get the relevant realisations
	Y_i, trans_number	= Y_min_to_plus[Y_i]
	Y_1		            = Y_all[:, 0, int(Y_i)]
    

	if trans_number == 1:
		#Transition to other branch
		ax1.plot(time, Y_1, linewidth = 0.5, linestyle = '-', color = 'k')
		counter_1 += 1

	if trans_number == 2:
		#Returns to the original branch
		ax1.plot(time, Y_1, linewidth = 0.5, linestyle = '-', color = 'firebrick')
		counter_2 += 1

	if trans_number == 3:
		#Transition, no sufficient information
		counter_3 += 1

ax1.set_xlim(0, 5)
ax1.set_ylim(-6000, 6000)
ax1.set_xlabel('Time')
ax1.set_ylabel('$Y_1$')
ax1.grid()

graph_3		= ax1.plot([-10000, -10000], [-10000, -10000], '-k', linewidth = 2.0, label = r'$Y_1^{-} \rightarrow Y_1^{+}$')
graph_4		= ax1.plot([-10000, -10000], [-10000, -10000], '-', color = 'firebrick', linewidth = 2.0, label = r'$Y_1^{-} \rightarrow Y_1^{+} \rightarrow Y_1^{-}$')

graphs		= graph_1 + graph_2 + graph_3 + graph_4
legend_labels 	= [l.get_label() for l in graphs]
legend      	= ax1.legend(graphs, legend_labels, loc = (0.65, 0.90), ncol=2, numpoints = 1, framealpha=1)

ax1.text(4.85, -5000, str(counter_1), verticalalignment='center', horizontalalignment='right', color = 'k', fontsize=14)
ax1.text(4.85, -5600, str(counter_2), verticalalignment='center', horizontalalignment='right', color = 'firebrick', fontsize=14)

ax1.set_title(r'b) $Y_1^{-} \rightarrow Y_1^{+}$')

if counter_3 != 0:
	print('Y- to Y+:', counter_3, 'transitions are not defined well, consider running DO further')


#-----------------------------------------------------------------------------


fig1, ax1	= subplots()

for Y_i in range(len(Y_plus_to_min)):

    Y_i, trans_number	= Y_plus_to_min[Y_i]

    if trans_number != 1:
        #Did not stay in the other branch
        continue

    #Now get the branches for the current period
    Y_1		  = Y_all[:, 0, int(Y_i)]
    Y_2		  = Y_all[:, 1, int(Y_i)]
    Y_3		  = Y_all[:, 2, int(Y_i)]

    #Create a set of line segments so that we can color them individually
    #This creates the points as a N x 1 x 2 array so that we can stack points
    #together easily to get the segments. The segments array for line collection
    #needs to be (numlines) x (points per line) x 2 (for x and y)
    points 		= np.array([Y_2, Y_3]).T.reshape(-1, 1, 2)
    segments 	= np.concatenate([points[:-1], points[1:]], axis=1)

    #Create a continuous norm to map from data points to colors
    norm 	= plt.Normalize(-6000, 6000)
    lc 	    = LineCollection(segments, cmap='RdYlBu_r', norm=norm)

    #Set the values used for colormapping

    lc.set_array(Y_1)
    lc.set_linewidth(1.5)
    line = ax1.add_collection(lc)

ax1.set_xlim(-3000, 9000)
ax1.set_ylim(-6000, 6000)
ax1.set_xlabel('$Y_2$')
ax1.set_ylabel('$Y_3$')
ax1.grid()

cbar	= colorbar(line, ax=ax1, ticks = np.arange(-6000, 6000.1, 2000))
cbar.set_label('$Y_1$')
ax1.set_title(r'c) Transition paths, $Y_1^{+} \rightarrow Y_1^{-}$, $t = 0 - 5$')


#-----------------------------------------------------------------------------

fig1, ax1	= subplots()

for Y_i in range(len(Y_min_to_plus)):

    Y_i, trans_number	= Y_min_to_plus[Y_i]

    if trans_number != 1:
        #Did not stay in the other branch
        continue

    #Now get the branches for the current period
    Y_1		  = Y_all[:, 0, int(Y_i)]
    Y_2		  = Y_all[:, 1, int(Y_i)]
    Y_3		  = Y_all[:, 2, int(Y_i)]

    #Create a set of line segments so that we can color them individually
    #This creates the points as a N x 1 x 2 array so that we can stack points
    #together easily to get the segments. The segments array for line collection
    #needs to be (numlines) x (points per line) x 2 (for x and y)
    points 		= np.array([Y_2, Y_3]).T.reshape(-1, 1, 2)
    segments 	= np.concatenate([points[:-1], points[1:]], axis=1)

    #Create a continuous norm to map from data points to colors
    norm 	= plt.Normalize(-6000, 6000)
    lc 	    = LineCollection(segments, cmap='RdYlBu_r', norm=norm)

    #Set the values used for colormapping

    lc.set_array(Y_1)
    lc.set_linewidth(1.5)
    line = ax1.add_collection(lc)

ax1.set_xlim(-3000, 9000)
ax1.set_ylim(-6000, 6000)
ax1.set_xlabel('$Y_2$')
ax1.set_ylabel('$Y_3$')
ax1.grid()

cbar	= colorbar(line, ax=ax1, ticks = np.arange(-6000, 6000.1, 2000))
cbar.set_label('$Y_1$')
ax1.set_title(r'd) Transition paths, $Y_1^{-} \rightarrow Y_1^{+}$, $t = 0 - 5$')

show()
