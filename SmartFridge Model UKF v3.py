# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:03:24 2015

@author: Matt Roeschke
"""

import numpy as np
import scipy.io as io
import pylab as pl
from pykalman import UnscentedKalmanFilter
import matplotlib.pyplot as plt

plt.style.use('ggplot')

states = 4
params = 4

#p[0] should equal p[1]
def transition_function(state, noise):
    s = state[:states]
    p = state[states:]
    # States: (0) T_soda, (1) T_fridge, (2) T_ambient, (3) compressor_state
    
    state[0] = s[0] + p[0]*(s[1] - s[0])
    state[1] = s[1] + p[1]*(s[0] - s[1]) + p[2]*(s[2] - s[1]) + p[3]*s[3]
    return state + noise

def observation_function(state, noise):
    return state + noise
    
###Initalize Transition Coveriance###    
transition_covariance = np.eye(states+params)
# Set diagonal for measurements
measure_v = 1
for i in range(states):
    transition_covariance[i][i] = measure_v  
    
# Set diagonal for parameters 
param_v = 0.000012 # Lab
for i in range(params):
    transition_covariance[i+states][i+states] = param_v  
    
#transition_covariance[1+states][1+states] = 0.000000001 # Constant
#transition_covariance[4+states][4+states] = 0.000000001 # Constant 

###Initialize Observation Covariance, Initial State Mean and Initial State Covariance###
#Mean starts at 0, Covariance Set as 1
observation_covariance = np.eye(states+params) 
initial_state_mean = np.zeros(states+params)
for i in range(4):
    initial_state_mean[i] = 0
initial_state_covariance = np.eye(states+params) 

###Load Observation from Data###
#Output Data: Unix Time, Fridge Temp, Water Bottle Temp, Soda Bottle Temp, Ambient Temp, RMS Current, ON/OFF
data = io.loadmat('/Users/zoltand/code/Fridge-Project/fridge_data_3_12_15_celcius.mat')
data = data['fridge']

observations = []
for i in range(len(data)-10080,len(data)):
    observations.append( [data[i][3], data[i][1], data[i][4], data[i][6], 0, 0, 0, 0,] )
# Observations: T_soda, T_fridge, T_ambient, compressor State
# 0's = unknown parameters    
observations = np.array(observations)

n_timesteps = 10080#int(observations.shape[0])
n_dim_state = int(transition_covariance.shape[0])
filtered_state_means = np.zeros((n_timesteps, n_dim_state))
filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))

for t in range(n_timesteps - 1):
    if t == 0:
        filtered_state_means[t] = initial_state_mean
        filtered_state_covariances[t] = initial_state_covariance
        
    ukf = UnscentedKalmanFilter(
        transition_function, observation_function,
        transition_covariance, observation_covariance,
        initial_state_mean, initial_state_covariance)
        
    filtered_state_means[t + 1], filtered_state_covariances[t + 1] = (
        ukf.filter_update(filtered_state_means[t],filtered_state_covariances[t],observations[t + 1]))

p0 = []
p1 = []
p2 = []
p3 = []

#Soda Data
Ts = []
Ts_est = []

#Ambient Data
To = []
To_est = []

#Fridge Data
Tf = []
Tf_est = []

#Compressor State Data
d = []
d_est = []

for t in range(n_timesteps - 1):
    Ts.append( observations[t][0] )
    Ts_est.append( filtered_state_means[t][0] )
    Tf.append( observations[t][1] )
    Tf_est.append( filtered_state_means[t][1] )
    To.append( observations[t][2] )
    To_est.append( filtered_state_means[t][2] )
    d.append( observations[t][3] )
    d_est.append( filtered_state_means[t][3] )

    p0.append( filtered_state_means[t][4] )    
    p1.append( filtered_state_means[t][5] )
    p2.append( filtered_state_means[t][6] )
    p3.append( filtered_state_means[t][7] )
    
offset = 50
T_soda_kplus = [0]*offset
T_soda_kplus_est = [0]*offset
T_fridge_kplus = [0]*offset
T_fridge_kplus_est = [0]*offset
err_soda = []
err_fridge = []

for k in range(offset,n_timesteps - 2):
    T_soda_kplus1 = Ts[k] + p0[k]*(Tf[k] - Ts[k])
    T_fridge_kplus1 = Tf[k] + p1[k]*(Ts[k] - Tf[k]) + p2[k]*(To[k] - Tf[k]) + p3[k]*d[k]
    #Estimates with variable parameters
    T_soda_kplus.append( T_soda_kplus1 )
    T_fridge_kplus.append(T_fridge_kplus1 )
    #Error with varying parameters
    err_soda.append(Ts[k+1] - T_soda_kplus1 )
    err_fridge.append(Tf[k+1] - T_fridge_kplus1 )
    
    T_soda_kplus1_est = Ts[k] + p0[-1]*(Tf[k] - Ts[k]) 
    T_fridge_kplus1_est = Tf[k] + p1[-1]*(Ts[k] - Tf[k]) + p2[-1]*(To[k] - Tf[k]) + p3[-1]*d[k]
    #Estimates with constant parameters
    T_soda_kplus_est.append( T_soda_kplus1_est )
    T_fridge_kplus_est.append( T_fridge_kplus1_est )

soda_rmse = np.sqrt( np.mean( np.square( np.array(err_soda) ) ) )
print "Soda RMSE: ",soda_rmse
fridge_rmse = np.sqrt( np.mean( np.square( np.array(err_fridge) ) ) )
print "Fridge RMSE: ",fridge_rmse
"""    
################################################################################################
#Week Subsection
###############

#Plot Fridge Temp
pl.subplot(2,1,1)
lines_ukf = []
lines_ukf.append( pl.plot(Tf, color='r', ls='-') )
lines_ukf.append( pl.plot(Tf_est, color='b', ls='--') )
#lines_ukf.append( pl.plot(T_fridge_kplus, color='g', ls='-') )
#lines_ukf.append( pl.plot(T_fridge_kplus_est, color='k', ls='-') )
#pl.legend((lines_ukf[0][0], lines_ukf[1][0], lines_ukf[2][0],  lines_ukf[3][0]),
          #('T measured', 'T filtered', 'T sim variable $\Theta$', 'T sim constant $\Theta$'),
          #loc='upper right'
#)
pl.legend((lines_ukf[0][0], lines_ukf[1][0]),('T_{Fridge} Measured', 'T_{Fridge} filtered'),loc='lower left')
pl.title('Temperature States')
pl.ylim([270, 290])


#Plot Soda Temp
pl.subplot(2,1,2)
lines_ukf = []
lines_ukf.append( pl.plot(Ts, color='r', ls='-') )
lines_ukf.append( pl.plot(Ts_est, color='b', ls='--') )
#lines_ukf.append( pl.plot(T_soda_kplus, color='g', ls='-') )
#lines_ukf.append( pl.plot(T_soda_kplus_est, color='k', ls='-') )
#pl.legend((lines_ukf[0][0], lines_ukf[1][0], lines_ukf[2][0],  lines_ukf[3][0]),
          #('T measured', 'T filtered', 'T sim variable $\Theta$', 'T sim constant $\Theta$'),
          #loc='upper right'
#)
pl.legend((lines_ukf[0][0], lines_ukf[1][0]),('T_{Soda} Measured', 'T_{Soda} Filtered'),loc='lower left')   
pl.ylim([270, 290])
pl.xlim([0, 10080])
pl.xlabel('Time (Minutes)')
pl.ylabel('Temperature (K)')
pl.tight_layout()

pl.subplot(3,1,1)
pl.plot(To)
pl.title('Ambient Temperature')
pl.ylabel('Temperature (K)')
pl.xlim([0, 10080])

pl.subplot(3,1,2)
pl.plot(d)
pl.title('Compressor State (1 = On, 0 = Off')
pl.xlabel('Time (Minutes)')
pl.ylabel('State')
pl.xlim([0, 10080])
pl.tight_layout()
"""
################################################################################################
#Day Subsection
###############

#Plot Fridge Temp
pl.subplot(2,1,1)
lines_ukf = []
lines_ukf.append( pl.plot(Tf, color='r', ls='-') )
lines_ukf.append( pl.plot(Tf_est, color='b', ls='--') )
#lines_ukf.append( pl.plot(T_fridge_kplus, color='g', ls='-') )
#lines_ukf.append( pl.plot(T_fridge_kplus_est, color='k', ls='-') )
#pl.legend((lines_ukf[0][0], lines_ukf[1][0], lines_ukf[2][0],  lines_ukf[3][0]),
          #('T measured', 'T filtered', 'T sim variable $\Theta$', 'T sim constant $\Theta$'),
          #loc='upper right'
#)
pl.legend((lines_ukf[0][0], lines_ukf[1][0]),('$T_{Fridge}$ Measured', '$T_{Fridge}$ Filtered'),loc='lower right')
pl.title('Temperature States')
#pl.xlim([0, 1440])
pl.ylabel('Temperature (K)')
#pl.ylim([270, 290])
#pl.yticks([279,280,281,282])


#Plot Soda Temp
pl.subplot(2,1,2)
lines_ukf = []
lines_ukf.append( pl.plot(Ts, color='r', ls='-') )
lines_ukf.append( pl.plot(Ts_est, color='b', ls='--') )
#lines_ukf.append( pl.plot(T_soda_kplus, color='g', ls='-') )
#lines_ukf.append( pl.plot(T_soda_kplus_est, color='k', ls='-') )
#pl.legend((lines_ukf[0][0], lines_ukf[1][0], lines_ukf[2][0],  lines_ukf[3][0]),
          #('T measured', 'T filtered', 'T sim variable $\Theta$', 'T sim constant $\Theta$'),
          #loc='upper right'
#)
pl.legend((lines_ukf[0][0], lines_ukf[1][0]),('$T_{Soda}$ Measured', '$T_{Soda}$ Filtered'),loc='lower right')   
#pl.xlim([0, 1440])
pl.xlabel('Time (Minutes)')
pl.ylabel('Temperature (K)')
#pl.ylim([270, 290])
#pl.yticks([280,280.5,281])
pl.tight_layout()
"""
pl.subplot(3,1,1)
pl.plot(To[1440:2880])
pl.title('Ambient Temperature')
pl.ylabel('Temperature (K)')
pl.xlim([0, 1440])
pl.yticks(range(293,298))

pl.subplot(3,1,2)
pl.plot(d[1440:2880])
pl.title('Compressor State (1 = On, 0 = Off)')
pl.xlabel('Time (Minutes)')
pl.ylabel('State')
pl.xlim([0, 1440])
pl.yticks([0,1])
pl.tight_layout()
"""

####################################################################################

#Plot Params
pl.figure()
lines_ukf = []
lines_ukf.append( pl.plot(p0, color='y', ls='-') )
lines_ukf.append( pl.plot(p1, color='r', ls='-') )
lines_ukf.append( pl.plot(p2, color='b', ls='-') )
lines_ukf.append( pl.plot(p3, color='c', ls='-') )
##lines_ukf.append( pl.plot(p4, color='g', ls='-') )
##lines_ukf.append( pl.plot(p5, color='m', ls='-') )
'''
pl.legend((lines_ukf[0][0], lines_ukf[1][0], lines_ukf[2][0], lines_ukf[3][0], lines_ukf[4][0], lines_ukf[5][0]),
          ('p0','p1', 'p2', 'p3', 'p4', 'p5'),
          loc='right'
)
'''
pl.legend((lines_ukf[0][0], lines_ukf[1][0], lines_ukf[2][0], lines_ukf[3][0]),
          ('p0','p1', 'p2', 'p3')
)
pl.title('Parameters')
pl.xlabel('Time (Minutes)')
pl.ylabel('Filtered Value')
#pl.xlim([6000, 7500])
pl.ylim([-0.1, 0.1])


print p0[-1],p1[-1],p2[-1],p3[-1]

plt.show()
