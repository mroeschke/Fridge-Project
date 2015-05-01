# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:42:29 2015

@author: Matt Roeschke
"""
import requests as req
import datetime as DT
import numpy as np
import scipy.linalg as LA
import openopt as opt
import matplotlib.pyplot as plt

def getWUForecast():
    c = req.get('http://api.wunderground.com/api/514ebcd01c166ac3/conditions/q/CA/San_Francisco.json')
    current = c.json()
    temp_forecast = np.array([[DT.datetime.now().hour, current['current_observation']['temp_c']]])
    f = req.get('http://api.wunderground.com/api/514ebcd01c166ac3/hourly/q/CA/San_Francisco.json')
    forecast = f.json()
    for i in range(len(forecast['hourly_forecast'])):
        temp_forecast = np.append(temp_forecast, [[(int)(forecast['hourly_forecast'][i]['FCTTIME']['hour']), (float)(forecast['hourly_forecast'][i]['temp']['metric'])]], axis=0)
    return temp_forecast


#### WattTime API
## Returns an array with the hour values in column 0 and the electricity CI forecast in column 1, beginning with the current CI

def getWTForecast():
    c = req.get('http://api.watttime.org/api/v1/current/', headers={'Authorization': 'Token a06051267880dfc14751624b628eeca50e5e1138'}, params={'ba': 'CAISO'})
    forecast_raw = c.json()[0]['forecast']
    current_raw = c.json()[0]['current']
    ci_forecast = np.zeros((24))
    for i in range(len(forecast_raw)):
        timestamp_UTC = DT.datetime.strptime(forecast_raw[i]['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
        timestamp = timestamp_UTC - DT.timedelta(hours=7) # Pacific Daylight Time (-6 for standard)
        ci_forecast[timestamp.hour] = forecast_raw[i]['marginal_carbon']['value']
    ci_forecast[DT.datetime.now().hour] = current_raw['marginal_carbon']['value']
    ci_forecast_ret = np.array([[i+24, ci_forecast[i]] for i in range(DT.datetime.now().hour - 24, DT.datetime.now().hour - 24 + len(forecast_raw))])
    return ci_forecast_ret


###############################################################
#Discretize Parameters
###############################################################
#Identified Parameters
p0 = -0.0097648
p1 = 0.0053631
p2 = 0.0020665
p3 = -0.15983

A_cont = np.array([[p0,      -p0      ],
                   [p1,      -p1-p2   ]])

B_cont = np.array([[0,       0    ],
                  [p2,      p3   ]])
                  
# Create discrete-time matrices
dt = 1 # Timestep length (1 min)
X = np.zeros((4,4))
X[0:2,0:2] = A_cont
X[0:2,2:] = B_cont
Y = LA.expm(X*dt)
A_disc = Y[0:2,0:2]
B_disc = Y[0:2,2:]

###############################################################
#Formulate Optimization Matricies
###############################################################

# Enumerate Time Steps and States
N = 1440 #minutes in a day
s_states = N+5
Ts_states = N+1
Tf_states = N+1
total_states = s_states + Ts_states + Tf_states

# Download Carbon Cost and Temperature Forcast 
T_forecast = getWUForecast()
#CI_forecast = getWTForecast()

# Hour Values Hold on a Minute Basis
T_amb = np.zeros((N,1))
#carbon_cost = np.zeros((N,1))
hour_vector = np.zeros((N,1))
for i in range(N/60):
    T_amb[60*i:60*i+60,0] = T_forecast[i,1]
    #carbon_cost[60*i:60*i+60,1] = CI_forecast[i,1]
    hour_vector[60*i:60*i+60,0] = T_forecast[i,0]

######################
#Cost Fuction (just electricity cost for now)
######################
# Rate Schedule (this is just a template until a real one is determined)
RS = np.zeros((total_states))
min2hour = N/60

RS[0:9*min2hour] = 0.07 # Off-peak
RS[9*min2hour:11*min2hour] = 0.09 # Part-peak
RS[11*min2hour:17*min2hour] = 0.12 # Peak
RS[17*min2hour:21*min2hour] = 0.09 # Part-peak
RS[21*min2hour:24*min2hour] = 0.07 # Off-peak
    
######################
#Equality Contraints
######################    
# Soda Dynamics
A_eq_soda = np.zeros((N,total_states))
b_eq_soda = np.zeros((N,1))

for i in range(1,N):
    A_eq_soda[i-1,s_states + i] = A_disc[0,1]
    A_eq_soda[i-1,s_states + Ts_states + i] = A_disc[0,0]
    A_eq_soda[i-1,s_states + Ts_states + 1 + i] = -1
    
# Fridge Dynamics
A_eq_fridge = np.zeros((N,total_states))
b_eq_fridge = np.zeros((N,1))
A_eq_fridge[0,:4] = 0

for i in range(1,N):
    A_eq_fridge[i-1, 4 + i] = B_disc[1,1]
    A_eq_fridge[i-1,s_states + i] = A_disc[1,1]
    A_eq_fridge[i-1,s_states + Ts_states + i] = A_disc[1,0]
    b_eq_fridge[i-1,0] = -B_disc[1,0]*T_amb[i,0]

# Initial Temperature Conditions
Tf_init = 5
Ts_init = 5

A_Tf_init = np.zeros((1,total_states))
A_Tf_init[0,s_states + Ts_states + 1] = 1
b_Tf_init = Tf_init*np.eye(1)

A_Ts_init = np.zeros((1,total_states))
A_Ts_init[0,s_states + 1] = 1
b_Ts_init = Ts_init*np.eye(1)

 
# Concatenate   
A_eq = np.concatenate((A_eq_soda,A_eq_fridge,A_Tf_init,A_Ts_init),axis = 0)
b_eq = np.concatenate((b_eq_soda,b_eq_fridge,b_Tf_init,b_Ts_init),axis = 0)

######################
#Inequality Contraints
######################  

# 5 minute contraints
A_compressor_highbound = np.zeros((N,total_states))
b_compressor_highbound = 5*np.ones((N,1))

for i in range(N):
    A_compressor_highbound[i,i:i+3] = 1
    A_compressor_highbound[i,i+4] = -4
    A_compressor_highbound[i,i+5] = 5
    
A_compressor_lowbound = np.zeros((N,total_states))
b_compressor_lowbound = np.zeros((N,1))

for i in range(N):
    A_compressor_lowbound[i,i:i+3] = -1
    A_compressor_lowbound[i,i+4] = 4
    A_compressor_lowbound[i,i+5] = -5
    
# Soda Temperature Schedule: Two temperature periods (on and off demand) 
A_soda_schedule_highbound = np.zeros((N,total_states))
A_soda_schedule_lowbound = np.zeros((N,total_states))

for i in range(1,N):  
    A_soda_schedule_highbound[i-1,s_states + i]
    A_soda_schedule_lowbound[i-1,s_states + i]

T_high_on = 2
T_low_on = -4

T_high_off = 15
T_low_off = -6

b_soda_schedule_highbound = np.zeros((N,1))
b_soda_schedule_lowbound = np.zeros((N,1))

for i in range(len(hour_vector)):
    if hour_vector[i,0] >= 10 and hour_vector[i,0] <= 16: #between 10am and 4pm
            b_soda_schedule_highbound[i,0] = T_high_on
            b_soda_schedule_lowbound[i,0] = -T_low_on
    else:
        b_soda_schedule_highbound[i,0] = T_high_off
        b_soda_schedule_lowbound[i,0] = -T_low_off
        
# Concatenate 
        
A = np.concatenate((A_compressor_highbound,A_compressor_lowbound,A_soda_schedule_highbound,A_soda_schedule_lowbound),axis = 0)
b = np.concatenate((b_compressor_highbound,b_compressor_lowbound,b_soda_schedule_highbound,b_soda_schedule_lowbound),axis = 0)
######################
#Solve Optimization
######################

intVars = range(s_states)
ub = np.inf*np.ones((total_states))
ub[intVars] = 1

p = opt.MILP(f=RS, A=A, b=b, Aeq=A_eq, beq=b_eq, ub=ub, intVars=intVars, goal='min')
r = p.solve('lpSolve')

# Resultant States
compressor_opt = r.xf[:s_states]
Ts_opt = r.xf[s_states+1:Ts_states]
Tf_opt = r.xf[Ts_states+1:]

######################
#Plot Results
######################
fig, (axis) = plt.subplots(2,1, sharex=True)
fig.set_size_inches(6,6)
t_0 = range(len(compressor_opt))
axis[0].plot(t_0,compressor_opt,'b-') 
axis[0].set_ylabel('Compressor State')
axis[0].set_ylim(-.2,1.2)

t_1 = range(len(Ts_opt))
axis[1].plot(t_1,Ts_opt,'r-',t_1,b_soda_schedule_highbound,'b--',t_1,b_soda_schedule_lowbound,'b--')
axis[1].set_ylabel('Soda Temperature [Celcuis]')
axis[1].plot([], [], 'k-', label=r'Temperature Schedule', linewidth=10, alpha=0.1) ## Dummy plot for legend
axis[1].legendlegend(fontsize=10)













    
    
