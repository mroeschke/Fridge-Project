#### Smart Fridge Optimization Model ####
##   Author: Zoltan DeWitt
##   UC Berkeley, Civil and Environmental Engineering
##   eCal Lab, Professor Moura, Advisor

##   OptimizationModel.py

import numpy as np
import pylab as plt
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.linalg as LA
# import scipy.io as io
# import scipy.signal as signal
# import scipy.integrate as inte
import control.matlab as ctrl
import requests as req
import datetime as DT

fs = 12    ## Font Size for plot axes
plt.close('all')



## Main Fridge Model ##
#  Recieves a setpoint schedule and predicted ambient temperature
#  Returns compressor state and soda and fridge temperature for every minute of the day

def fridge(T_sp, T_amb, A, B, x0):
    C = np.eye(2)
    D = np.zeros((2,2))
    #sys = ctrl.ss(A, B, C, D)
    T_amb = interp.interp1d([i*60 for i in range(25)], np.append(T_amb, T_amb[23]) ) # Interpolate ambient temp predictions
    t = range(1440)  # Minutes in the day
    values = np.zeros((4, 1441))  # Initialize the output data (0 = time, 1 = soda, 2 = fridge, 3 = Compressor)
    values[0,:1440] = t
    values[0,1440] = 1440
    values[1:,0] = x0  # Initial condition at the beginning
    minOn = 0
    minOff = 5
    for i in t:
        u = np.array([T_amb(i), values[3,i]])
        x = values[1:3, i]
        #nextStep = ctrl.lsim(sys, u, [1,2], values[1:3, i])[2]
        #values[1:3,i+1] = nextStep[1,:]
        currentSP = T_sp[(int)(i/60.)]
        values[1:3, i+1] = A.dot(x) + B.dot(u)
        

        if values[3, i] == 0. :
            minOff += 1
            if (values[2, i+1] > (currentSP+0.5) ) and (minOff >= 5) :
                values[3, i+1] = 1
                minOn = 0
            else:
                values[3, i+1] = 0
        else:
            minOn += 1
            if ( ( (values[2, i+1] < (currentSP-0.5) ) and (minOn >= 5) ) or (minOn >= 20) ):
                values[3, i+1] = 0
                minOff = 0
            else:
                values[3, i+1] = 1

    return values


#### Weather Underground API
## Returns an array with the hour values in column 0 and the temp forecast in column 1, beginning with the current temp

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


#### Dollar cost function

def cost(comp_schedule, rate_schedule, time_start, dt, KW):
    time = time_start
    dollars = 0.
    for i in range(len(comp_schedule)):
        dollars += comp_schedule[i]*rate_schedule[time.hour]*KW*(dt/60)
        time += DT.timedelta(minutes=dt)


    
#### Optimization Script

## Identified Parameters
p0 = -0.0097648
p1 = 0.0053631
p2 = 0.0020665
p3 = -0.15983

## Rate Schedule
RS = np.zeros((24))
RS[0:9] = 0.07 # Off-peak
RS[9:11] = 0.09 # Part-peak
RS[11:17] = 0.12 # Peak
RS[17:21] = 0.09 # Part-peak
RS[21:] = 0.07 # Off-peak

## Download weather data
T_forecast = getWUForecast()

## Download WattTime data
CI_forecast = getWTForecast()

## Only simulate for time of shortest forecast
simHours = min([len(T_forecast), len(CI_forecast)])


T_sp = np.zeros((24))
T_sp[0:8] = 10.
T_sp[8:11] = 5.
T_sp[11:16] = 4.
T_sp[16:] = 10.
T_amb = [15.,15.,16.,17.,18.,19.,20.,20.,20.,21.,22.,23.,22.,21.,21.,21.,21.,20.,20.,20.,19.,18.,17.,16.]

A_cont = np.array([[p0,      -p0      ],
                   [p1,      -p1-p2   ]])

B_cont = np.array([[0,       0    ],
                  [p2,      p3   ]])

# Create discrete-time matrices
X = np.zeros((4,4))
X[0:2,0:2] = A_cont
X[0:2,2:] = B_cont
Y = LA.expm(X)
A = Y[0:2,0:2]
B = Y[0:2,2:]

test = fridge(T_sp, T_amb, A, B, [10.,10.,0.])

f2, (thAxis) = plt.subplots(3,1,sharex=True)
f2.set_size_inches(6,9)
thAxis[0].plot((test[0,:]/60.), test[1,:], 'b-', label=r'Soda Temp')
thAxis[0].legend(fontsize=10)
#thAxis[0].set_ylim([0,20])
thAxis[1].plot((test[0,:]/60.), test[2,:], 'b-', label=r'Fridge Temp')
thAxis[1].legend(fontsize=10)
#thAxis[1].set_ylim([0,20])
thAxis[2].plot((test[0,:]/60.), test[3,:], 'b-', label=r'Compressor')
thAxis[2].legend(fontsize=10)
thAxis[2].set_ylim([0,1.5])
thAxis[2].set_xlabel(r'Hour of Day', fontsize=fs)
thAxis[2].set_xticks([0,6,12,18,24])
thAxis[2].set_xlim([0,24])

f2.savefig('ModelTest.pdf')

plt.show()


