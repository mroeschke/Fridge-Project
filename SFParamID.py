#### Smart Fridge Parameter Identification ####
##   Author: Zoltan DeWitt
##   UC Berkeley, Civil and Environmental Engineering
##   eCal Lab, Professor Moura, Advisor

##   SFParamID.py

import numpy as np
import pylab as plt
import scipy.io as io
import scipy.signal as signal
import scipy.integrate as inte
import scipy.interpolate as interp
#from control.matlab import *

def ode_gradient(theta_h_vect,t,dataIn,Gam):

    #### Parse Input Data
    it = dataIn[0] ## t   : time vector [min]
    iT_s = dataIn[1] ## T_s : Soda temp [deg C]
    iT_f = dataIn[2] ## T_f : Fridge temp [deg C]
    iT_a = dataIn[3] ## T_a : Ambient outdoor temperature [deg C]
    iS_c = dataIn[4] ## S_c : Compressor state
    theta_h = np.reshape(theta_h_vect, (3,2))

    #### Interpolate data
    
    T_s = np.interp(t,it,iT_s)
    T_f = np.interp(t,it,iT_f)
    T_a = np.interp(t,it,iT_a)
    ## Fake it with compressor state
    S_c = iS_c[(int)(t)]

    #### Parametric model notation
    ## Sampling time step
    dt = 1

    ## Compute temperatures at NEXT time step
    T_s_plus = np.interp(t+dt,it,iT_s)
    T_f_plus = np.interp(t+dt,it,iT_f)

    ## Compute z using forward difference in time
    z = np.array([[(T_s_plus - T_s)/dt], [(T_f_plus - T_f)/dt]])

    ## Assemble regressor vector, \phi
    phi = np.array([[T_s-T_f],
                    [T_a-T_f],
                    [S_c]])

    #### Gradient Update Law
    ## Normalization signal
    msq = 1. + (phi.T.dot(phi))[0,0]

    ## Estimation error: \epsilon = z - \theta_h^T \phi
    epsilon = (z - theta_h.T.dot(phi))/msq

    ## Update Law
    theta_h_dot = Gam * phi.dot(epsilon.T)
    returnVal = np.reshape(theta_h_dot, 6)
    return returnVal



fs = 12    ## Font Size for plot axes
plt.close('all')
plt.style.use('ggplot')

###Load Observation from Data###
#Output Data: Unix Time, Fridge Temp, Water Bottle Temp, Soda Bottle Temp, Ambient Temp, RMS Current, ON/OFF
dataIn = io.loadmat('fridge_data_3_12_15_celcius.mat')
dataIn = dataIn['fridge']
start = -18080
end = -1
# Observations: T_soda, T_fridge, T_ambient, compressor State
data = {'t': range(end-start), 'T_s': dataIn[start:end,3], 'T_f': dataIn[start:end,1], 'T_a': dataIn[start:end,4], 'S_c': dataIn[start:end,6]}


## Assemble Data
dataArr = np.array([data['t'], data['T_s'], data['T_f'], data['T_a'], data['S_c']])

## Initial conditions
theta_hat0 = np.array([[-0.00963, 0.00599],
                       [0,    0.00207],
                       [0,    -0.159]])
theta_hat0_vect = np.reshape(theta_hat0, 6)

## Update Law Gain
Gam = np.array([[0.01,  0.01],
                [0,      0.0002],
                [0,      0.002]])

## Integrate ODEs
t_int = np.delete(np.delete(data['t'],-1),-1,)
y = inte.odeint(ode_gradient, theta_hat0_vect, t_int, args=(dataArr, Gam), hmax=1)

print([y[-1,0], y[-1,1], y[-1,3], y[-1,5]])
"""
## Plot parameter estimates
f, (thAxis) = plt.subplots(4,1,sharex=True)
f.set_size_inches(9,9)
thAxis[0].plot(t_int, y[:,0], 'b-', label=r'p0')
thAxis[1].plot(t_int, y[:,1], 'r-', label=r'p1')
thAxis[2].plot(t_int, y[:,3], 'y-', label=r'p2')
thAxis[3].plot(t_int, y[:,5], 'g-', label=r'p3')
thAxis[0].legend(fontsize=10)
thAxis[1].legend(fontsize=10)
thAxis[2].legend(fontsize=10)
thAxis[3].legend(fontsize=10)
thAxis[3].set_xlabel(r'Minutes', fontsize=fs)
"""
plt.figure().set_size_inches(9,9)
param_lines = []
param_lines.append( plt.plot(y[:,0], color='y', ls='-') ) #p0?
param_lines.append( plt.plot(y[:,1], color='r', ls='-') ) #p1?
param_lines.append( plt.plot(y[:,3], color='b', ls='-') ) #p2?
param_lines.append( plt.plot(y[:,5], color='c', ls='-') ) #p3?
plt.legend((param_lines[0][0], param_lines[1][0], param_lines[2][0], param_lines[3][0]),
          ('p0','p1', 'p2', 'p3'),
          loc='right',fontsize=15
)
plt.title('Parameters',fontsize=22)
plt.xlabel('Time [Minutes]',fontsize=15)
plt.ylabel('Filtered Value',fontsize=15)
plt.tick_params(labelsize=15)
plt.savefig('Parameter Estimates.pdf')
plt.savefig('Parameter Estimates')

plt.figure().set_size_inches(9,9)
plt.subplot(2,1,1)
condition_lines =[]
condition_lines.append( plt.plot(dataArr[1,1:1440]))
condition_lines.append( plt.plot(dataArr[2,1:1440]))
condition_lines.append( plt.plot(dataArr[3,1:1440]))
plt.title('States for Parameter Identification (Typical Control)', fontsize=16)
plt.ylabel('Temperature [Celcius]',fontsize=15)
plt.tick_params(labelsize=15)
plt.legend((condition_lines[0][0], condition_lines[1][0], condition_lines[2][0]),
          ('$T_{Soda}$','$T_{Refrigerator}$', '$T_{Ambient}$'),loc='right',fontsize=15)#,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.subplot(2,1,2)
condition_lines =[]
condition_lines.append( plt.plot(dataArr[4,1:1440]))
plt.ylabel('Compressor State',fontsize=15)
plt.xlabel('Time [Minutes]',fontsize=15)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.savefig('Parameter Estimation_Typical Control States.pdf')
plt.savefig('Parameter Estimation_Typical Control States')


plt.figure().set_size_inches(9,9)
plt.subplot(2,1,1)
condition_lines =[]
condition_lines.append( plt.plot(dataArr[1,-1440:]))
condition_lines.append( plt.plot(dataArr[2,-1440:]))
condition_lines.append( plt.plot(dataArr[3,-1440:]))
plt.tick_params(labelsize=15)
plt.title('States for Parameter Identification (Forced Control)', fontsize=16)
plt.ylabel('Temperature [Celcius]',fontsize=15)
plt.legend((condition_lines[0][0], condition_lines[1][0], condition_lines[2][0]),
          ('$T_{Soda}$','$T_{Refrigerator}$', '$T_{Ambient}$'),loc='right',fontsize=15)#,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.subplot(2,1,2)
condition_lines =[]
condition_lines.append( plt.plot(dataArr[4,-1440:]))
plt.ylabel('Compressor State',fontsize=15)
plt.xlabel('Time [Minutes]',fontsize=15)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.savefig('Parameter Estimation_Forced Control States.pdf')
plt.savefig('Parameter Estimation_Forced Control States')


## f.savefig('/Users/zoltand/code/')

plt.show()
