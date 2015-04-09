#### Smart Fridge Optimization Model ####
##   Author: Zoltan DeWitt
##   UC Berkeley, Civil and Environmental Engineering
##   eCal Lab, Professor Moura, Advisor

##   OptimizationModel.py

import numpy as np
import pylab as plt
import scipy.optimize as opt
import scipy.interpolate as interp
# import scipy.io as io
# import scipy.signal as signal
# import scipy.integrate as inte
import control.matlab as ctrl

fs = 12    ## Font Size for plot axes
plt.close('all')

## Main Fridge Model ##
#  Recieves a setpoint schedule and predicted ambient temperature
#  Returns compressor state and soda and fridge temperature for every minute of the day

def fridge(T_sp, T_amb, x0):
    C = np.eye(2)
    D = np.zeros((2,2))
    #sys = ctrl.ss(A, B, C, D)
    T_amb = interp.interp1d([i*60 for i in range(25)], np.append(T_amb, T_amb[23]) )
    t = range(1440)  # Minutes in the day
    values = np.zeros((4, 1441))  # Initialize the output data (0 = time, 1 = soda, 2 = fridge, 3 = Compressor)
    values[0,:1440] = t
    values[0,1440] = 1440
    values[1:,0] = x0  # Initial condition at the beginning
    minOn = 0
    minOff = 5
    for i in t:
        U = [[values[3,i], T_amb(i)], [values[3,i], T_amb(i+1)]]
        ##nextStep,, = crtl.lsim(sys, U, [0,1], values[1:3, i])
        currentSP = T_sp[(int)(i/60.)]
        values[1, i+1] = 0.49848787456465649*values[1, i] + 0.49996946220019722*values[2, i]
        values[2, i+1] = 0.31995576239621448*values[1, i] + 0.31994311766435279*values[2, i] + 0.3390683335507893*T_amb(i) + 0.00021930759174310435*values[3, i]
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

#### Optimization Script

T_sp = [4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.]
T_sp = [T_sp[i] + 273. for i in range(len(T_sp))]
T_amb = [15.,15.,16.,17.,18.,19.,20.,20.,20.,21.,22.,23.,22.,21.,21.,21.,21.,20.,20.,20.,19.,18.,17.,16.]
T_amb = [T_amb[i] + 273. for i in range(len(T_amb))]
A_cont = np.array([[0.5,  0.5],
                   [0.32, 0.32]])
B_cont = np.array([[0,    0],
                   [0.34, 4.1E-4]])

test = fridge(T_sp, T_amb, [281,281,0])

f2, (thAxis) = plt.subplots(3,1,sharex=True)
f2.set_size_inches(6,9)
thAxis[0].plot((test[0,:]/60.), test[1,:], 'b-', label=r'Soda Temp')
thAxis[0].legend(fontsize=10)
thAxis[0].set_ylim([273,293])
thAxis[1].plot((test[0,:]/60.), test[2,:], 'b-', label=r'Fridge Temp')
thAxis[1].legend(fontsize=10)
thAxis[1].set_ylim([273,293])
thAxis[2].plot((test[0,:]/60.), test[3,:], 'b-', label=r'Compressor')
thAxis[2].legend(fontsize=10)
thAxis[2].set_ylim([0,1.5])
thAxis[2].set_xlabel(r'Hour of Day', fontsize=fs)
thAxis[2].set_xticks([0,6,12,18,24])
thAxis[2].set_xlim([0,24])

f2.savefig('/Users/zoltand/Desktop/test.pdf')

plt.show()


