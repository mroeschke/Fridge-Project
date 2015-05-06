#### CE 295 - Energy Systems and Control
##   HW 3 : State Estimation in Oil Well Drilling
##   Zoltan DeWitt, SID 13175699
##   Prof. Moura

## DEWITT_ZOLTAN_HW3.m

import numpy as np
import pylab as plt
## import matplotlib.patches as mpatches
import scipy.io as io
## import scipy.signal as signal
import scipy.integrate as inte
## import control as ctrl


def main():
    fs = 17    ## Font Size for plot axes
    plt.close('all')
    plt.style.use('ggplot')

    ## Identified Parameters
    p0 = -0.0097648
    p1 = 0.0053631
    p2 = 0.0020665
    p3 = -0.15983
    
    ## Define matrices for 2-state system (T_s, T_f)
    A = np.array([[p0,      -p0      ],
                  [p1,      -p1-p2   ]])

    B = np.array([[0,       0    ],
                  [p2,      p3   ]])

    C = np.array([[0,1]]) # Assuming only fridge measurement

    D = np.zeros((1,1))

    ###Load Observation from Data###
    #Output Data: Unix Time, Fridge Temp, Water Bottle Temp, Soda Bottle Temp, Ambient Temp, RMS Current, ON/OFF
    dataIn = io.loadmat('fridge_data_4_9_15.mat')
    dataIn = dataIn['fridge']
    start = -6080
    end = -1
    # Observations: T_soda, T_fridge, T_ambient, compressor State
    data = {'t': range(end-start), 'T_s': dataIn[start:end,3], 'T_f': dataIn[start:end,1], 'T_a': dataIn[start:end,4], 'S_c': dataIn[start:end,6]}

    #### Problem 5 - Kalman Filter
    ## Noise Covariances
    W =  np.array([[0.002,   0.   ],
                   [0.,   0.2   ]])
    N =  np.array([[0.01]])
    Sig0 = 5*np.eye(2)

    ## Input data
    input_data = np.array([data['t'], data['T_s'], data['T_f'], data['T_a'], data['S_c']])
    '''
    ## Create interpolation objects
    T_s_int = interp.interp1d(data['t'],data['T_s'])
    T_f_int = interp.interp1d(data['t'],data['T_f'])
    T_a_int = interp.interp1d(data['t'],data['T_a'])
    '''
    ## Initial Condition
    x_hat0 = np.array([0.,0.])
    states0 = np.concatenate((x_hat0, np.reshape(Sig0, 4)))

    ## Simulate Kalman Filter
    states = inte.odeint(ode_kf,states0,data['t'],args=(A,B,C,input_data,W,N))

    ## Parse States
    T_s_hat = states[:,0]
    T_f_hat = states[:,1]
    Sig11 = states[:,2]
    Sig22 = states[:,5]

    ## Compute the upper and lower bounds.
    T_s_hat_upperbound = T_s_hat + np.sqrt(Sig11)
    T_s_hat_lowerbound = T_s_hat - np.sqrt(Sig11)
    T_f_hat_upperbound = T_f_hat + np.sqrt(Sig22)
    T_f_hat_lowerbound = T_f_hat - np.sqrt(Sig22)

    ## Plot Results
    f, (kfPlot) = plt.subplots(3,1, sharex=True)
    f.set_size_inches(9,9)
    
    ##   Plot true and estimated soda temp plus/minus one sigma
    kfPlot[0].fill_between(data['t'], T_s_hat_lowerbound, T_s_hat_upperbound, alpha=0.1, color='k', label=r'One Standard Deviation')
    kfPlot[0].plot(data['t'],data['T_s'], 'g-', label='Measured')
    kfPlot[0].plot(data['t'],T_s_hat, 'r--', label='Estimated')
    kfPlot[0].plot([], [], 'k-', label=r'One Standard Deviation', linewidth=10, alpha=0.1) ## Dummy plot for legend
    kfPlot[0].set_ylabel(r'Soda Temp [Celcius]', fontsize=13)
    kfPlot[0].set_ylim(0,18)
    kfPlot[0].legend(fontsize=13)
    kfPlot[0].set_title('State Estimation of Soda Temerature with Kalman Filter',fontsize=22)
    kfPlot[0].tick_params(labelsize=15)
     ##   Plot true and estimated fridge temp plus/minus one sigma
    kfPlot[1].fill_between(data['t'], T_f_hat_lowerbound, T_f_hat_upperbound, alpha=0.1, color='k', label=r'One Standard Deviation')
    kfPlot[1].plot(data['t'],data['T_f'], 'g-', label='Measured')
    kfPlot[1].plot(data['t'],T_f_hat, 'r--', label='Estimated')
    kfPlot[1].plot([], [], 'k-', label=r'One Standard Deviation', linewidth=10, alpha=0.1) ## Dummy plot for legend
    kfPlot[1].set_ylabel(r'Fridge Temperature [Celcius]', fontsize=13)
    kfPlot[1].set_ylim(0,18)
    kfPlot[1].legend(fontsize=13)
    kfPlot[1].tick_params(labelsize=15)

    ##   Plot error between true and estimated soda temp
    kfPlot[2].plot(data['t'], data['T_s']-T_s_hat, 'b-')
    kfPlot[2].set_ylabel(r'Soda Estimation Error', fontsize=13)
    kfPlot[2].set_xlabel(r'Time [min]', fontsize=fs)
    kfPlot[2].set_ylim(-0.5, .5)
    kfPlot[2].tick_params(labelsize=15)

    f.savefig('Kamlan Filter Results.pdf')
    f.savefig('Kamlan Filter Results')

    plt.show()
    
    plt.figure().set_size_inches(9,9)
    plt.subplot(2,1,1)
    condition_lines =[]
    condition_lines.append( plt.plot(input_data[0,:],input_data[3,:]))
    plt.title('Inputs for Kamlan Filter',fontsize=22)
    plt.ylabel('Ambient Temperature [Celcius]',fontsize = fs)
    plt.tick_params(labelsize=15)
    

    plt.subplot(2,1,2)
    condition_lines =[]
    condition_lines.append( plt.plot(input_data[0,:],input_data[4,:]))
    plt.ylabel('Compressor State',fontsize = fs)
    plt.xlabel('Time [Minutes]',fontsize = fs)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('State Estimation_Typical Input States.pdf')
    plt.savefig('State Estimation_Typical Input States')

## Define function to integrate
def ode_kf(x,t,A,B,C,input_data,W,N):
    
    ## Parse States
    x_hat = np.reshape(x[0:2], (2,1))
    Sig = np.reshape(x[2:],(2,2))   ## Need to reshape Sigma from vector to matrix
    
    ## Parse and interpolate input signal data
    it = input_data[0,:]
    iT_s = input_data[1,:]
    iT_f = input_data[2,:]
    iT_a = input_data[3,:]
    iS_c = input_data[4,:]
    
    T_s = np.interp(t,it,iT_s)
    T_f = np.interp(t,it,iT_f)
    T_a = np.interp(t,it,iT_a)
    S_c = iS_c[(int)(t)] # Just finds the nearest previous value for the compressor state

    ## Assemble input and measured values vectors
    u = np.array([[T_a],
                  [S_c]])
    y_m = np.array([[T_f]])
    
    ## Compute Kalman Gain (Look at Chapter 3, Section 4)
    N_inv = np.linalg.inv(N)
    L = Sig.dot(C.T.dot(N_inv))
    
    ## Kalman Filter equations
    x_hat_dot = A.dot(x_hat) + B.dot(u) + L.dot(y_m - C.dot(x_hat))
    
    ## Riccati Equation for Sigma
    Sig_dot = Sig.dot(A.T) + A.dot(Sig) + W - Sig.dot(C.T.dot(N_inv.dot(C.dot(Sig))))
    
    ## Concatenate LHS
    xdot = np.concatenate((x_hat_dot[:,0], np.reshape(Sig_dot, 4)))
    
    return xdot


if __name__=="__main__":
    main()
