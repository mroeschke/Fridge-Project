# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:42:29 2015

@author: Matt Roeschke
"""
import requests as req
import datetime as DT
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
from sklearn import preprocessing
from lpsolve55 import *

################################
# Functions and APIs
################################
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

def lp_solve_alt(f = None, a = None, b = None, e = None, vlb = None, vub = None, xint = None, xbool = None, scalemode = None, keep = None, quick = None, bb_maxlevel = None):
  """LP_SOLVE  Solves mixed integer linear programming problems.

  SYNOPSIS: [obj,x,duals,stat] = lp_solve(f,a,b,e,vlb,vub,xint,scalemode,keep)

     solves the MILP problem

             min v = f'*x
               a*x <> b
                 vlb <= x <= vub
                 x(int) are integer
                 x(bool) are boolean

  ARGUMENTS: The first four arguments are required:

           f: n vector of coefficients for a linear objective function.
           a: m by n matrix representing linear constraints.
           b: m vector of right sides for the inequality constraints.
           e: m vector that determines the sense of the inequalities:
                     e(i) = -1  ==> Less Than
                     e(i) =  0  ==> Equals
                     e(i) =  1  ==> Greater Than
         vlb: n vector of lower bounds. If empty or omitted,
              then the lower bounds are set to zero.
         vub: n vector of upper bounds. May be omitted or empty.
        xint: vector of integer variables. May be omitted or empty.
       xbool: vector of boolean (binary) variables.
 bb_maxlevel: maximum branch-and-bound depth.
   scalemode: scale flag. Off when 0 or omitted.
        keep: Flag for keeping the lp problem after it's been solved.
              If omitted, the lp will be deleted when solved.
       quick: stops at the first found solution (sub-optimal).

  OUTPUT: A nonempty output is returned if a solution is found:

         obj: Optimal value of the objective function.
           x: Optimal value of the decision variables.
       duals: solution of the dual problem."""

  if f == None:
          help(lp_solve)
          return

  m = len(a)
  n = len(a[0])
  lp = lpsolve('make_lp', m, n)
  lpsolve('set_verbose', lp, NORMAL)
  lpsolve('set_mat', lp, a)
  lpsolve('set_rh_vec', lp, b)
  lpsolve('set_obj_fn', lp, f)
  #lpsolve('set_maxim', lp) # default is solving minimum lp.

  for i in range(m):
    if e[i] < 0:
          con_type = LE
    elif e[i] == 0:
          con_type = EQ
    else:
          con_type = GE
    lpsolve('set_constr_type', lp, i + 1, con_type)

  if vlb != None:
    for i in range(n):
      lpsolve('set_lowbo', lp, i + 1, vlb[i])

  if vub != None:
    for i in range(n):
      lpsolve('set_upbo', lp, i + 1, vub[i])

  if xint != None:
    for i in range(len(xint)):
      lpsolve('set_int', lp, xint[i], 1)
        
  if xbool != None:
    for i in range(len(xbool)):
      lpsolve('set_int', lp, xbool[i], True)
        
  if scalemode != None:
    if scalemode != 0:
      lpsolve('set_scaling', lp, scalemode)

  if bb_maxlevel != None:
    lpsolve('set_bb_depthlimit', lp, int(bb_maxlevel))

  if quick != None:
    lpsolve('set_break_at_first', lp, quick)

  result = lpsolve('solve', lp)
  if result == 0 or result == 1 or result == 11 or result == 12:
    [obj, x, duals, ret] = lpsolve('get_solution', lp)
    stat = result
  else:
    obj = []
    x = []
    duals = []
    stat = result

  if keep != None and keep != 0:
    lpsolve('delete_lp', lp)

  return [obj, x, duals]
############################################################
 
############################################################
#Download Input Data
############################################################
def initialize_model(hours=24):
    # Download Carbon Cost and Temperature Forcast 
    T_forecast = getWUForecast()
    CI_forecast = getWTForecast()

    # Remove rollover at midnight
    T_forecast[:,0] = [T_forecast[0,0] + i for i in range(len(T_forecast[:,0]))]
    CI_forecast[:,0] = [CI_forecast[0,0] + i for i in range(len(CI_forecast[:,0]))]

    # Calculate time horizon
    dt = 1. # Timestep length (minutes)
    now = DT.datetime.now()
    N = int(hours*(60/dt)) #20 hour horizon int(((min(len(T_forecast), len(CI_forecast), 24) + 1)*60. - now.minute)/dt) # Timesteps to forecast

    # Enumerate Time Steps and States
    min_cycle = 5. # Minumum cycle minutes
    extra_ts = int(np.ceil(min_cycle/dt))
    s_states = N+extra_ts
    Ts_states = N+1
    Tf_states = N+1
    total_states = s_states + Tf_states + Ts_states
    s_index = 0
    Tf_index = s_states
    Ts_index = s_states + Tf_states
    hour_vector = np.array([(now.hour + (now.minute + i)/60.) % 24 for i in range(N)]) # array of hour values in time horizon

    # Interpolate ambient forecast
    T_amb = np.interp((np.array(range(N))+now.minute)/60.+now.hour, T_forecast[:,0], T_forecast[:,1])

    data = [N, dt, now, s_states, Ts_states, Tf_states, total_states, s_index, Tf_index, Ts_index, hour_vector, T_amb, T_forecast, CI_forecast]
    return data


def run_opt(lam, data):
    [N, dt, now, s_states, Ts_states, Tf_states, total_states, s_index, Tf_index, Ts_index, hour_vector, T_amb, T_forecast, CI_forecast] = data
    #############################################################

    #############################################################
    #Cost Function
    #############################################################
    # Cost weigting factor
    #lam = 0  # 0: Dollars only, 100: Carbon only

    # Rate Schedule (this is just a template until a real one is determined)
    #A6 PGE Rate "Small Time of Use" (Summer)
    peak = 0.61173 #Peak :$0.61173/kWh
    part_peak = 0.28551 #Part Peak :$0.28551/kWh
    off_peak = 0.15804 #Off Peak: $0.15804/kWh

    # Build cost vector
    fridge_watts = 0.1 # Fridge power in kW
    # Dollars vector is calculated for dollars spent per kW per timestep
    dollars = np.zeros((total_states))
    for i in range(N):
        if ((hour_vector[i] >= 0 and hour_vector[i] < 8.5) or hour_vector[i] >= 21.5):
            dollars[i+5] = off_peak*(dt/60.)
        elif (hour_vector[i] >= 12 and hour_vector[i] < 18):
            dollars[i+5] = peak*(dt/60.)
        else:
            dollars[i+5] = part_peak*(dt/60.)

    '''
    dollar_scaler = preprocessing.MinMaxScaler(feature_range=[1,10]) #normalize dollar values as values btwn 1-10
    dollars_scaled = dollars #normalize dollar values as values btwn 1-10
    dollars_scaled[5:N+5] = dollar_scaler.fit_transform(dollars_scaled[5:N+5]) #normalize dollar values as values btwn 1-10
    '''
    # Carbon vector is calculated in lb CO2 per kW per timestep
    carbon = np.zeros((total_states))
    carbon[5:N+5] = np.interp((np.array(range(N))+now.minute)/60.+now.hour, CI_forecast[:,0], CI_forecast[:,1])*(dt/60.)/1000
    '''
    carbon_scaler = preprocessing.MinMaxScaler(feature_range=[1,10]) #normalize carbon values as values btwn 1-10
    carbon_scaled = carbon #normalize carbon values as values btwn 1-10
    carbon_scaled[5:N+5] = carbon_scaler.fit_transform(carbon_scaled[5:N+5]) #normalize carbon values as values btwn 1-10
    '''
    f = lam*carbon + (100-lam)*dollars

    ##############################################################
    #Descretize Model
    ##############################################################

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
    X = np.zeros((4,4))
    X[0:2,0:2] = A_cont
    X[0:2,2:] = B_cont
    Y = LA.expm(X*dt)
    A_disc = Y[0:2,0:2]
    B_disc = Y[0:2,2:]

    ##############################################################
    #Equality Contraints
    ##############################################################
    # Soda Dynamics
    A_eq_soda = np.zeros((N,total_states))
    b_eq_soda = np.zeros((N,1))

    for i in range(N):
        A_eq_soda[i,Tf_index + i] = A_disc[0,1]
        A_eq_soda[i,Ts_index + i] = A_disc[0,0]
        A_eq_soda[i,Ts_index + 1 + i] = -1

    # Fridge Dynamics
    A_eq_fridge = np.zeros((N,total_states))
    b_eq_fridge = np.zeros((N,1))

    for i in range(N):
        A_eq_fridge[i, 4 + i] = B_disc[1,1]
        A_eq_fridge[i,Tf_index + i] = A_disc[1,1]
        A_eq_fridge[i,Tf_index + 1 + i] = -1.
        A_eq_fridge[i,Ts_index + i] = A_disc[1,0]
        b_eq_fridge[i,0] = -B_disc[1,0]*T_amb[i]

    # Initial Temperature Conditions
    Tf_init = 4
    Ts_init = 4
    b_eq_init = np.array([[0,0,0,0,0, Tf_init, Ts_init]]).T

    A_eq_init = np.zeros((7,total_states))
    A_eq_init[:5, :5] = np.eye(5)
    A_eq_init[5, Tf_index] = 1
    A_eq_init[6, Ts_index] = 1

    # Concatenate   
    A_eq = np.concatenate((A_eq_soda,A_eq_fridge,A_eq_init),axis = 0)
    b_eq = np.concatenate((b_eq_soda,b_eq_fridge,b_eq_init),axis = 0)

    ##############################################################
    #Inequality Contraints
    ##############################################################
    # 5 minute contraints
    A_compressor_highbound = np.zeros((N,total_states))
    b_compressor_highbound = 5*np.ones((N,1))

    for i in range(N):
        A_compressor_highbound[i,i:i+4] = 1
        A_compressor_highbound[i,i+4] = -4
        A_compressor_highbound[i,i+5] = 5

    A_compressor_lowbound = np.zeros((N,total_states))
    b_compressor_lowbound = np.zeros((N,1))

    for i in range(N):
        A_compressor_lowbound[i,i:i+4] = -1
        A_compressor_lowbound[i,i+4] = 4
        A_compressor_lowbound[i,i+5] = -5

    # Soda Temperature Schedule: Two temperature periods (on and off demand) 
    A_soda_schedule_highbound = np.zeros((N,total_states))
    A_soda_schedule_lowbound = np.zeros((N,total_states))

    for i in range(N):  
        A_soda_schedule_highbound[i,s_states + Tf_states + i] = 1
        A_soda_schedule_lowbound[i,s_states + Tf_states + i] = -1

    T_high_on = 5
    T_low_on = 0

    T_high_off = 5
    T_low_off = 0

    b_soda_schedule_highbound = np.zeros((N,1))
    b_soda_schedule_lowbound = np.zeros((N,1))

    for i in range(len(hour_vector)):
        if hour_vector[i] >= 10 and hour_vector[i] <= 16: #between 10am and 4pm
                b_soda_schedule_highbound[i,0] = T_high_on
                b_soda_schedule_lowbound[i,0] = -T_low_on
        else:
            b_soda_schedule_highbound[i,0] = T_high_off
            b_soda_schedule_lowbound[i,0] = -T_low_off

    # Concatenate         
    A = np.concatenate((A_compressor_highbound,A_compressor_lowbound,A_soda_schedule_highbound,A_soda_schedule_lowbound),axis = 0)
    b = np.concatenate((b_compressor_highbound,b_compressor_lowbound,b_soda_schedule_highbound,b_soda_schedule_lowbound),axis = 0)

    ##############################################################
    #Solve MILP
    ##############################################################

    boolVars = range(1,s_states+1)  # Compressor states are boolean variables, lpsolve is 1 indexed, not 0
    eq_const = len(A_eq)
    ineq_const = len(A)
    # lpsolve inputs one set of A, b constraint matrices and an e vector that specifies equality or inequaity for each row.
    # The current version does not seem to accept numpy arrays properly, so they are transformed to lists.
    # Data fields are cast to 16-bit floats to mitigate over-precision errors
    a1 = np.concatenate((A_eq, A), axis=0).astype(np.float16).tolist()
    b1 = np.concatenate((b_eq, b), axis=0)[:,0].astype(np.float16).tolist()
    e = np.zeros((eq_const + ineq_const))
    e[eq_const:] = -1  # 0: equality, -1: less than or eq, 1: greater than or eq
    e = e.tolist()
    vlb = np.empty((total_states))
    vlb[s_states:] = -np.inf # Must allow temp states to be negative
    [obj,x,duals] = lp_solve_alt(f=f.astype(np.float16).tolist(),a=a1,b=b1,e=e,vlb=vlb,xbool=boolVars,scalemode=0,quick=True)

    ##############################################################
    #Plot Results
    ##############################################################
    # Resultant States
    compressor_opt = x[:Tf_index]
    Tf_opt = x[Tf_index:Ts_index]
    Ts_opt = x[Ts_index:]

    plt.clf()
    plt.style.use('ggplot')
    fs = 12
    fig, (axis) = plt.subplots(3,1, sharex=True)
    fig.set_size_inches(6,5)
    t_1 = range(N)
    hour_ticks = np.empty((0,2))
    for i in range (N):
        if (hour_vector[i] in [0., 4., 8., 12., 16., 20.]):
            hour_ticks = np.append(hour_ticks, [[i,hour_vector[i]]], axis=0)
    
    axis[1].plot(t_1,compressor_opt[5:],'b-') 
    axis[1].set_ylabel('Compressor', fontsize=fs)
    axis[1].set_ylim(-.2,1.2)
    axis[1].set_yticks([0,1])
    axis[1].set_yticklabels(['Off','On'])

    axis[2].plot(t_1,dollars[5:N+5]*(60./dt),'g-') 
    axis[2].set_ylabel('USD/kWh', color='g', fontsize=fs)
    twin = axis[2].twinx()
    twin.plot(t_1,carbon[5:N+5]*60/dt,'r-')
    twin.set_ylabel(r'lb CO$_2$/kWh', color='r', fontsize=fs)
    axis[2].set_xticks(hour_ticks[:,0])
    axis[2].set_xticklabels(['%02i:00' % int(hour) for hour in hour_ticks[:,1]])

    axis[0].fill_between(t_1[:N], b_soda_schedule_highbound[:N,0], b_soda_schedule_lowbound[:N,0], alpha=0.3, color='y', label=r'Temp Bounds')
    axis[0].plot(t_1[:N],Ts_opt[:N],'r-', label=r'$T_s$')
    axis[0].plot(t_1[:N],Tf_opt[:N],'b--', label=r'$T_f$')
    axis[0].plot(t_1,T_amb,'g--', label='$T_{amb}$') 
    axis[0].set_ylabel(r'$\degree$C', fontsize=fs)
    #axis[2].set_xlabel('Time of Day', fontsize=fs)
    axis[0].set_xlim(0,N)
    axis[0].set_ylim(-6,26)
    axis[0].plot([], [], 'y-', label=r'Temp Sched.', linewidth=10, alpha=0.3) ## Dummy plot for legend
    axis[0].legend(fontsize=10, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.1))

    plt.tight_layout()

    plt.show()

    opt_dollars = fridge_watts*np.vdot(x, dollars)
    opt_carbon = fridge_watts*np.vdot(x, carbon)
    opt_kwh = fridge_watts*(dt/60.)*sum(x[:s_states])
    print('Optimal dollar cost: $%f' % opt_dollars)
    print('Optimal carbon cost: %f lb CO2' % opt_carbon)
    print('Total kWh: %f' % opt_kwh)

    return fig
