# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:52:10 2015

@author: Matt Roeschke
Adapted from Eric Burger
"""

# -*- coding: utf-8 -*-


import numpy as np
import scipy.io as io
import datetime
import csv
import matplotlib.pyplot as plt


def to_unix_time(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()

def from_unix_time(sec):
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = datetime.timedelta(seconds=sec)
    return epoch+delta

##################################################################
#                                                                #
#               Prepare the data (one big list)                  #
#                                                                #
##################################################################    
all_records = []
filename = 'FRIDGE_W.csv'
with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    #CSV Headers : Unix Time, Fridge Temp, Water Bottle Temp, Soda Bottle Temp, Ambient Temp, RMS Current
    #Temp coverted to Kelvin
    for row in reader:
        
        the_sample = np.zeros(7)
        the_sample[0] = int(row[0])
        the_sample[1] = float(row[1]) + 273
        the_sample[2] = float(row[2]) + 273
        the_sample[3] = float(row[3]) + 273
        the_sample[4] = float(row[4]) + 273
        the_sample[5] = int(row[5]) 
        if the_sample[5] == 0:
            the_sample[6] = 0
        else:
            the_sample[6] = 1
        
        the_sample[1] = the_sample[1] if the_sample[1] > -20+273 else all_records[-1][1]
        the_sample[2] = the_sample[2] if the_sample[2] > -20+273 else all_records[-1][2]
        the_sample[3] = the_sample[3] if the_sample[3] > -20+273 else all_records[-1][3]
        
        # Store the sample
        all_records.append( the_sample ) 
        

all_records = np.array(all_records)
output = {}
output["fridge"] = all_records
#Output Headers: Unix Time, Fridge Temp, Water Bottle Temp, Soda Bottle Temp, Ambient Temp, RMS Current, ON/OFF
# Output to .mat file
io.savemat('fridge_data_3_12_15.mat', mdict=output)
