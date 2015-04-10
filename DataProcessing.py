#### CSV file to Matlab data file converter
## Author: Zoltan DeWitt

import csv
import scipy.io as io

def csvtodict(path_in, keynames):
    dict_out = {names: None for name in keynames}
    with open(path_in, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for i in range(len(keynames)):
                dict_out[keynames(i)].append(row[i])
    return dict_out

fridgedata = csvtodict('/Users/zoltand/Google Drive/FridgeProjectShared/Fridge Data/FRIDGE_W.CSV', 
