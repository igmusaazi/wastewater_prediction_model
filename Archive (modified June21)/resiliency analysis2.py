#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 04 13:28hrs 2021

@author: 
@author:isaacmusaazi
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import file with results
os.getcwd()

os.chdir("C:\\Users\\musaa\\Dropbox\\GOALI shared\\Modeling\\Python")

df = pd.read_excel('Users/musaa/Dropbox/GOALI shared/Modeling/SimulationOutputs/WEFTECSimulations.xlsx', sheet_name='MABR_Storm 4')

# identification of necessary data......subject to change based on arrangement of data in imported file

# time_df = df['t']  # stored the time in MBx4

# pollutant_df = df['bod11']  # stored the effluent parameters in MBy4


# time = np.array(time_df)

# pollutant = np.array(pollutant_df)


# # For time equivalent performance reduction
# def tepr(conc, time):
    
#     area_conc = np.trapz(conc, time)-max(time)*conc[0]
    
#     return area_conc


# tepr = tepr(pollutant, time)

# print(f"Time Equivalent Performance Reduction: {tepr}")


# # for maximum performance reduction
# #z = np.polyfit(time[-5:], pollutant[-5:], 0) #used this line when wanted the y=a fit from the last 5 points 

# z = np.polyfit(time[:10], pollutant[:10], 0)  # used this line when wanted the y=a fit from the first 10 points

# p = np.poly1d(z)

# plt.plot(time, pollutant, '.', time, p(time), '--')

# mpr = pollutant.max() - np.mean(p(time))

# print(f'Maximum Performance Reduction: {mpr}')

# print (p)


# # for recovery time......index numbers based on data in the excel sheet. These are subject to change.
# def recovery_time(conc):
#     lower_limit = 0
#     upper_limit = 0
#     for i in range(0, 36): # to get the greater lower limit for ammonia, change the range to (0, 50)
        
#         if conc[i] <= p and conc[i+1] >= p and conc[i+2] >= conc[i+1]:
            
#             lower_limit = time[i]
            
#             print(f"Lower Limit: {lower_limit}")

#     for i in range(36, len(conc)): # use the polyline based on the last 5 points to get the upper limit for ammonia
        
#         if conc[i] <= p and conc[i-1] >= p and (conc[i-2] > conc[i-1] or conc[i - 2] == conc[i-1]):
            
#             upper_limit = time[i]
            
#             print(f"Upper Limit: {upper_limit}")
            
#             RT = upper_limit-lower_limit
            
#             return RT


# RT_pollutant = recovery_time(pollutant)

# print(f"Recovery Time: {RT_pollutant}")
