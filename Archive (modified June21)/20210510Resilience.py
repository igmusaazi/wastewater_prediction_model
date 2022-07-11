# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:47:40 2021

@author: isaacmusaazi
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



os.getcwd()

os.chdir('D:\Howard\Modelling\Python')

storm = pd.read_csv('StormDataAnalysis_Isaac.csv')

#####data cleaning ############
storm.head() ###top five rows
storm = storm[['t','srtSRTtotal','q11','vss11','scod11','cod11','bod11','snh11','tkn11']].copy() ##extract specific columns
#storm = storm.loc[:,['t','srtSRTtotal','q11','vss11','scod11','cod11','bod11','snh11','tkn11']]
storm = storm.drop([0],axis=0) ##drop row with parameter units 
storm = storm.astype(float).copy() ##converts dtype to float
storm.describe() #summary statistics

#plt.plot(storm.index,storm.bod11);plt.title('BOD concn over time');plt.xlabel('time in days');plt.plot('BOD concentration')
#sns.scatterplot(x=storm.t[:150],y=storm.bod11).set(title='BOD concentration during storm', xlabel='time in days',ylabel ='BOD concentration');sns.set_theme(style="darkgrid"); plt.savefig('BOD.png')


storm_fit = np.polyfit(storm.t,storm.bod11,0) ##fit data to a zero degree polynomial assuming that pollutant concentration doesnot change over time
fit_coeff = np.poly1d(storm_fit)[0] ####defines expected pollutant concentration under standard conditions
storm['bod_removal'] = (storm.bod11 - abs(storm.bod11 - fit_coeff[0]))/storm.bod11 ####pollutant removal performance 
sns.lineplot(x=storm.t,y=storm.bod_removal).set(title='pollutant removal', xlabel='time in days',ylabel ='pollutant removal');sns.set_theme(style="darkgrid") ###lineplot of pollutant removal over time

def recovery_time(conc):
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

###time equivalent performance reduction#########
def tepr(bod11, t):
    I = np.trapz(bod11, t) - max(storm.t)*storm.bod11[1]
    return I
I = tepr(storm.bod11,storm.t)

print("The Time Equivalent Performance Reduction is {}".format(I))

#####maximum performance reduction ##############
data_fit = x 
curve = np.polyfit(storm.t,storm.bod11,2) ##fit data to a second degree polynomial
poly= np.poly1d(curve) ###equation that describes change 0.0004099 x**2 - 0.05048 x + 3.142
bod_predict = np.polyval(data_fit,storm.t[:150])

mpr = storm.bod11.max() - np.mean(poly(storm.t))
print("The Maximum Performance Reduction is {}".format(mpr))

#storm.rename(columns = {'bod performance':'bod_performance'}, inplace=True)

###read excel file and extract .csv files ####
# data = pd.read_excel('WEFTECSimulations.xlsx', sheet_name=None)

# for sheet_name, df in data.items():
#     for sheet_name, df in data.items():
#     df.to_csv(f"{sheet_name}.csv")
