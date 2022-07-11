"""
This script calculates the time equivalent performance reduction, maximum performance
reduction and recovery time for a given storm event. 

Author: 
    Isaac Musaazi
Created:
    May 10, 2021
    
Latest update: 
    June 30, 2021
"""
# import os
# os.getcwd()
import pandas as pd
import numpy as np
import datetime as date
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

hurrSandy = pd.read_excel('../Python/Effluent data_Hurricane Sandy Jun092021.xlsx', header =2)
hurrSandy = hurrSandy.loc[:, [' Date', 'AAI39721', 'ANI39721','AXI39723','FI_PLTINF','FI_CMPTRT']]
hurrSandy.rename(columns ={' Date': 'DateRead', 'AAI39721':'ammonia_ppm','ANI39721':'nitrate_ppm', 'AXI39723':'phosphorus_ppm',
'FI_PLTINF':'influent_mgd','FI_CMPTRT':'treated_mgd'}, inplace = True)
hurrSandy['DateRead']= pd.to_datetime(hurrSandy['DateRead'])
timeDay = pd.Series(range(len(hurrSandy['DateRead'])), index=hurrSandy['DateRead'])/24
hurrSandy['time'] = timeDay.reset_index(drop = True)
hurrSandy['performance'] = (1 - ((hurrSandy['influent_mgd']-hurrSandy['treated_mgd'])/(hurrSandy['influent_mgd'])))*100

fit1 = SimpleExpSmoothing(hurrSandy.nitrate_ppm, initialization_method="estimated").fit()
fit2 = SimpleExpSmoothing(hurrSandy.nitrate_ppm, initialization_method="heuristic").fit(smoothing_level=0.016,optimized=False) #manually selected smoothing level
# hurrSandy['nitrate_opt'] = fit1.fittedvalues
hurrSandy['nitrate_man'] = fit2.fittedvalues


baseline = np.polyfit(hurrSandy.time, hurrSandy.ammonia_ppm, 0) ##fit data to a zero degree polynomial assuming that pollutant concentration doesnot change over time, ~the average ammonia concn
baseValue = np.poly1d(baseline)[0]

baseline = np.polyfit(hurrSandy.time, hurrSandy.phosphorus_ppm, 0) ##fit data to a zero degree polynomial assuming that pollutant concentration doesnot change over time, ~the average ammonia concn
baseValue = np.poly1d(baseline)[0]

baseline = np.polyfit(hurrSandy.time, hurrSandy.nitrate_load, 0) ##fit data to a zero degree polynomial assuming that pollutant concentration doesnot change over time, ~the average ammonia concn
baseValue = np.poly1d(baseline)[0]


time = hurrSandy.time
performance = hurrSandy.performance

#need to figure out how the first flash can be separated from an actual storm effect

#recovery time
def recoveryTimeA(conc, performance):
    '''function considers the lower limit for recovery time at the beginning of disturbance
    '''
    lower_limit = 0
    upper_limit = 0
    for i in range(5, len(time)): 
        if conc[i] < baseValue and conc[i+1] > conc[i] and conc[i-1] < conc[i] and (performance[i-2] > performance[i] and performance[i+1] < performance[i]):
            lower_limit = time[i]
            print(f" lowerLimit: {lower_limit}")
        if performance[i-2] < performance[i-1] and performance[i-1] < performance[i] and performance[i+1] == performance[i]:
            upper_limit = time[i]
            print(f" upperLimit: {upper_limit}")
            RT = upper_limit - lower_limit
            return RT
        
def recoveryTimeB(conc, performance):
    '''function considers the lower limit for recovery time based on the time with the highest pollutant concentration
    '''
    lower_limit = 0
    upper_limit = 0
    for i in range(5, len(time)): 
        if conc[i] > baseValue and conc[i-2] < conc[i-1] and conc[i-1] < conc[i] and conc[i+1] < conc[i] and (performance[i-2] and performance[i-1] > performance[i]):
            lower_limit = time[i]
            print(f" lowerLimit: {lower_limit}")
        if performance[i-2] < performance[i-1] and performance[i-1] < performance[i] and performance[i+1] == performance[i]:
            upper_limit = time[i]
            print(f" upperLimit: {upper_limit}")
            RT = upper_limit - lower_limit
            return RT

recoveryTimeA(hurrSandy.ammonia_ppm, hurrSandy.performance) #recovery time for ammonia since the concentration does not change rapidly over time
recoveryTimeB(hurrSandy.ammonia_ppm, hurrSandy.performance) #recovery time for ammonia
recoveryTimeA(hurrSandy.phosphorus_ppm, hurrSandy.performance) #recovery time for phosphorus since the concentration does not change rapidly over time
recoveryTimeB(hurrSandy.phosphorus_ppm, hurrSandy.performance) #recovery time for phosphorus
recoveryTimeA(hurrSandy.nitrate_load, hurrSandy.performance) #recovery time for the nitrate load since the concentration does not change rapidly over time
recoveryTimeB(hurrSandy.nitrate_load, hurrSandy.performance) #recovery time for nitrate load
    
            
#maximum performance reduction 
def maximumPerformanceReduction(conc, performance):
    '''function considers best performance(lowest pollutant concentration) and worst performance(highest pollutant concn)
    '''
    per_best = 0
    per_worst = 0
    for i in range(5, len(time)):
        if conc[i] > baseValue and conc[i-2] < conc[i-1] and conc[i-1] < conc[i] and conc[i+1] < conc[i] and (performance[i-2] and performance[i-1] > performance[i]):
            per_worst = performance[i]
            print(f" Worst Treatment Performnce: {per_worst}")
        if performance[i-2] < performance[i-1] and performance[i-1] < performance[i] and performance[i+1] == performance[i]: 
            per_best = performance[i]
            print(f" Best Treatment Performance: {per_best}")
            mpr = per_best - per_worst
            return mpr
maximumPerformanceReduction(hurrSandy.ammonia_ppm, hurrSandy.performance)
maximumPerformanceReduction(hurrSandy.phosphorus_ppm, hurrSandy.performance)      
maximumPerformanceReduction(hurrSandy.nitrate_load, hurrSandy.performance)      
        
def timeEquivalentPerformance(conc, performance):
      start = []
      medium = []
      end = []
      for i in range(5, len(time)):
        if conc[i] < baseValue and conc[i+1] > conc[i] and conc[i-1] < conc[i] and (performance[i-2] > performance[i] and performance[i+1] < performance[i]):
            start.append(i)
        if conc[i] > baseValue and conc[i-2] < conc[i-1] and conc[i-1] < conc[i] and conc[i+1] < conc[i] and (performance[i-2] and performance[i-1] > performance[i]): 
            medium.append(i)
        if performance[i-2] < performance[i-1] and performance[i-1] < performance[i] and performance[i+1] == performance[i]: 
            end.append(i)
            return  start, medium, end

tepr = (hurrSandy.ammonia_ppm[226] - np.trapz((hurrSandy.ammonia_ppm[226]-hurrSandy.ammonia_ppm[226:231])/hurrSandy.ammonia_ppm[226],hurrSandy.time[226:231]))

def timeEquivalentPerformance(pollutant, time):
    timePerformance = np.trapz(pollutant, time) - max(time)*pollutant[0]
    return timePerformance

timeEquivalentPerformance(hurrSandy.ammonia_ppm, hurrSandy.time)

##significance test
storm = pd.read_csv('../dcWaterStorms(October 2021)/april2018.csv')
stormDRY = storm[storm['prcp_hr']==0].reset_index(drop=True)
stormWET = storm[storm['prcp_hr']>0].reset_index(drop=True)



