"""
This script calculates the time equivalent performance reduction, maximum performance
reduction and recovery time for a given storm event. 

Latest update: 
    Nov 28, 2021
"""
import os
os.getcwd()
import pandas as pd
import numpy as np
import datetime as date

storm = pd.read_csv('../dcStormsFinal(November2021)/july2018.csv')
storm['datetime']= pd.to_datetime(storm['datetime'])
timeDay = pd.Series(range(len(storm['datetime'])), index=storm['datetime'])/6 ##creates a time column in hours based on a 10 min timestamp 
storm['time'] = timeDay.reset_index(drop = True)

storm['ammoniaMA'] = storm['ammonia_ppm_imp'].rolling(72, min_periods=1).mean() ##ammonia half-day rolling mean showed clear trend for easy analysis 
# =============================================================================
# storm['performanceMA'] = storm['performance'].rolling(72, min_periods=1).mean()
# storm['nitrateMA'] = storm['nitrate_ppm_imp'].rolling(144, min_periods=1).mean() ##nitrate daily rolling mean showed clear trend for analysis
# storm['TINMA'] = storm['TIN'].rolling(72, min_periods=1).mean()
# 
# =============================================================================
baseline = np.polyfit(storm.time, storm.ammoniaMA, 0) #fit data assuming pollutant concentration doesnot change over time
baseValue = np.poly1d(baseline)[0] #assume pollutant concentration is constant

# =============================================================================
# baseline = np.polyfit(storm.time, storm.nitrateMA, 4) #fit data assuming pollutant concentration doesnot change over time
# =============================================================================

time = storm.time
performance = storm.performanceMA
conc = storm.ammoniaMA

def recovery_time(conc,performance):
    lower_limit = 0
    upper_limit = 0
    for i in range(0, 400): # range for lower limit of ammonia based on a row number that corresponds to the time slightly after a storm event (at least greater than the preceeding dry weather days ~2 for this analysis
        
        if conc[i] <= baseValue and conc[i+1] >= baseValue and conc[i+2] > conc[i+1] and performance[i] <= 100:
            
            lower_limit = time[i]
            
            print(f"Lower Limit: {lower_limit}")

    for i in range(400, len(conc)): #range selected based on the lower limit 
        
        if conc[i] <= baseValue and conc[i-1] >= baseValue and conc[i-2] > conc[i-1] and performance[i]<= 100:
            
            upper_limit = time[i]
            
            print(f"Upper Limit: {upper_limit}")
            
            RT = upper_limit-lower_limit
            
            return RT

def tepr(conc, time):
    area_conc = abs(np.trapz(conc, time)- max(time)*conc[0]) #area under curve divided into two parts (area above baseline and area below baseline). absolute value taken because area below baseline might be greater than area above baseline
    return area_conc

tepr(storm.ammoniaMA, storm.time)

mpr = storm.nitrateMA.max() - baseValue #calculate height of peak as mpr
           