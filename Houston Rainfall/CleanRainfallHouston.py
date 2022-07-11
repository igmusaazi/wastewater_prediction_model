# -*- coding: utf-8 -*-
"""
This script reads sentry flow minute data and determines the average daily flow
Line 24 - 27 reorganizes Houston Rainfall data to only capture the 53rd minute of every time stamp
  

@author: Isaac Musaazi and Moriah Brown
Date created: September 29, 2021

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as date
import matplotlib.dates as mdates
import scipy.stats as stats

os.getcwd()

flowDecatur = pd.read_csv('../Sanitary District Decatur (SDD)_Data_August_2021/FlowData.csv')
flowDecatur['TIMESTAMP']= pd.to_datetime(flowDecatur['TIMESTAMP']) #change Dtype to datetime64[ns]

##re-organizes the data and captures only those timestamps at the 53rd min of every hour in Houston Dataset########### 
#rainHouston = pd.read_csv('../Downloads/Houston Texas Hourly Rainfall 2018 through 2021_Messy.csv')
#rainHouston['DATE']= pd.to_datetime(rainHouston['DATE'])
#rainH = rainHouston.groupby(pd.Grouper(key='DATE', freq='60min', offset ='53min')).head(1).reset_index()

rainDecatur = pd.read_csv('../Sanitary District Decatur (SDD)_Data_August_2021/Daily rainfall Decatur Airport.csv')
rainDecatur['DATE']= pd.to_datetime(rainDecatur['DATE'])

flowDecatur = flowDecatur.dropna() #drop missing data
dailyDecatur = flowDecatur.groupby(pd.Grouper(key='TIMESTAMP', freq='1d'))['FLOW IN MGD'].mean().reset_index() #determine the daily average flow
#dailyDecatur.to_csv('Average Daily Flow.csv')

decaturCCR = pd.read_csv('../Sanitary District Decatur (SDD)_Data_August_2021/Sentry Influent.csv')
decaturCCR = decaturCCR.dropna() #drop missing data
decaturCCR['Timestamp']= pd.to_datetime(decaturCCR['Timestamp']) #change Dtype to datetime64[ns]

#plot changes in flow over time 
fig, ax = plt.subplots()
ax.plot('TIMESTAMP', 'FLOW IN MGD', color = 'crimson',label ='Average Daily Flow(SDD)', data = dailyDecatur)
month_interval = mdates.MonthLocator(interval=1)
ax.xaxis.set_major_locator(month_interval) ###defines major axis every 6 months
month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(month) ###defines minor axis every month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.legend()
plt.title('Average Daily Flow ', fontweight = 'bold',fontsize = 10)
ax.set_ylabel('FLOW RATE (MGD)', fontweight ='bold',fontsize = 10)
ax.tick_params(color = 'black', labelcolor='black',labelsize = 'large', width=2)
ax.yaxis.label.set_color('black')
ax.set_xlabel('Time (Days)', fontweight = 'bold',fontsize = 15)
#plt.savefig('Flow Changes (SDD).png',dpi = 1200)


###determine storm days########

flowDaily = dailyDecatur
counts, bins = np.histogram(flowDaily['FLOW IN MGD'], bins = 10)
pdf = counts / sum(counts)  ####
cdf = np.cumsum(pdf)

plt.xlabel('Flow in MGD', size = 15)
plt.ylabel('Cumulative Probability', size = 15)
plt.title('Cumulative Probability and Flow')
plt.plot(bins[1:],pdf, color="red", label="PDF")
plt.plot(bins[1:], cdf, label="CDF")
plt.axhline(y=0.90, color='k', linestyle='dashed',label = '90th percentile')
plt.axvline(x=np.percentile(flowDaily['FLOW IN MGD'], 90),  color='r', linestyle='dashdot', label = 'Estimate of Storm Flow')
plt.legend()
plt.savefig('Threshold Flow.png',dpi = 1200)


rainDecatur['rank'] = rainDecatur['PRCP'].rank(ascending=False,method='dense')
observation = len(rainDecatur['rank'].unique())
rainDecatur['prob_exceed'] = ((rainDecatur['rank'] - 0.44)/(observation + 0.12))*100

y=20
plt.scatter(rainDecatur['PRCP'], rainDecatur['prob_exceed'], color="purple")
plt.xlabel('Rainfall (Inches)', size = 15)
plt.ylabel('Probability of exceedance(%)', size = 15)
plt.title('Rainfall Frequency Analysis')
plt.axhline(y, color='k', linestyle='dashed',label = '20% probability')
#plt.axvline(x=np.interp(0.2,data0['rain_inches'],data0['prob_exceed']),  color='r', linestyle='dashdot', label = 'Storm Cutoff')
#plt.axvline(x=data0[['prob_exceed']],  color='r', linestyle='dashdot', label = 'Storm Cutoff')
plt.legend()
plt.savefig('Storm Rainfall.png',dpi = 1200)


flowRainDecatur = pd.merge(left =dailyDecatur,right = rainDecatur, left_on = 'TIMESTAMP', right_on ='DATE') #column merge based on 'DateRead'

def storm_day(flowRainDecatur):
    '''this function considers a heavy rainfall event to have >0.5 inches recorded '''
    baseFlow = np.percentile(flowRainDecatur['FLOW IN MGD'], 90)
    
    for i in range(0, len(flowRainDecatur)): 
        rain = flowRainDecatur['PRCP']
        flow = flowRainDecatur['FLOW IN MGD']
        DateRead = flowRainDecatur['DATE']
        if flow[i] >= baseFlow and rain[i] >= 0.5: 
            stormDay = DateRead[i]
            with open('identified_stormDay.csv', 'a') as output:
                print('stormDay', stormDay, file = output)
                
stormDay = storm_day(flowRainDecatur)

flowCCRDecatur = pd.merge(left = decaturCCR,right = flowDecatur, on = 'Timestamp') #CCR and flow column merge based on 'Timestamp'


# =============================================================================
# def flowCCR(start_date, end_date, flowCCRDecatur):
#     storm = 0
#     for i in range(0, len(flowCCRDecatur)):
#         start_date[i] = 0
#         end_date[i]   = 0
#     stormFilter = flowCCRDecatur[(flowCCRDecatur['Timestamp'] > start_date[i]) & (flowCCRDecatur['Timestamp'] < end_date[i])]
#     storm = flowCCRDecatur.loc[stormFilter]
# 
# =============================================================================
