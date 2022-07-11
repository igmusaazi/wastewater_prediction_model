# -*- coding: utf-8 -*-
"""
Created on Thu May 20 19:36:32 2021

@author: musaa
"""
import os
os.getcwd()
os.chdir()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as date
from sklearn.cluster import KMeans

######combining spreadsheets with flow data#############
waterDC = pd.read_excel('../Python/DC Water Data.xlsx', sheet_name=None, header =2) #read all sheets
waterDComb = pd.concat(waterDC.values(),join = 'inner', axis = 0,ignore_index=True) ##combine the sheets into one sheet
waterDComb[' Date']= pd.to_datetime(waterDComb[' Date']) ##convert Date column from string to datetime
waterDComb.rename(columns ={' Date': 'time', 'FI_PLTINF':'inf_flo', 'FI_CMPTRT':'treat_flow' ,'AAI39721':'ammonia_ppm','ANI39721':'nitrate_ppm'}, inplace = True)
#waterDComb['year'], waterDComb['month'] = waterDComb[' Date'].dt.year, waterDComb[' Date'].dt.month ##create month and year columns

cluster =KMeans(n_clusters = 2)
waterDComb['cluster'] = cluster.fit_predict(waterDComb[waterDComb.columns[1:4]])

###plot influent flow over time cluster one #######
fig, ax = plt.subplots()
ax.plot('time', 'inf_flo', data0=waterDComb[waterDComb['cluster'] == 0])
six_month_interval = mdates.MonthLocator(interval=6)
ax.xaxis.set_major_locator(six_month_interval) ###defines major axis every 6 months
month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(month) ###defines minor axis every month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.set_ylabel('Flow in MGD')
ax.set_xlabel('Period of Flow')
fig.autofmt_xdate() # Rotates and right aligns the x labels
plt.savefig('combined flow cluster 1.png',dpi = 1200)

###plot influent flow over time cluster two #######

fig, ax = plt.subplots()
ax.plot('time', 'inf_flo', data1=waterDComb[waterDComb['cluster'] == 1])
six_month_interval = mdates.MonthLocator(interval=6)
ax.xaxis.set_major_locator(six_month_interval) ###defines major axis every 6 months
month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(month) ###defines minor axis every month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.set_ylabel('Flow in MGD')
ax.set_xlabel('Period of Flow')
fig.autofmt_xdate() # Rotates and right aligns the x labels
plt.savefig('combined flow cluster 2.png',dpi = 1200)



###cumulative distribution function for influent flow cluster one##########
data0 = waterDComb[waterDComb['cluster'] == 0]
counts, bins = np.histogram(data0['inf_flo'], bins = 10)
pdf = counts / sum(counts)  ####
cdf = np.cumsum(pdf)

plt.xlabel('Flow in MGD', size = 15)
plt.ylabel('Cumulative Probability', size = 15)
plt.title('Cumulative Probability and Flow (cluster one)')
plt.plot(bins[1:],pdf, color="red", label="PDF")
plt.plot(bins[1:], cdf, label="CDF")
plt.axhline(y=0.95, color='k', linestyle='dashed',label = '95th percentile')
plt.axvline(x=np.percentile(data0['inf_flo'], 95),  color='r', linestyle='dashdot', label = 'Estimate of Storm Flow')
plt.legend()
plt.savefig('Storm Threshold cluster one.png', dpi = 1200)

###cumulative distribution function for influent flow cluster two##########
data1 = waterDComb[waterDComb['cluster'] == 1]
counts, bins = np.histogram(data1['inf_flo'], bins = 10)
pdf = counts / sum(counts)  ####
cdf = np.cumsum(pdf)

plt.xlabel('Flow in MGD', size = 15)
plt.ylabel('Cumulative Probability', size = 15)
plt.title('Cumulative Probability and Flow (cluster two)')
plt.plot(bins[1:],pdf, color="red", label="PDF")
plt.plot(bins[1:], cdf, label="CDF")
plt.axhline(y=0.95, color='k', linestyle='dashed',label = '95th percentile')
plt.axvline(x=np.percentile(data1['inf_flo'], 95),  color='r', linestyle='dashdot', label = 'Estimate of Storm Flow')
plt.legend()
plt.savefig('Storm Threshold cluster two.png', dpi = 1200)


####function that defines a storm based on flow data####################
# time = cluster_two['time']
# flowMax = np.percentile(cluster_two['inf_flo'], 95)
time = data0['time']
flowMax = np.percentile(data0['inf_flo'], 95)
stormTime = []
def storm_date(flowdata):
    for i in range(0, len(data0)):
            print('stormDay', stormTime)
            
stormDay = storm_date(data0.inf_flo)

