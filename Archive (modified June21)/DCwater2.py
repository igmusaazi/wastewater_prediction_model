# -*- coding: utf-8 -*-
"""
Created on Thu May 20 19:36:32 2021

@author: musaa
"""
import os
os.getcwd()
os.chdir('D:/Howard/Modelling/Python') ###directory with the file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as date

####sheets with flow data combined i.e. 2017, 2018 and 2019###############
waterDC = pd.read_excel('../Python/DC Water Data.xlsx', sheet_name=None, header =2) #read all sheets
waterDComb = pd.concat(waterDC.values(),join = 'inner', axis = 0,ignore_index=True) ##combine the sheets into one sheet
waterDComb[' Date']= pd.to_datetime(waterDComb[' Date']) ##convert Date column from string to datetime
#waterDComb['year'], waterDComb['month'] = waterDComb[' Date'].dt.year, waterDComb[' Date'].dt.month ##create month and year columns

###plotting influent flow over time #######
fig, ax = plt.subplots()
ax.plot(' Date', 'FI_PLTINF', data=waterDComb)
fmt_half_year = mdates.MonthLocator()
ax.xaxis.set_minor_locator(fmt_month)

fmt_month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(fmt_month)

# Text in the x axis will be displayed in 'YYYY-mm' format.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Round to nearest years.
#datemin = np.datetime64(waterDComb[' Date'][0], 'Y')
#datemax = np.datetime64(waterDComb[' Date'][-1], 'Y') + np.timedelta64(1, 'Y')
#ax.set_xlim(datemin, datemax)

# Format the coords message box, i.e. the numbers displayed as the cursor moves
# across the axes within the interactive GUI.
ax.format_xdata = mdates.DateFormatter('%Y-%m')
ax.format_ydata = lambda x: f'${x:.2f}'  # Format the price.
ax.grid(True)

# Rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them.
fig.autofmt_xdate()

###define cumulative distribution function##########
counts, bins = np.histogram(waterDComb[['FI_PLTINF']], bins = 10)
pdf = counts / sum(counts)
cdf = np.cumsum(pdf)
plt.xlabel('Flow', size = 10)
plt.plot(bins[1:],pdf, color="red", label="PDF")
plt.plot(bins[1:], cdf, label="CDF")
plt.axhline(y=0.95, color='k', linestyle='dashed',label = '95th percentile')
plt.axvline(x=np.percentile(waterDComb[['FI_PLTINF']], 95),  color='r', linestyle='dashdot', label = 'Storm Flow')
plt.legend()
plt.savefig('DC Water Storm Identification.png')

####define the storm days####################
time = waterDComb[' Date']
flowMax = np.percentile(waterDComb[['FI_PLTINF']], 95)
def storm_date(flowdata):
    stormTime = 0
    for i in range(0, 1092):
        
        if flowdata[i] >= flowMax:
            stormTime = time[i]
            print(f'The storm days are:{stormTime}')

storm_date(waterDComb.FI_PLTINF)
