"""
This script reads the DC water flow data file and identifies storm events based\
    on a threshold flow rate determined from a cumulative distribution function

Author: 
    Isaac Musaazi
"""
import os
os.getcwd()
os.chdir()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as date
import scipy.stats as stats
from sklearn.cluster import KMeans
#from sklearn.impute import SimpleImputer

##blank_prec = prec.loc[prec['PRCP'].isnull()] ##identifies missing values in PRCP column

######combine flow data for different years together#############
waterDC = pd.read_excel('../Python/DC Water Data.xlsx', sheet_name=None, header =2) #read all sheets
flowData = pd.concat(waterDC, axis = 0).reset_index(drop= True) ##combine the sheets into one sheet, continous indexes
flowData[' Date']= pd.to_datetime(flowData[' Date']) ##convert Date column from string to dateDateRead
flowData.rename(columns ={' Date': 'DateRead', 'FI_PLTINF':'influent_mgd', 'FI_CMPTRT':'treatFlow_mgd' ,'AAI39721':'ammonia_ppm','ANI39721':'nitrate_ppm'}, inplace = True)
#flowData['year'], flowData['month'] = flowData[' Date'].dt.year, flowData[' Date'].dt.month ##create month and year columns
flowData = flowData.loc[:, ['DateRead', 'influent_mgd', 'treatFlow_mgd','ammonia_ppm', 'nitrate_ppm']] 

rainfall = pd.read_csv('../Python/DC precipitation.csv')
#rainfall = rainfall[rainfall['PRCP']>0]
rainfall = rainfall.loc[:, ['DATE','PRCP']]
rainfall.rename(columns ={'DATE': 'DateRead', 'PRCP':'rain_inches'}, inplace = True)
rainfall['DateRead']= pd.to_datetime(rainfall['DateRead']) ##convert Date column from string to dateDateRead
flowRain = pd.merge(left =flowData,right = rainfall, left_on = 'DateRead', right_on ='DateRead') #column merge based on 'DateRead'

#flowRain.to_csv('DCWaterFinal.csv') 

cluster =KMeans(n_clusters = 2)
flowRain['cluster'] = cluster.fit_predict(flowRain[flowData.columns[1:5]])

###influent flow rate and rainfall over DateRead

def flow_rain(toPlot):
    '''function that plots flowRate and rainfall data for each cluster. the figure is saved to the local directory
    '''
    fig, ax = plt.subplots()
    ax.plot('DateRead', 'influent_mgd', color = 'red', data = toPlot)
    six_month_interval = mdates.MonthLocator(interval=6)
    plt.title('Flow Rate and Rainfall Measurements over Time', fontweight = 'bold')
    ax.xaxis.set_major_locator(six_month_interval) ###defines major axis every 6 months
    month = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(month) ###defines minor axis every month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  
    ax.set_ylabel('Flow in MGD', fontweight ='bold')
    ax.tick_params(axis = 'y', labelcolor='r', labelsize='medium', width=2)


    ax.yaxis.label.set_color('red')
    ax.set_xlabel('Date of Measurement', fontweight = 'bold')
    fig.autofmt_xdate() # Rotates and right aligns the x labels
    ax1 = ax.twinx()
    ax1.plot('DateRead', 'rain_inches',color = 'blue', data = toPlot)
    ax1.set_ylabel('Precipitation in inches', fontweight = 'bold') 
    ax1.tick_params(axis = 'y', labelcolor='b', labelsize='medium', width=2)
    
    ax1.yaxis.label.set_color('blue')
    ax1.invert_yaxis()
    return(plt.savefig('Flow Rate and Rainfall.png',dpi = 1200))

flow_rain(flowRain[flowRain['cluster'] == 0]) 
flow_rain(flowRain[flowRain['cluster'] == 1]) 

###cumulative distribution function for influent flow cluster one##########

data0 = flowRain[flowRain['cluster'] == 0].reset_index(drop= True)
counts, bins = np.histogram(data0['influent_mgd'], bins = 10)
pdf = counts / sum(counts)  ####
cdf = np.cumsum(pdf)

plt.xlabel('Flow in MGD', size = 15)
plt.ylabel('Cumulative Probability', size = 15)
plt.title('Cumulative Probability and Flow (cluster one)')
plt.plot(bins[1:],pdf, color="red", label="PDF")
plt.plot(bins[1:], cdf, label="CDF")
plt.axhline(y=0.90, color='k', linestyle='dashed',label = '90th percentile')
plt.axvline(x=np.percentile(data0['influent_mgd'], 90),  color='r', linestyle='dashdot', label = 'Estimate of Storm Flow')
plt.legend()
#plt.savefig('Storm Threshold cluster one.png', dpi = 1200)

###cumulative distribution function for influent flow cluster two##########
data1 = flowRain[flowRain['cluster'] == 1].reset_index(drop= True)
counts, bins = np.histogram(data1['influent_mgd'], bins = 10)
pdf = counts / sum(counts)  ####
cdf = np.cumsum(pdf)

plt.xlabel('Flow in MGD', size = 15)
plt.ylabel('Cumulative Probability', size = 15)
plt.title('Cumulative Probability and Flow (cluster two)')
plt.plot(bins[1:],pdf, color="red", label="PDF")
plt.plot(bins[1:], cdf, label="CDF")
plt.axhline(y=0.95, color='k', linestyle='dashed',label = '95th percentile')
plt.axvline(x=np.percentile(data1['influent_mgd'], 95),  color='r', linestyle='dashdot', label = 'Estimate of Storm Flow')
plt.legend()
#plt.savefig('Storm Threshold cluster two.png', dpi = 1200)

#estimating the probability of exceedance using the Gringorten (1983) method
flowRain['rank'] = flowRain['rain_inches'].rank(ascending=False,method='dense')
observation = len(flowRain['rank'].unique())
flowRain['prob_exceed'] = ((flowRain['rank'] - 0.44)/(observation + 0.12))*100

data0 = flowRain[flowRain['cluster'] == 0].reset_index(drop= True)
y=20
plt.scatter(data0['rain_inches'], data0['prob_exceed'], color="purple")
plt.xlabel('Rainfall in inches', size = 15)
plt.ylabel('Probability of exceedance(%)', size = 15)
plt.title('Rainfall Frequency Analysis (cluster one)')
plt.axhline(y, color='k', linestyle='dashed',label = '10% probability')
#plt.axvline(x=np.interp(0.2,data0['rain_inches'],data0['prob_exceed']),  color='r', linestyle='dashdot', label = 'Storm Cutoff')
#plt.axvline(x=data0[['prob_exceed']],  color='r', linestyle='dashdot', label = 'Storm Cutoff')
plt.legend()
plt.savefig('Rainfall Cutoff cluster one.png', dpi = 1200)


plt.scatter(data1['rain_inches'], data1['prob_exceed'], color="purple")
plt.xlabel('Rainfall in inches', size = 15)
plt.ylabel('Probability of exceedance(%)', size = 15)
plt.title('Rainfall Frequency Analysis (cluster two)')
plt.axhline(y=10, color='k', linestyle='dashed',label = '10% probability')
plt.axvline(x=np.percentile(data1['rain_inches'], 70),  color='r', linestyle='dashdot', label = 'Storm Cutoff')
plt.legend()
plt.savefig('Rainfall Cutoff cluster two.png', dpi = 1200)

plt.scatter(data0['rain_inches'], data0['prob_exceed'])
plt.xlabel('Rainfall in inches', size = 15)
plt.ylabel('Probability of exceedance(%)', size = 15)
plt.title('Rainfall frequency Analysis (cluster two)')
plt.axhline(y=10, color='k', linestyle='dashed',label = '10% probability')
plt.axvline(x=np.percentile(data0['rain_inches'], 95),  color='r', linestyle='dashdot', label = 'Estimate of Storm Flow')
plt.legend()

####function that defines a storm based on flow data####################
# DateRead = cluster_two['DateRead']
# flowMax = np.percentile(cluster_two['influent_mgd'], 95)
DateRead = data0['DateRead']
baseFlow = np.percentile(data0['influent_mgd'], 95)
baseRain = np.percentile(data0['rain_inches'], 70)
#storm = (data0['influent_mgd'] >= baseFlow) & (data0['rain_inches'] >= baseRain)
stormDateRead = []
            
def storm_date(flowData, rainData):
    for i in range(0, len(data0)):
        if flowData[i] >= baseFlow and rainData[i] >= baseRain:
            stormDateRead = DateRead[i] 
            print('stormDay', stormDateRead)
            
stormDay = storm_date(data0.influent_mgd, data0.rain_inches)

DateRead = data1['DateRead']
baseFlow = np.percentile(data1['influent_mgd'], 95)
baseRain = np.percentile(data1['rain_inches'],70)

#storm = (data1['influent_mgd'] >= baseFlow) & (data1['rain_inches'] >= baseRain)

stormDateRead = []
        
def storm_date(flowData,rainData):
    for i in range(0, len(data1)):
        if flowData[i] >= baseFlow and rainData[i] >= baseRain :
            stormDateRead = DateRead[i] 
            print('stormDay', stormDateRead)
            
stormDay = storm_date(data1.influent_mgd, data1.rain_inches)

