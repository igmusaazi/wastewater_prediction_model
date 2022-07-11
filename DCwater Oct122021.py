"""
This script reads the DC water flow data and identifies storm events based
    on a threshold flow rate and rainfall depth
    
October 12 - 10 min flow data was provided, storm events were extracted at that interval
Author: 
    Isaac Musaazi 
Latest version: 
    October 12, 2021 @ 8:00p.m
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
##blank_prec = prec.loc[prec['PRCP'].isnull()] #select missing values in a given column

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

cluster =KMeans(n_clusters = 2) #divide data into two clusters based on treated flow, ammonia, nitrate, rainfall depth
flowRain['cluster'] = cluster.fit_predict(flowRain[flowData.columns[1:5]]) #generate column defined by two clusters

###plot influent flow and rainfall over time

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

###plot cdf and pdf based on influent flow rate##########
def cum_plot(data):
    '''Function used to plot the cumulative distribution function to determine the flow rate that constitutes
    a storm event. In this function the 70% percentile is the cutoff'''
    
    counts, bins = np.histogram(data['influent_mgd'], bins = 10)
    pdf = counts / sum(counts)  
    cdf = np.cumsum(pdf)
            
    plt.xlabel('Flow in MGD', size = 15)
    plt.ylabel('Cumulative Probability', size = 15)
    plt.title('Cumulative Probability and Flow') 
    plt.plot(bins[1:],pdf, color="red", label="PDF")
    plt.plot(bins[1:], cdf, label="CDF")
    plt.axhline(y = 0.70, color='k', linestyle='dashed',label = '70th percentile')
    plt.axvline(x=(np.percentile(data['influent_mgd'],70)), color='r', linestyle='dashdot', label = 'Influent Flow Threshold')
    plt.legend()
    return plt.savefig('Storm Threshold cluster one.png', dpi = 1200)


cum_plot(flowRain[flowRain['cluster'] == 0].reset_index(drop= True)) ##cluster one
cum_plot(flowRain[flowRain['cluster'] == 1].reset_index(drop= True)) ##cluster two

#estimate the probability of exceedance from rainfall data 
flowRain['rank'] = flowRain['rain_inches'].rank(ascending=False,method='dense') #rank in descending order
observation = len(flowRain['rank'].unique())
flowRain['prob_exceed'] = ((flowRain['rank'] - 0.44)/(observation + 0.12))*100 #uses the Gringorten method to determine the dependable rainfall

def rainfall_plot(rain):
    '''Function used for rainfall frequency analysis to determine the rainfall depth that constitutes
    a storm event. In this current function a 10% probability of exceedance is the cutoff'''
    
    plt.scatter(rain['rain_inches'], rain['prob_exceed'], color="purple")
    plt.xlabel('Rainfall in inches', size = 15)
    plt.ylabel('Probability of exceedance(%)', size = 15)
    plt.title('Rainfall Frequency Analysis')
    plt.axhline(y=10, color='k', linestyle='dashed',label = '10% probability')
    plt.axvline(x=1.5,  color='r', linestyle='dashdot', label = 'Rainfall Threshold')
    plt.legend()
    return plt.savefig('Rainfall Cutoff cluster two.png', dpi = 1200)

rainfall_plot(flowRain[flowRain['cluster'] == 0].reset_index(drop= True)) ##cluster one
rainfall_plot(flowRain[flowRain['cluster'] == 1].reset_index(drop= True)) ##cluster two

# identify storm events from flow and rainfall data
def storm_day(storm):
    '''this function considers a heavy rainfall event to have >1.5 inches recorded '''
    baseFlow = np.percentile(storm['influent_mgd'], 70)
    # baseRain = np.percentile(storm['rain_inches'], 98)
    
    for i in range(0, len(storm)): 
        rain = storm['rain_inches']
        flow = storm['influent_mgd']
        DateRead = storm['DateRead']
        if flow[i] >= baseFlow and rain[i] >= 1.5: 
            stormDay = DateRead[i]
            with open('identified_stormDay.csv', 'a') as output:
                print('stormDay', stormDay, file = output)
            
storm_day(flowRain[flowRain['cluster'] == 0].reset_index(drop= True))
storm_day(flowRain[flowRain['cluster'] == 1].reset_index(drop= True))           

### extract storm event from 10 min flow data #############
waterDCten = pd.read_excel('../Python/DC Water Data October 2021/Effluent Chemscan.xlsx', header = 2)                
waterDCten['Date'] = pd.to_datetime(waterDCten['Date'],errors = 'coerce') ##ignore warning about timezone name
waterDCten = waterDCten.loc[:, ['Date', 'AAI39721', 'ANI39721', 'ANI39722', 'AXI39723']]
waterDCten.rename(columns = {'Date':'DateRead','AAI39721':'ammonia_ppm','ANI39721':'nitrate_ppm', 'ANI39722': 'nitrite_ppm','AXI39723':'orthophosphate'}, inplace = True)

def stormTen(starttime, endtime):
        for i in range(0, len(waterDCten['DateRead'])):
            waterDCten[(waterDCten['DateRead'] >= 'starttime') & (waterDCten['DateRead'] <= 'endtime')]

stormTen('2018-09-07 23:00:00','2018-09-10 23:00:00')        