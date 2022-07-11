"""
Script converts to 10 min rainfall data from Fourmile, Clarksburg and Slidell run gage stations.
uses station-average method to find average precipitation 

Author: 
    Isaac Musaazi 
Latest version: 
    November 19, 2021 @ 11:00p.m
"""
import os
os.getcwd()
#os.chdir()
import pandas as pd
import numpy as np
from functools import reduce #merging multiple dataframes
import datetime as date
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats


####determine 60 min rainfall for three rain gage stations near BluePlains################
rainFOUR = pd.read_csv('../Rainfall/fourmile.csv')
rainCLARK = pd.read_csv('../Rainfall/clarksburg.csv')
rainSLIDELL = pd.read_csv('../Rainfall/slidell.csv')
rainFOUR['datetime']= pd.to_datetime(rainFOUR['datetime']) ##convert Date column from string to dateDateRead
rainCLARK['datetime']= pd.to_datetime(rainCLARK['datetime']) ##convert Date column from string to dateDateRead
rainSLIDELL['datetime']= pd.to_datetime(rainSLIDELL['datetime']) ##convert Date column from string to dateDateRead

rainFOUR = rainFOUR.groupby(pd.Grouper(key='datetime', freq='60min'))['rain(inches)'].mean().reset_index()
rainCLARK = rainCLARK.groupby(pd.Grouper(key='datetime', freq='60min'))['rain(inches)'].mean().reset_index()
rainSLIDELL = rainSLIDELL.groupby(pd.Grouper(key='datetime', freq='60min'))['rain(inches)'].mean().reset_index()

rain = [rainFOUR, rainCLARK, rainSLIDELL]
rainCOMB = reduce(lambda  left,right: pd.merge(left,right,on=['datetime'],
                                            how='outer'), rain)            #rainfall intensity from three rain gage measurements
rainCOMB.rename(columns ={'rain(inches)_x': 'station1', 'rain(inches)_y':'station2', 'rain(inches)':'station3'}, inplace = True) ###slidell_prcp represents inches/hour
rainCOMB['mean_prp'] = rainCOMB[['fourMile_prcp', 'clarksburg_prcp', 'slidell_prcp']].mean(axis =1, skipna=True)
rainCOMB.isnull().sum() / rainCOMB.count() #percentage of missing values

#######double mass curve plotting#################

dmass = pd.read_csv('../Rainfall/Analysis_02202022/doublemassmethod.csv')

X = dmass['AVERAGECUM']
Y = dmass['STA1']
Y1 = dmass['STA2']
Y2 = dmass['STA3']

annotations=["2015","2016","2017","2018","2019"]

plt.scatter(X,Y,s=50,color="g")
plt.plot(X, Y, 'o--y', linewidth=2, color ='g', label= 'STATION 1')

plt.scatter(X,Y1,s=50,color="r")
plt.plot(X, Y1, ':', linewidth=4, color= 'r', label = 'STATION 2')

plt.scatter(X,Y2,s=50,color="y")
plt.plot(X, Y2, '-.', linewidth=2, color= 'y', label ='STATION 3')

plt.xlabel("CUMMULATIVE PREC. FOR PATTERN, IN INCHES", fontsize =10)
plt.ylabel("CUMMULATIVE PREC. FOR INDIVIDUAL STATIONS, IN INCHES", fontsize=8)
plt.title("Double Mass Curve for Precipitation Data",fontsize=15)
for i, label in enumerate(annotations):
    plt.annotate(label, (X[i], Y[i]),fontsize = 8)
    plt.annotate(label, (X[i], Y1[i]),fontsize = 8)
plt.legend()
plt.savefig('DoubleMassCurve.png',dpi = 1200)

#########missing data piechart##############
labels = 'Station 2', 'Station 1','Station 3', 'Complete'
sizes = [1.51258947, 21.5038261, 0.449375738, 76.53420869]
explode = (0.15, 0.05, 0.15, 0.05)  
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=45)
plt.savefig('MD Pie Chart.png',dpi = 1200)

####filling in missing data################
imputer = KNNImputer(n_neighbors=4)
rainCOMB_imp = imputer.fit_transform(rainCOMB[['mean_prp']])
raincomplete = pd.DataFrame(rainCOMB_imp, columns = ['mean_prcp_imp'])
rainCLEAN = pd.concat([rainCOMB, raincomplete], axis=1)
                             ###rainCLEAN contains both original and imputated rainfall intensity data
###rainCLEAN['prcp']= rainCLEAN['mean_prcp_imp'].diff().abs() find rainfall point estimate divide by time 
rainCLEAN['prcp_1yr']= rainCLEAN['mean_prcp_imp']  #rainfall depth (1 year return period)

rainCLEAN['prcp_2yr']= rainCLEAN['mean_prcp_imp']*1.401 ##rainfall depth (2 year return period) based on a correction factor

rainCLEAN['prcp_5yr']= rainCLEAN['mean_prcp_imp']*1.949 ##rainfall depth (5 year return period) based on a correction factor

rainCLEAN['int_one']= rainCLEAN['prcp_1yr']*(60/10)  ####rainfall intensity
rainCLEAN['int_two']= rainCLEAN['prcp_2yr']*(60/10)  ####rainfall intensity
rainCLEAN['int_five']= rainCLEAN['prcp_5yr']*(60/10)  ####rainfall intensity

waterDCEffluent = pd.read_excel('../Python/DC Water Data October 2021/Effluent Chemscan.xlsx', header = 2)                
waterDCEffluent['Date'] = pd.to_datetime(waterDCEffluent['Date'],errors = 'coerce') ##ignore warning about timezone name
waterDCEffluent = waterDCEffluent.loc[:, ['Date', 'AAI39721', 'ANI39721', 'ANI39722', 'AXI39723']]
waterDCEffluent.rename(columns = {'Date':'DateRead','AAI39721':'ammonia_ppm','ANI39721':'nitrate_ppm', 'ANI39722': 'nitrite_ppm','AXI39723':'orthophosphate'}, inplace = True)

imputer = KNNImputer(n_neighbors=4)
waterEffluent_imp = imputer.fit_transform(waterDCEffluent[['ammonia_ppm','nitrate_ppm','nitrite_ppm','orthophosphate']])
effluentcomplete = pd.DataFrame(waterEffluent_imp, columns = ['ammonia_ppm_imp','nitrate_ppm_imp','nitrite_ppm_imp','orthophosphate_imp'])
effluentCLEAN = pd.concat([waterDCEffluent, effluentcomplete], axis=1)

waterDCFlow = pd.read_csv('../Python/DC Water Data October 2021/FlowData(October2021).csv')
waterDCFlow[' Date'] = pd.to_datetime(waterDCFlow[' Date'],errors = 'coerce') ##ignore warning about timezone name
#waterDCEffluent = waterDCEffluent.loc[:, ['Date', 'AAI39721', 'ANI39721', 'ANI39722', 'AXI39723']]
waterDCFlow.rename(columns = {' Date':'datetime','FI_PLTINF':'influent_flow','FI_EASTINF':'influent_flow_east', 'FI_WESTINF': 'influent_flow_west','FI_CMPTRT':'complete_flow'}, inplace = True)

rainFLOW = pd.merge(left =waterDCFlow,right = rainCLEAN, left_on = 'datetime', right_on ='datetime') #column merge based on 'DateRead'

rainFLOW_imp = imputer.fit_transform(rainFLOW[['influent_flow', 'influent_flow_east', 'influent_flow_west',
       'complete_flow']])
rainFLOWcomplete = pd.DataFrame(rainFLOW_imp, columns = ['influent_flow_imp', 'influent_flow_east_imp', 'influent_flow_west_imp',
       'complete_flow_imp'])
flowCLEAN = pd.concat([rainFLOW,rainFLOWcomplete], axis=1)

rainFLOWcleaned = pd.merge(left =flowCLEAN,right = effluentCLEAN, left_on = 'datetime', right_on ='DateRead') ####combines flow, rain and effluent
rainFLOWcleaned = rainFLOWcleaned.loc[:, ['datetime', 'mean_prcp_imp','prcp_1yr','prcp_2yr','prcp_5yr','int_one', 'int_two','int_five','influent_flow_imp', 'influent_flow_east_imp',
       'influent_flow_west_imp', 'complete_flow_imp','ammonia_ppm_imp', 'nitrate_ppm_imp', 'nitrite_ppm_imp',
       'orthophosphate_imp']]

#rainFLOWcleaned.to_csv('BluePlainsFinal(October2021).csv')

######plotting data################
fig, ax = plt.subplots()
ax.plot('datetime', 'complete_flow_imp', color = 'cornflowerblue',linestyle = 'solid',label ='Effluent flow (MGD)', data = rainFLOWcleaned)
ax.set_ylabel('effluent flow (MGD)', fontweight = 'bold')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=None, useLocale=None, useMathText=True)
ax.tick_params(color = 'black', labelcolor='black',labelsize = 'large', width=2)
plt.legend(loc = 2)
ax1 = ax.twinx()
ax1.plot('datetime', 'prcp_hr',color = 'black', linestyle = 'solid', label = 'rainfall intensity (inches/hr)', data = rainFLOWcleaned)
ax1.set_ylabel('rainfall intensity (inches/hr)', fontweight = 'bold') 
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=None, useLocale=None, useMathText=True)
ax1.tick_params(axis = 'y', labelcolor='k', labelsize='medium', width=2)
plt.legend(loc = 2)

plt.title('Flow and Rainfall (10 min interval)', fontweight = 'bold',fontsize = 12)
plt.legend(loc = 1)
one_month_interval = mdates.HourLocator(interval=1)
month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(month) ###defines minor axis every month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H-%M')) 
ax.set_xlabel('Year-Month', fontweight = 'bold',fontsize = 12)
fig.autofmt_xdate() # Rotates and right aligns the x labels
plt.savefig('BluePlains Flow and Rainfall.png',dpi = 1200)

###extract storms from 10 min flow data#####
dcWater = pd.read_csv('../Python/DC Water Data October 2021/BluePlainsFinal(October2021).csv')
dcWater['datetime']= pd.to_datetime(dcWater['datetime'])
stormEvent= pd.read_csv('../Python/DC Water Data October 2021/dcWater stormEvent.csv')
#stormEvent['stormDay']= pd.to_datetime(stormEvent['stormDay'])
#stormEvent['startDate']= pd.to_datetime(stormEvent['startDate'])
#stormEvent['endDate']= pd.to_datetime(stormEvent['endDate'])

starttime = '2018-07-15 23:00:00'
endtime = '2018-07-20 23:00:00'
#def stormTen(starttime, endtime):for i in range(0, len(dcWater['datetime'])):
storm = dcWater[(dcWater['datetime'] >= starttime) & (dcWater['datetime'] <= endtime)]
            
#storm = pd.read_csv('../DC Water Data October 2021/dcWaterStorms(October 2021)/september2018storm.csv',parse_dates=['datetime'], index_col='datetime')
#storm['datetime']= pd.to_datetime(stormSeptember['datetime'])
#stormTrend = seasonal_decompose(stormSeptember['nitrate_ppm_imp'], model='multiplicative'\
 #                  ,extrapolate_trend='freq', period = 12)

#stormTrend = seasonal_decompose(storm['nitrate_ppm_imp'], model='multiplicative'\
#                 ,extrapolate_trend='freq', period = 120)

storm = pd.read_csv('../DC Water Data October 2021/dcWaterStorms(October 2021)/september2018.csv')
storm['datetime']= pd.to_datetime(storm['datetime'])
timeDay = pd.Series(range(len(storm['datetime'])), index=storm['datetime'])/24
storm['time'] = timeDay.reset_index(drop = True)
storm['performance'] = (1 - ((storm['complete_flow_imp']-storm['influent_flow_imp'])/(storm['influent_flow_imp'])))*100
storm['TIN']= storm['ammonia_ppm_imp']+storm['nitrite_ppm_imp']+storm['nitrate_ppm_imp']

baseline = np.polyfit(storm.time, storm.nitrate_ppm_imp, 0) ##fit data to a zero degree polynomial assuming that pollutant concentration doesnot change over time, ~the average ammonia concn
baseValue = np.poly1d(baseline)[0]

baseline = np.polyfit(storm.time, storm.tot_in_nitr, 0) ##fit data to a zero degree polynomial assuming that pollutant concentration doesnot change over time, ~the average ammonia concn
baseValue = np.poly1d(baseline)[0]


time = storm.time
#performance = storm.performance
#rain = storm.prcp_hr

def recovery_time(conc):
    lower_limit = 0
    upper_limit = 0
    for i in range(0, 305): # to get the greater lower limit for ammonia, change the range to (0, 50)
        
        if conc[i] <= baseValue and conc[i+1] >= baseValue and conc[i+2] >= conc[i+1]:
            
            lower_limit = time[i]
            
            print(f"Lower Limit: {lower_limit}")

    for i in range(305, len(conc)): # use the polyline based on the last 5 points to get the upper limit for ammonia
        
        if conc[i] <= baseValue and conc[i-1] >= baseValue and (conc[i-2] > conc[i-1] or conc[i - 2] == conc[i-1]):
            
            upper_limit = time[i]
            
            print(f"Upper Limit: {upper_limit}")
            
            RT = upper_limit-lower_limit
            
            return RT

recovery_time(storm.tot_in_nitr)

def tepr(conc, time):
    area_conc = np.trapz(conc, time)-max(time)*conc[0] 
    return area_conc

tepr(storm.nitrate_ppm_imp, time)
tepr(storm.tot_in_nitr, time)
storm.tot_in_nitr.max() - baseValue


#mpr = storm.ammonia_ppm_imp.max() - np.mean(baseValue(time))

mpr = storm.nitrate_ppm_imp.max() - baseValue



def recoveryTimeA(conc, performance):
    '''function considers the lower limit for recovery time at the beginning of disturbance
    '''
    lower_limit = 0
    upper_limit = 0
    for i in range(20, len(time)): 
        if conc[i] > baseValue and (conc[i+1] - conc[i]) > 0.3 and performance[i]  < 100:
            lower_limit = time[i]
            print(f" lowerLimit: {lower_limit}")
        if performance[i-2] < performance[i-1] and performance[i-1] < performance[i]:
            upper_limit = time[i]
            print(f" upperLimit: {upper_limit}")
            RT = upper_limit - lower_limit
            return RT

recoveryTimeA(storm.ammonia_ppm_imp, storm.performance) #recovery time for ammonia since the concentration does not change rapidly over time
        
# =============================================================================
# def recoveryTimeB(conc, performance):
#     '''function considers the lower limit for recovery time based on the time with the highest pollutant concentration
#     '''
#     lower_limit = 0
#     upper_limit = 0
#     for i in range(20, len(time)): 
#         if conc[i] > baseValue and conc[i-2] < conc[i-1] and conc[i-1] < conc[i] and conc[i+1] < conc[i] and (performance[i-2] and performance[i-1] > performance[i]):
#             lower_limit = time[i]
#             print(f" lowerLimit: {lower_limit}")
#         if performance[i-2] < performance[i-1] and performance[i-1] < performance[i] and performance[i+1] == performance[i]:
#             upper_limit = time[i]
#             print(f" upperLimit: {upper_limit}")
#             RT = upper_limit - lower_limit
#             return RT
# recoveryTimeB(storm.ammonia_ppm_imp, storm.performance) #recovery time for ammonia since the concentration does not change rapidly over time
# =============================================================================

def maximumPerformanceReduction(conc, performance):
    '''function considers best performance(lowest pollutant concentration) and worst performance(highest pollutant concn)
    '''
    per_best = 0
    per_worst = 0
    for i in range(20, len(time)):
        if conc[i] > baseValue and conc[i-2] < conc[i-1] and conc[i-1] < conc[i] and conc[i+1] < conc[i] and (performance[i-2] and performance[i-1] > performance[i]):
            per_worst = performance[i]
            print(f" Worst Treatment Performnce: {per_worst}")
        if performance[i-2] < performance[i-1] and (performance[i-1] < performance[i]) and performance[i+1] == performance[i]: 
            per_best = performance[i]
            print(f" Best Treatment Performance: {per_best}")
            mpr = per_best - per_worst
            return mpr
maximumPerformanceReduction(storm.ammonia_ppm_imp, storm.performance)

####spearman's correlation
corrMetrics = pd.read_csv('../DC Water Data October 2021/resiliency metrics summary findings (Nov 21).csv')
spear = corrMetrics.corr(method='spearman', min_periods =2)
corrMtrx = scipy.stats.spearmanr(a=corrMetrics,b=None, axis=0)
coefMatrx = pd.DataFrame(corrMtrx[0])
pvalues = pd.DataFrame(corrMtrx[1])