"""
This script contains plots of the visualizations contained in Musaazi et al. 2022...

Author: 
    Isaac Musaazi 
Latest version: 
    March 21, 2022 @ 9p.m
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import missingno as msno
from matplotlib import rcParams
import matplotlib.dates as mdates
import datetime as date
import shap
from sklearn import metrics 
from sklearn.tree import export_graphviz
import pydot

#####plot for rainfall information#######33
dmass = pd.read_csv('../Analysis_02202022/doublemassmethod.csv')
X = dmass['AVERAGECUM']
Y = dmass['STA1']
Y1 = dmass['STA2']
Y2 = dmass['STA3']

fig, ax = plt.subplots(figsize=(12, 12))
annotations=["2015","2016","2017","2018","2019"]
my_cmap = cm.get_cmap('seismic') 
my_cma = cm.get_cmap('bwr') 


plt.scatter(X,Y,s=50,color =my_cma(0.0))
plt.plot(X, Y, '--', linewidth=2, color =my_cma(0.0), label= 'STATION 1')

plt.scatter(X,Y1,s=50,color =my_cmap(0.0))
plt.plot(X, Y1, '-.', linewidth=4, color =my_cmap(0.0), label = 'STATION 2')

plt.scatter(X,Y2,s=50,color =my_cmap(0.9))
plt.plot(X, Y2, '-.', linewidth=2, color= my_cmap(0.9), label ='STATION 3')

plt.xlabel("CUMMULATIVE PREC. FOR PATTERN, IN INCHES", fontsize =12)
plt.ylabel("CUMMULATIVE PREC. FOR INDIVIDUAL STATIONS, IN INCHES", fontsize=12)
for i, label in enumerate(annotations):
    plt.annotate(label, (X[i], Y[i]),fontsize = 12)
    plt.annotate(label, (X[i], Y1[i]), fontsize=12)
ax.legend(loc ='best', fontsize =15)
plt.savefig('DoubleMassCurve.png',dpi = 1200)

#####plot rainfall missing information#######
rainMiss=pd.read_csv('../Analysis_02202022/hourlyrain.csv')
rainMiss = rainMiss.loc[:,['datetime','station1', 'station2', 'station3']]
rainMiss['datetime'] = pd.to_datetime(rainMiss['datetime'])
rainMiss=rainMiss.loc[(rainMiss['datetime'] >= '2018-4-1') & (rainMiss['datetime'] < '2019-11-1')]
rainMiss = rainMiss.loc[:,['station1', 'station2', 'station3']]
#msno.matrix(rainMiss)
my_cmap = cm.get_cmap('seismic') 
msno.bar(rainMiss,figsize=(10,5), fontsize=10,color=my_cmap(0.2))
plt.savefig('missingD visualization.png',dpi = 1200)

####box plots original data############
dcBlue = pd.read_csv('../DC Water Data October 2021/dcWaterStorms(October 2021)/BluePlainsFinal(October2021)_Update(Feb21).csv')
dcBlue['datetime'] = pd.to_datetime(dcBlue['datetime'])
dcBlue['Year'] = dcBlue['datetime'].dt.year
dc2018 = dcBlue[dcBlue["Year"] == 2018]
dc2019 = dcBlue[dcBlue["Year"] == 2019]

dc2018.boxplot(column = ['influent_flow_imp'], by = ['season', 'event'], notch=True, patch_artist=True)
plt.xticks(rotation=10)
plt.suptitle('Grouping Influent Flow, 2018')
plt.gca().set_title("")
plt.ylabel('Infuent Flow, in MGD')
plt.xlabel('Season and Weather Event')
plt.savefig('2018_Boxplot',dpi = 1200)

dc2019.boxplot(column = ['influent_flow_imp'], by = ['season', 'event'], notch=True, patch_artist=True)
plt.xticks(rotation=10)
plt.suptitle('Grouping Influent Flow, 2019')
plt.gca().set_title("")
plt.ylabel('Infuent Flow, in MGD')
plt.xlabel('Season and Weather Event')
plt.savefig('2019_Boxplot',dpi = 1200)

##define duplicates and plot ############
dupCounts = pd.read_csv('../DC Water Data October 2021/duplicates.csv')
dupCounts['Time'] = pd.to_datetime(dupCounts['Time'])

fig, ax = plt.subplots(figsize=(12, 12))
my_cmap = cm.get_cmap('seismic') 
ax.bar(dupCounts.Time, dupCounts.hours, color=my_cmap(0.2))

# Set title and labels for axes
ax.set(xlabel="Date of Sensor Reading",
       ylabel="Hours",
       title="No change in Effluent Ammonia, Nitrite, and Nitrate Sensor Measurements\n 2018")
ax.set_xlim([date.date(2018, 5, 20), date.date(2018, 12, 20)])
ax.set_ylim(0.0, 10.0, 1.0)

date_form = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_form)
week = mdates.WeekdayLocator(interval =2)
day = mdates.DayLocator()
ax.xaxis.set_major_locator(week)
ax.xaxis.set_minor_locator(day) ###defines minor axis every day
fig.autofmt_xdate() # Rotates and right aligns the x labels
plt.savefig('duplicates1.png',dpi = 1200)

fig, ax = plt.subplots(figsize=(12, 12))
ax.bar(dupCounts.Time, dupCounts.hours,
       color=my_cmap(0.2))

ax.set(xlabel="Date of Sensor Reading",
       ylabel="Hours",
       title="No change in Effluent Ammonia, Nitrite, and Nitrate Measurements\n 2019")
ax.set_xlim([date.date(2019, 1, 10), date.date(2019, 11, 10)])
#ax.set_ylim(0.0, 10.0, 1.0)

date_form = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_form)
week = mdates.WeekdayLocator(interval =2)
day = mdates.DayLocator(interval = 5)
ax.xaxis.set_major_locator(week)
ax.xaxis.set_minor_locator(day) ###defines minor axis every day
fig.autofmt_xdate() # Rotates and right aligns the x labels
plt.savefig('duplicates2.png',dpi = 1200)

#need to first define extreme influent flow conditions to define an extreme wet weather event########
dcBlue = pd.read_csv('../DC Water Data October 2021/dcWaterStorms(October 2021)/BluePlainsFinal(October2021)_Update(Feb21).csv') ##effluent quality and precipitation
dcBlue['datetime'] = pd.to_datetime(dcBlue['datetime'])

counts, bins = np.histogram(dcBlue['influent_flow_imp'], bins = 10)
pdf = counts / sum(counts)  ####
cdf = np.cumsum(pdf)
my_cmap = cm.get_cmap('seismic') 
plt.style.use('seaborn-bright')
plt.xlabel('Influent Flow,in MGD', size = 15)
plt.ylabel('Probability Distribution', size = 15)
plt.plot(bins[1:],pdf,color=my_cmap(0.), label="pdf", linewidth =3)
plt.plot(bins[1:], cdf, color=my_cmap(1.0),label="cdf", linewidth = 3)
plt.axhline(y=0.95, linestyle='dashed',label = '95th percentile', color="k")
plt.axvline(x=np.percentile(dcBlue['influent_flow_imp'], 95),  color='k', linestyle='dotted', label = 'Wet Weather Flow')
plt.legend(loc= 'best', fontsize=18)
plt.savefig('probability distribution.png',dpi = 1200)
# =============================================================================
#####wet vs dry weather plots #######
##Whole DataSet#############
countDry = len(dcBlue[dcBlue.event == 'Dry'])
countWet = len(dcBlue[dcBlue.event == 'Wet'])
totalEvent = len(dcBlue)
data = pd.DataFrame({'Weather':['Dry','Wet'],
                      'Count': [countDry, countWet],
                      'Percent':["{:.2%}".format(countDry/totalEvent), "{:.2%}".format(countWet/totalEvent)]
                    }) 
my_cmap = cm.get_cmap('seismic')
plt.figure(figsize=(8,8))
graph = plt.bar(data.Weather,data.Count, color=my_cmap(data.Count))
plt.xlabel('Type of Weather Event', family='serif',fontweight = 'heavy',fontsize = 12)
plt.ylabel('Total Number of Data Points', family='serif', fontweight = 'heavy',fontsize = 12)

i = 0
for p in graph:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plt.text(x+width/2,
              y+height*1.01,
              str(data.Percent[i]),
              ha='center',
              weight='bold')
    i+=1
#plt.show()    
plt.savefig('Weather Event Data Count(WholeData).png',dpi = 1200)
  
######training Dataset(NonBalanced)##########
countDry = len(train_dcBlue[train_dcBlue.event == 'Dry'])
countWet = len(train_dcBlue[train_dcBlue.event == 'Wet'])
totalEvent = len(train_dcBlue)
data = pd.DataFrame({'Weather':['Dry','Wet'],
                      'Count': [countDry, countWet],
                      'Percent':["{:.2%}".format(countDry/totalEvent), "{:.2%}".format(countWet/totalEvent)]
                    }) 
my_cmap = cm.get_cmap('seismic')
plt.figure(figsize=(8,8))
graph = plt.bar(data.Weather,data.Count, color=my_cmap(data.Count))
plt.xlabel('Type of Weather Event', family='serif',fontweight = 'heavy',fontsize = 12)
plt.ylabel('Total Number of Data Points', family='serif', fontweight = 'heavy',fontsize = 12)

i = 0
for p in graph:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plt.text(x+width/2,
              y+height*1.01,
              str(data.Percent[i]),
              ha='center',
              weight='bold')
    i+=1
plt.savefig('Weather Event Data Count(NonBalanced).png',dpi = 1200)

######training Dataset(Balanced)##########

countDry = len(trainBalanced[trainBalanced.event == 'Dry'])
countWet = len(trainBalanced[trainBalanced.event == 'Wet'])
totalEvent = len(trainBalanced)
data = pd.DataFrame({'Weather':['Dry','Wet'],
                      'Count': [countDry, countWet],
                      'Percent':["{:.2%}".format(countDry/totalEvent), "{:.2%}".format(countWet/totalEvent)]
                    }) 
my_cmap = cm.get_cmap('RdBu')
plt.figure(figsize=(8,8))
graph = plt.bar(data.Weather,data.Count, color=my_cmap(data.Count))
plt.xlabel('Type of Weather Event', family='serif',fontweight = 'heavy',fontsize = 12)
plt.ylabel('Total Number of Data Points', family='serif', fontweight = 'heavy',fontsize = 12)

i = 0
for p in graph:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plt.text(x+width/2,
              y+height*1.01,
              str(data.Percent[i]),
              ha='center',
              weight='bold')
    i+=1
plt.savefig('Weather Event Data Count(BalancedROS).png',dpi = 1200)
  

######Training and validation plots ###########
fig, ax = plt.subplots(figsize=(12, 12))
plt.scatter(train_labels, train_predictions, c = 'blue', marker = '.', label = 'Training Set')
plt.scatter(test_labels, test_predictions, c = 'red', marker = '.', label = 'Testing Set')
plt.xlabel('Measured Influent Flow, in MGD', fontsize=15)
plt.ylabel('Predicted Influent Flow, in MGD', fontsize=15)
plt.text(450, 210, 'Training MAPE: 11.54%\n' 'Testing MAPE: 33.83%\n''Training MAE: 36.20 MGD\n''Testing MAE: 81.64 MGD', fontsize=12, fontweight = 'bold' )
ax.axline([0, 0], [1, 1], transform=ax.transAxes, linewidth = 3.5, color ='black')
ax.set_aspect('equal')
plt.legend(loc = 'upper left', fontsize=11)
plt.savefig('TrainingValidationPlotNoBal.png',dpi = 1200)


#######Measured vs Predicted Flow Plots ###################
testingPred = pd.read_csv('../DC Water Data October 2021/dcWaterStorms(October 2021)/BluePlainsTestingPredictions(March2022).csv')
testingPred['datetime'] = pd.to_datetime(testingPred['datetime'])

fig, ax = plt.subplots(figsize=(12, 12))
ax.plot('datetime', 'influent_measured',linestyle = 'dashdot', label ='Measured Influent Flow, in MGD',linewidth =2, data = testingPred)
ax.plot('datetime', 'influent_nobal',linestyle = 'dashed', label ='Predicted Influent Flow , in MGD ', linewidth =2, data = testingPred)
ax.plot('datetime', 'influent_bal',linestyle = 'dashed', label ='Predicted Influent Flow, in MGD (Balanced) ', linewidth =2, data = testingPred)

ax1 = ax.twinx()
ax1.bar(testingPred.datetime, testingPred.mean_prcp,
       color='mediumblue', label = 'Rainfall')


# Set title and labels for axes
ax.set(xlabel="Time of the Year",
       ylabel="Influent Flow, in MGD",
       title="Measured and Predicted Influent Flow\n September, 2019")
ax1.set_ylabel('Rainfall, in INCHES') 

ax.set_xlim([date.date(2019, 9, 3), date.date(2019, 9, 30)])

date_form = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_form)
week = mdates.WeekdayLocator(interval =1)
day = mdates.DayLocator(interval =1)
ax.xaxis.set_major_locator(week)
ax.xaxis.set_minor_locator(day) ###defines minor axis every day
fig.autofmt_xdate() # Rotates and right aligns the x labels
ax.legend(prop={"size":16}, loc = 2)
ax1.legend(prop={"size":16}, loc = 'best')
plt.savefig('MeasuredvsPredictedFlow(Sep)_Update(March2022).png',dpi = 1200)

fig, ax = plt.subplots(figsize=(12, 12))
ax.plot('datetime', 'influent_measured',linestyle = 'dashdot', label ='Measured Influent Flow, in MGD',linewidth =2, data = testingPred)
ax.plot('datetime', 'influent_nobal',linestyle = 'dashed', label ='Predicted Influent Flow , in MGD ', linewidth =2, data = testingPred)
ax.plot('datetime', 'influent_bal',linestyle = 'dashed', label ='Predicted Influent Flow, in MGD (Balanced) ', linewidth =2, data = testingPred)

ax1 = ax.twinx()
ax1.bar(testingPred.datetime, testingPred.mean_prcp, color ='mediumblue', label = 'Rainfall')


# Set title and labels for axes
ax.set(xlabel="Time of the Year",
       ylabel="Influent Flow, in MGD",
       title="Measured and Predicted Influent Flow\n October, 2019")
ax1.set_ylabel('Rainfall, in INCHES') 

ax.set_xlim([date.date(2019, 9, 30), date.date(2019, 10, 31)])

date_form = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_form)
week = mdates.WeekdayLocator()
day = mdates.DayLocator()
ax.xaxis.set_major_locator(week)
ax.xaxis.set_minor_locator(day) ###defines minor axis every day
fig.autofmt_xdate() # Rotates and right aligns the x labels
ax.legend(prop={"size":16}, loc=2)
ax1.legend(prop={"size":16})
plt.savefig('MeasuredvsPredictedFlow(October)_Update(March2022).png',dpi = 1200)

plt.scatter(testingPred.influent_measured, testingPred.influent_pred_bal_SMOTE)
testFlow = testingPred.loc[:, ['influent_measured', 'influent_pred_bal_SMOTE']]
tranData = np.log(testFlow).diff()
tranData.dropna(inplace=True)
rcParams['figure.figsize'] = 8,6
plt.style.use('seaborn')
plt.axvline(0, ls ='--', linewidth = 3, color ='k')
plt.axhline(0, ls ='--', linewidth = 3, color ='k')
plt.scatter(tranData.influent_measured, tranData.influent_pred_bal_SMOTE, c=tranData.influent_pred_bal_SMOTE,cmap="plasma", edgecolor="k")
plt.text(-0.3,0.5, '(A)', {'fontsize':20, 'fontweight': 'bold'})
plt.text(0.4,0.5, '(B)', {'fontsize':20, 'fontweight': 'bold'})
plt.text(0.4,-0.3, '(D)', {'fontsize':20, 'fontweight': 'bold'})
plt.text(-0.3,-0.3, '(C)', {'fontsize':20, 'fontweight': 'bold'})
plt.xlim(-0.4,0.6)
plt.ylim(-0.4,0.6)    
plt.colorbar()
plt.savefig('MeasuredvsPredictedSMOTE.png', dpi=1200)

######plotting balanced datasets###################
rcParams['font.family'] = 'serif'
my_cmap = cm.get_cmap('seismic')
sns.kdeplot(train_dcBlue['influent_flow'], color=my_cmap(0.25), label ='Imbalanced')
sns.kdeplot(trainBal_labels,label = "Balanced", color=my_cmap(0.8))
plt.xlabel('Influent Flow, IN MGD', fontsize=10)
plt.ylabel('Kernel Density Estimate', fontsize=10)
plt.legend(loc = 'best', fontsize =12)
plt.savefig('BalancedvsNonBalanced.png', dpi=1200)

#######time lag correlation heatmap##########
timeLag = pd.read_csv('../DC Water Data October 2021/timelag_correlation.csv')
fig, ax = plt.subplots(figsize=(8, 8))
my_cmap = cm.get_cmap('seismic')
ax.plot('lag', 'rainfall',linestyle = '-', linewidth =2, label = 'rainfall' , color=my_cmap(0), data = timeLag)
ax.plot('lag', 'ammonia',linestyle = '-', linewidth =2,  label = 'ammonia' ,color=my_cmap(0.3), data = timeLag)
ax.plot('lag', 'nitrate',linestyle = '-', linewidth =2,  label = 'nitrate' ,color=my_cmap(0.6), data = timeLag)
ax.plot('lag', 'nitrite',linestyle = '-', linewidth =2, label = 'nitrite' ,color=my_cmap(0.9),  data = timeLag)
ax.set_ylabel('cross correlation') 
ax.set_xlabel('Hourly Lag') 
plt.legend(fontsize =12)
plt.savefig('cross correlation plot version2.png', dpi=1200)

corHeatmap= pd.read_csv('../DC Water Data October 2021/corrtest.csv')
corHeatmap = corHeatmap.set_index('Unnamed: 0')
plt.figure(figsize=(20, 20))
heatmap = sns.heatmap(corHeatmap2, vmin=-1, vmax=1, annot=False, cmap='viridis')
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
