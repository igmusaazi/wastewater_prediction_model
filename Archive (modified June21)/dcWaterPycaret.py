"""
 
(1) develops several regression models in pycaret. the training and testing data are defined.
(2) anomaly detection
Author: 
    Isaac Musaazi 
Latest version: 
    November 24, 2021 @ 12:00p.m
"""
import os
from pycaret.regression import * 
from pycaret.anomaly import *
import pandas as pd
import numpy as np
import datetime as date
import plotly.graph_objects as go

##develop regression models########
dcBlue = pd.read_csv('../dcWaterStorms(October 2021)/BluePlainsFinal(October2021).csv') 
dcBlue['datetime'] = pd.to_datetime(dcBlue['datetime'])
###dcBlue['Day'] = dcBlue['datetime'].dt.strftime('%A').reset_index(drop=True)
###dcBlue = dcBlue.groupby(pd.Grouper(key='datetime', freq='1D'))['influent_flow_imp'].mean().reset_index()
#dcBlue = dcBlue.loc[:, ['prcp_hr', 'complete_flow_imp', 'ammonia_ppm_imp', 'nitrate_ppm_imp', 'nitrite_ppm_imp', 'influent_flow_imp']]

#dcBlue = dcBlue.loc[:, ['prcp_hr', 'complete_flow_imp', 'Day','ammonia_ppm_imp', 'nitrate_ppm_imp', 'nitrite_ppm_imp', 'influent_flow_imp']]
#compare influent flow at different weekdays
#dcBlueTime = dcBlue.groupby(pd.Grouper(key='datetime', freq='60min'))['influent_flow_imp'].mean().reset_index()
#dcBlueTime['Day'] = dcBlueTime['datetime'].dt.day_name()
#dcBlueTime['Time'] = dcBlueTime['datetime'].dt.strftime('%H:%M:%S')

#dcBlueM = dcBlueTime[dcBlueTime["Day"]=="Monday"]
#dcBlueS = dcBlueTime[dcBlueTime["Day"]=="Sunday"]
#dcBlueT = dcBlueTime[dcBlueTime["Day"]=="Tuesday"]

# =============================================================================
# fig, ax = plt.subplots()
# ax.plot('datetime', 'influent_flow_imp', color = 'cornflowerblue',linestyle = 'solid',label ='Influent Flow (MGD)', data = dcBlueM)
# ax.plot('datetime', 'influent_flow_imp', color = 'red',linestyle = 'solid',label ='Influent Flow (MGD)', data = dcBlueT)
# ax.set_ylabel('Influent flow (MGD)', fontweight = 'bold')
# ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=None, useLocale=None, useMathText=True)
# ax.tick_params(color = 'black', labelcolor='black',labelsize = 'large', width=2)
# plt.legend(loc = 2)
# ax1 = ax.twinx()
# ax1.plot('datetime', 'influent_flow_imp',color = 'black', linestyle = 'solid', label = 'Influent Flow (MGD)', data = dcBlueS)
# ax1.set_ylabel('Influent flow (MGD)', fontweight = 'bold') 
# ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=None, useLocale=None, useMathText=True)
# ax1.tick_params(axis = 'y', labelcolor='k', labelsize='medium', width=2)
# plt.legend(loc = 2)
# 
# plt.title('Daily Flow)', fontweight = 'bold',fontsize = 12)
# plt.legend(loc = 1)
# one_month_interval = mdates.HourLocator(interval=1)
# month = mdates.MonthLocator()
# ax.xaxis.set_minor_locator(month) ###defines minor axis every month
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H-%M')) 
# ax.set_xlabel('Year-Month', fontweight = 'bold',fontsize = 12)
# fig.autofmt_xdate() # Rotates and right aligns the x labels
# plt.savefig('BluePlains Flow and Rainfall.png',dpi = 1200)
# 
# =============================================================================
# =============================================================================
# regBlue = setup(data = dcBlueTrain, target = 'influent_flow_imp',
# , session_id = None,fold_shuffle=True,imputation_type='iterative')
# dcBest = compare_models(exclude = ['ransac'])
# =============================================================================

#dcBlue = dcBlue.loc[:, ['mean_prcp_imp','prcp_1yr','prcp_2yr','prcp_5yr','int_one', 'int_two','int_five','influent_flow_imp','complete_flow_imp','ammonia_ppm_imp', 'nitrate_ppm_imp', 'nitrite_ppm_imp',
#       'TIN','performance']]

dcBlue = dcBlue.loc[:, ['int_one', 'int_two','int_five','influent_flow_imp','ammonia_ppm_imp', 'nitrate_ppm_imp', 'nitrite_ppm_imp',
     'TIN']]


dcBlueTrain = dcBlue.sample(frac=0.8, random_state=786)
dataTest = dcBlue.drop(dcBlueTrain.index)

regBlue = setup(data = dcBlueTrain, target = 'influent_flow_imp',session_id = 123, normalize= True)
best_model = compare_models()
model_results = pull()   ####ranks best to worst model based on the R2

etmodel = create_model('et')
rfmodel = create_model('rf')

xgboostmodel = create_model('xgboost')
catboostmodel = create_model('catboost')
etmodel = create_model('et')


et_results = pull()
rf_results = pull()
boost_rf = ensemble_model(rfmodel, method= 'Boosting') ##improve performance of model
boost_results = pull()
# =============================================================================
# tuned_rf = tune_model(rf) #model cannot be tuned
# tuned_rf_results = pull()

# =============================================================================
xgboostmodel = create_model('xgboost')
xgboost_results = pull()

catboostmodel = create_model('catboost')
catboost_results = pull()
plot_model(catboostmodel, save=True)
plot_model(catboostmodel, plot='feature')
plot_model(catboostmodel, plot='error')
finalBlue = finalize_model(catboostmodel)
predBlue = predict_model(finalBlue, data=dataTest)


####detect anomalies in the dataset

dcBlue = pd.read_csv('../dcWaterStorms(October 2021)/BluePlainsFinal(October2021)_Update(Nov23).csv')
dcBlue['datetime'] = pd.to_datetime(dcBlue['datetime'])
# =============================================================================
# dcBlue.set_index('datetime', drop=True, inplace=True) #setting timestamp to index
# dcBlue['day_name'] = [i.day_name() for i in dcBlue.index]
anoMode = setup(dcBlue, normalize = True, ignore_features =['Unnamed: 0'], session_id = 123)
iforest = create_model('iforest')
iforest_results = assign_model(iforest)
iforest_results.head()
plot_model(iforest)

anom = iforest_results[iforest_results['Anomaly'] == 1]
# 
# 
# fig = px.line(iforest_results, x=iforest_results.index, y="prcp_1yr", title='Rainfall Depth - UNSUPERVISED ANOMALY DETECTION', template = 'plotly_dark')
# outlier_dates = iforest_results[iforest_results['Anomaly'] == 1].index
# y_values = [iforest_results.loc[i]['prcp_1yr'] for i in outlier_dates]
# fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
#                 name = 'Anomaly', 
#                 marker=dict(color='red',size=10)))
# fig.show()
# =============================================================================
