"""
This script ...

Author: 
    Isaac Musaazi 
Latest version: 
    November 11, 2021 @ 9:20a.m
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as date
from statsmodels.graphics.tsaplots import plot_acf


#####autocorrelation btn same variables - forecasting####
dcBlue = pd.read_csv('../DC Water Data October 2021/dcWaterStorms(October 2021)/BluePlainsFinal(October2021)_Update(Dec14).csv')
dcBlue['datetime'] = pd.to_datetime(dcBlue['datetime'])
dcBlue = dcBlue.groupby(pd.Grouper(freq='d', key='datetime'))['mean_prcp_imp',
       'influent_flow_imp', 'ammonia_ppm_imp', 'nitrate_ppm_imp',
       'nitrite_ppm_imp'].mean().reset_index()

df_ammonia = dcBlue[['datetime', 'ammonia_ppm_imp']].set_index(['datetime'])
df_nitrate = dcBlue[['datetime', 'nitrate_ppm_imp']].set_index(['datetime'])
df_nitrite = dcBlue[['datetime', 'nitrite_ppm_imp']].set_index(['datetime'])
df_precip = dcBlue[['datetime', 'mean_prcp_imp']].set_index(['datetime'])
df_influent = dcBlue[['datetime', 'influent_flow_imp']].set_index(['datetime'])


fig, (ax,ax1) = plt.subplots(1,2,figsize=(10,5))
ax.set_ylabel("Autocorrelation coefficient")
ax.set_xlabel("Time Lag")
ax1.set_xlabel("Time Lag")
plot_acf(df_ammonia, lags= 4, ax= ax, title="ammonia autocorrelation plot")
plot_acf(df_influent, lags= 25,ax = ax1, title= "influent flow autocorrelation plot")
plt.savefig('ammonia and influent flow correlation plots.png',dpi = 1200)

fig, (ax,ax1) = plt.subplots(1,2,figsize=(10,5))
ax.set_ylabel("Autocorrelation coefficient")
ax.set_xlabel("Time Lag")
ax1.set_xlabel("Time Lag")
plot_acf(df_ammonia, lags= 4, ax= ax, title="ammonia autocorrelation plot")
plot_acf(df_nitrate, lags= 25,ax = ax1, title="nitrate autocorrelation plot")
plt.savefig('ammonia and nitrate correlation plots.png',dpi = 1200)

fig, (ax,ax1) = plt.subplots(1,2,figsize=(10,5))
ax.set_ylabel("Autocorrelation coefficient")
ax.set_xlabel("Time Lag")
ax1.set_xlabel("Time Lag")
plot_acf(df_precip, lags= 3, ax= ax, title="precipitation autocorrelation plot")
plot_acf(df_nitrite, lags= 3,ax = ax1, title="nitrite autocorrelation plot")
plt.savefig('precipitation and nitrite correlation plots.png',dpi = 1200)


