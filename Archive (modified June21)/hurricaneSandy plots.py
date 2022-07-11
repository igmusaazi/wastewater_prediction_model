"""
This script plots the chabges of treatment performance during Hurricane Sandy. 

Author: 
    Isaac Musaazi
Created:
    June 10, 2021
    
Latest update: 
    June 10, 2021
"""
import pandas as pd
import numpy as np
import datetime as date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

hurrSandy = pd.read_excel('../Python/Effluent data_Hurricane Sandy Jun092021.xlsx', header =2)
hurrSandy = hurrSandy.loc[:, [' Date', 'AAI39721', 'ANI39721','FI_PLTINF','FI_CMPTRT']]
hurrSandy.rename(columns ={' Date': 'DateRead', 'AAI39721':'ammonia_ppm','ANI39721':'nitrate_ppm', 'FI_PLTINF':'influent_mgd','FI_CMPTRT':'treated_mgd'}, inplace = True)
hurrSandy['DateRead']= pd.to_datetime(hurrSandy['DateRead'])
timeDay = pd.Series(range(len(hurrSandy['DateRead'])), index=hurrSandy['DateRead'])/24
hurrSandy['time'] = timeDay.reset_index(drop = True)
hurrSandy['performance'] = (1 - ((hurrSandy['influent_mgd']-hurrSandy['treated_mgd'])/(hurrSandy['influent_mgd'])))*100

#plot the changes to water quality parameters and flow rates over time
fig, ax = plt.subplots()
ax.plot('DateRead', 'influent_mgd', color = 'maroon',label ='Influent Flow', data = hurrSandy)
ax.plot('DateRead', 'treated_mgd', color = 'cornflowerblue', label = 'Treated Flow', data = hurrSandy)
plt.legend()
three_month_interval = mdates.MonthLocator(interval=3)
plt.title('Hurricane Sandy Water Quality Changes over Time', fontweight = 'bold')
ax.xaxis.set_major_locator(three_month_interval) ###defines major axis every 6 months
month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(month) ###defines minor axis every month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  
ax.set_ylabel('Flow Rate(MGD)', fontweight ='normal')
ax.tick_params(axis = 'y', labelcolor='tab:red', labelsize='medium', width=2)

ax.yaxis.label.set_color('tab:red')
ax.set_xlabel('Date', fontweight = 'bold')
fig.autofmt_xdate() # Rotates and right aligns the x labels
ax1 = ax.twinx()
ax1.plot('DateRead', 'nitrate_ppm',color = 'black', label ='Nitrate', data = hurrSandy)
ax1.plot('DateRead', 'ammonia_ppm',color = 'indigo', label = 'Ammonia', data = hurrSandy)
ax1.set_ylabel('Effluent pollutant concentration (ppm)', fontweight = 'normal') 
ax1.tick_params(axis = 'y', labelcolor='k', labelsize='medium', width=2)
    
ax1.yaxis.label.set_color('black')
ax1.invert_yaxis()
plt.legend()
plt.savefig('Flow Rate and Effluent pollutant.png',dpi = 1200)
  
#plot treatment performance in form of changes to the flow rate 

fig, ax = plt.subplots()
ax.plot('time', 'performance', color = 'crimson',label ='Treatment of Flow Performance', data = hurrSandy)
plt.title('Flow Treatment Changes during Hurricane Sandy', fontweight = 'bold',fontsize = 15)
ax.set_ylabel('Treated Flow (%)', fontweight ='bold',fontsize = 15)
ax.tick_params(color = 'black', labelcolor='black',labelsize = 'large', width=2)
# ax.set_ylim(40, 110)
ax.yaxis.label.set_color('crimson')
ax.set_xlabel('Time (Days)', fontweight = 'bold',fontsize = 15)
plt.savefig('Flow Treatment Changes.png',dpi = 1200)
