"""
This script plots the chabges of treatment performance during Hurricane Sandy. 

Author: 
    Isaac Musaazi
Created:
    June 10, 2021
    
Latest update: 
    June 30, 2021
"""
import pandas as pd
import numpy as np
import datetime as date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

hurrSandy = pd.read_excel('../Python/Effluent data_Hurricane Sandy Jun092021.xlsx', header =2)
hurrSandy = hurrSandy.loc[:, [' Date', 'AAI39721', 'ANI39721','AXI39723','FI_PLTINF','FI_CMPTRT']]
hurrSandy.rename(columns ={' Date': 'DateRead', 'AAI39721':'ammonia_ppm','ANI39721':'nitrate_ppm', 'AXI39723':'phosphorus_ppm',
'FI_PLTINF':'influent_mgd','FI_CMPTRT':'treated_mgd'}, inplace = True)
hurrSandy['DateRead']= pd.to_datetime(hurrSandy['DateRead'])
timeDay = pd.Series(range(len(hurrSandy['DateRead'])), index=hurrSandy['DateRead'])/24
hurrSandy['time'] = timeDay.reset_index(drop = True)
hurrSandy['performance'] = (1 - ((hurrSandy['influent_mgd']-hurrSandy['treated_mgd'])/(hurrSandy['influent_mgd'])))*100

#plot the changes to water quality parameters and flow rates over time
fig, ax = plt.subplots()
ax.plot('DateRead', 'treated_mgd', color = 'tab:red', label = 'Treated Flow', data = hurrSandy)
plt.legend(loc = 2)
three_month_interval = mdates.MonthLocator(interval=3)
plt.title('Flow and Pollutant Concentration Changes', fontweight = 'bold')
ax.xaxis.set_major_locator(three_month_interval) ###defines major axis every 6 months
month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(month) ###defines minor axis every month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  
ax.set_ylabel('Influent(MGD)', fontweight ='bold')
ax.tick_params(axis = 'y', labelcolor='tab:red', labelsize='medium', width=2)

ax.yaxis.label.set_color('tab:red')
ax.set_xlabel('Date', fontweight = 'bold')
fig.autofmt_xdate() # Rotates and right aligns the x labels
ax1 = ax.twinx()
# ax1.plot('DateRead', 'phosphorus_ppm',color = 'black', label ='Phosphorus', data = hurrSandy)
# ax1.plot('DateRead', 'ammonia_ppm',color = 'indigo', label = 'Ammonia', data = hurrSandy)
ax1.plot('DateRead', 'nitrate_ppm',color = 'black', label = 'Nitrate', data = hurrSandy)
ax1.set_ylabel('Effluent concentration (ppm)', fontweight = 'bold') 
ax1.tick_params(axis = 'y', labelcolor='k', labelsize='medium', width=2)
    
ax1.yaxis.label.set_color('black')
ax1.invert_yaxis()
plt.legend(loc = 0)
plt.savefig('Flow Rate and Effluent pollutant.png',dpi = 1200)
  
#plot treatment performance in form of changes to the flow rate 

fig, ax = plt.subplots()

# ax.plot('time', 'nitrate_ppm', color = 'cornflowerblue',label ='Nitrate', data = hurrSandy)
ax.plot('time', 'nitrate_load', color = 'crimson',linestyle = 'solid',label ='Load', data = hurrSandy)
# ax.plot('time', 'nitrate_man3', color = 'cornflowerblue',linestyle ='dashed', label ='Manual Smooth',linewidth = 1.5,  data = hurrSandy)
ax.plot('time', 'smooth_nitrate_load', color = 'black',linestyle ='solid', label ='Smoothed Load',linewidth = 1.5,  data = hurrSandy)
plt.title('Smoothing Nitrate Load', fontweight = 'bold',fontsize = 12)
ax.set_ylabel('Nitrate in Effluent (Ibs/day)', fontweight = 'bold')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=None, useLocale=None, useMathText=True)
ax.tick_params(color = 'black', labelcolor='black',labelsize = 'large', width=2)
# ax.set_ylim(40, 110)
ax.yaxis.label.set_color('k')
ax.set_xlabel('Time (Days)', fontweight = 'bold',fontsize = 12)
plt.legend()
plt.savefig('Smoothed Nitrate Load.png',dpi = 1200)
