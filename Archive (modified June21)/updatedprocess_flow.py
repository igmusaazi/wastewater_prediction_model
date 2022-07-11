
"""
Created on Tue Sep 15 19:41:36 2020

@author: moria
"""


import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser
import re
import glob
import numpy as np
from datetime import date
import warnings
warnings.filterwarnings("ignore") # Supress all warnings

main = "D:/PhD research/Modelling/WWTP Flow Data_Python"
os.chdir("D:/PhD research/Modelling/WWTP Flow Data_Python/39 WWTP flow")
WWTP_ID = pd.read_excel('WWTP_ID_lookup.xlsx', index_col=None) # Load WWTP ID lookup spreadsheet
month = ["01","02","03","04","05","06","07","08","09","10","11","12"]

# Create an empty data frame to store data
total_days = date(2020, 3, 31) - date(2015, 1, 1) # Calculate number of days between 2015/1/1 and 2020/3/31
A = list(np.repeat(np.array(WWTP_ID.at[7,'File_ID']),2)) # Find the ID number for a specific treatment plant using index
B = np.array(["Timestamp","Value"]*len(A))
C = np.zeros(shape=(len(A),int((total_days.days+1)*24*60/10)))
df = pd.DataFrame(data=C.T, columns=pd.MultiIndex.from_tuples(zip(A,B))) # df is created to store compiled flow data

count = 0
for m in month: # Go through each month
    os.chdir(m)
    print("Processing month "+str(m)+"...")
    folder_name = os.listdir(os.getcwd()) # get a list of directory under current path
    for f in folder_name: # Go through each folder 
        os.chdir(f)      
        for file in glob.glob("*.xlsx"): # get all xlsx files and go through each one
            # Read raw data
            flow_data = pd.read_excel(file, index_col = None)
            WWTP_ID_match = re.split("_",file)[0] # Get all WWTP ID within the folder
            if WWTP_ID_match == (WWTP_ID.at[7,"File_ID"]): # Check if the extracted ID exist in the complete ID list     
                date_time = flow_data["Timestamp"].astype(str) # Extract Timestamp
                time_string = pd.DataFrame(index=flow_data.index, columns=['Timestamp']) # Convert to datatime object
                time_string = time_string.fillna(0)
                for i in range(0,len(date_time)):
                    time_string.iloc[i] = parser.parse(date_time[i])
                
                combined = pd.concat([time_string, pd.DataFrame(flow_data["Value"])], axis=1) # Combine time_string and flow value
      
                if count == 0:
                    df.loc[0:len(combined)-1,WWTP_ID_match]= combined.values
                else: # append combined 
                    df.loc[int(31*60*24/10*count):int(31*60*24/10*count+len(combined)-1),WWTP_ID_match] = combined.values
            else:
                continue # If the extracted ID is not in the complete list, skip this file and continue with the loop

        count = count + 1
        os.chdir('..')
        print("Completed folder "+str(f))
        
    os.chdir(main) # Go back to main folder
    print("Completed month "+str(m)+" ############################")

print("Completed data compilation!")
            
df.to_pickle("Compiled_flow.pkl") # Save compiled dataframe df as pickle file

#%reset # Delete all variables

########################################################################################################################
# open compiled dataframe df
os.chdir (main)
df = pd.read_pickle(r'Compiled_flow.pkl')
WWTP_permit = pd.read_excel('WWTP_permit.xlsx', index_col=None) 

#############################################################################
# Plot daily flow for Northeast plant
df_sort = df['IT0146FIT601A'] 
df_sort.loc[:,'Value'] = df_sort['Value'].replace(r'Bad','-999').astype(float) # Replace Bad with negative 999
df_sort = df_sort.loc[(df_sort!=0).any(axis=1)] # Remove all 0 Timestamp
df_sort.loc[~(df_sort['Value'] >= 0), 'Value'] = np.nan # Replace all negative values with NaN
df_sort = df_sort.sort_values(by='Timestamp',ascending=True) # Sort dataframe by Timestamp
df_sort.index = df_sort.Timestamp

df_davg = df_sort.resample('H').mean() # Aggregate from hourly to daily
df_davg['year'] = pd.DatetimeIndex(df_davg['Value'].index).year
df_davg['month'] = pd.DatetimeIndex(df_davg['Value'].index).month
df_davg['day'] = pd.DatetimeIndex(df_davg['Value'].index).day


rainfall = pd.read_excel ('/Users/kamausykes/Documents/GOALI Project/1610 Rainfall 2020-28-07_daily.xls')
rainfall.index = rainfall.Reading_Date_From
rainfall['year'] = pd.DatetimeIndex(rainfall['Rain'].index).year
rainfall['month'] = pd.DatetimeIndex(rainfall['Rain'].index).month
rainfall['day'] = pd.DatetimeIndex(rainfall['Rain'].index).day

#plt.plot(df_davg.loc[(df_davg['year'] == 2015) & (df_davg['month'] == 1) & (df_davg['day'] == 1)].Value, color='red', label = 'Daily interval', linewidth=2)
#plt.plot(df_davg.loc[(df_davg['year'] == 2015) & (df_davg['month'] == 1) & (df_davg['day'] == 2)].Value, color='blue', label = 'Daily interval', linewidth=2)
#plt.plot(df_davg.loc[(df_davg['year'] == 2015) & (df_davg['month'] == 1) & (df_davg['day'] == 3)].Value, color='green', label = 'Daily interval', linewidth=2)

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 25
fig_size[1] = 15
plt.rcParams["figure.figsize"] = fig_size
plt.rcParams.update({'font.size': 25})
#fig = plt.figure()
#ax1 = fig.add_subplot(211)
#ax1.plot(df_davg.loc[(df_davg['year'] == 2019) & (df_davg['month'] == 1)].Value, color='black', label = 'Daily interval', linewidth=2)
#ax2 = fig.add_subplot(212)
#ax2.plot(df_davg.loc[df_davg['year'] == 2015].Value, color='red', label = 'Daily interval', linewidth=2)
#ax2.plot(df_davg.loc[(df_davg['year'] == 2020) & (df_davg['month'] == 1)].Value, color='black', label = 'Daily interval', linewidth=2)
#ax2.plot(rainfall.loc[(rainfall['year'] == 2020) & (rainfall['month'] == 1)].Rain, color='blue', label = 'Daily interval', linewidth=2)
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df_davg.loc[(df_davg['year'] == 2020) & (df_davg['month'] == 1)].Value, 'black')
ax2.bar(rainfall.loc[(rainfall['year'] == 2020) & (rainfall['month'] == 1)].Rain.index,
        rainfall.loc[(rainfall['year'] == 2020) & (rainfall['month'] == 1)].Rain.values, color = 'b',width = 0.1)
ax2.invert_yaxis()

ax1.set_xlabel('Time')
ax1.set_ylabel('Flow', color='black')
ax2.set_ylabel('Rainfall', color='blue')

os.chdir("/Users/kamausykes/Documents/GOALI Project/Process Flow Results")
plt.savefig('Northeast_daily_flow_Jan.jpg')

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 25
fig_size[1] = 15
plt.rcParams["figure.figsize"] = fig_size
plt.rcParams.update({'font.size': 25})
#fig = plt.figure()
#ax1 = fig.add_subplot(211)
#ax1.plot(df_davg.loc[(df_davg['year'] == 2019) & (df_davg['month'] == 1)].Value, color='black', label = 'Daily interval', linewidth=2)
#ax2 = fig.add_subplot(212)
#ax2.plot(df_davg.loc[df_davg['year'] == 2015].Value, color='red', label = 'Daily interval', linewidth=2)
#ax2.plot(df_davg.loc[(df_davg['year'] == 2020) & (df_davg['month'] == 1)].Value, color='black', label = 'Daily interval', linewidth=2)
#ax2.plot(rainfall.loc[(rainfall['year'] == 2020) & (rainfall['month'] == 1)].Rain, color='blue', label = 'Daily interval', linewidth=2)
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df_davg.loc[(df_davg['year'] == 2020) & (df_davg['month'] == 2)].Value, 'black')
ax2.bar(rainfall.loc[(rainfall['year'] == 2020) & (rainfall['month'] == 2)].Rain.index,
        rainfall.loc[(rainfall['year'] == 2020) & (rainfall['month'] == 2)].Rain.values, color = 'b',width = 0.1)
ax2.invert_yaxis()

ax1.set_xlabel('Time')
ax1.set_ylabel('Flow', color='black')
ax2.set_ylabel('Rainfall', color='blue')

os.chdir("/Users/kamausykes/Documents/GOALI Project/Process Flow Results")
plt.savefig('Northeast_daily_flow_Feb.jpg')

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 25
fig_size[1] = 15
plt.rcParams["figure.figsize"] = fig_size
plt.rcParams.update({'font.size': 25})
#fig = plt.figure()
#ax1 = fig.add_subplot(211)
#ax1.plot(df_davg.loc[(df_davg['year'] == 2019) & (df_davg['month'] == 1)].Value, color='black', label = 'Daily interval', linewidth=2)
#ax2 = fig.add_subplot(212)
#ax2.plot(df_davg.loc[df_davg['year'] == 2015].Value, color='red', label = 'Daily interval', linewidth=2)
#ax2.plot(df_davg.loc[(df_davg['year'] == 2020) & (df_davg['month'] == 1)].Value, color='black', label = 'Daily interval', linewidth=2)
#ax2.plot(rainfall.loc[(rainfall['year'] == 2020) & (rainfall['month'] == 1)].Rain, color='blue', label = 'Daily interval', linewidth=2)
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df_davg.loc[(df_davg['year'] == 2020) & (df_davg['month'] == 3)].Value, 'black')
ax2.bar(rainfall.loc[(rainfall['year'] == 2020) & (rainfall['month'] == 3)].Rain.index,
        rainfall.loc[(rainfall['year'] == 2020) & (rainfall['month'] == 3)].Rain.values, color = 'b',width = 0.1)
ax2.invert_yaxis()

ax1.set_xlabel('Time')
ax1.set_ylabel('Flow', color='black')
ax2.set_ylabel('Rainfall', color='blue')

os.chdir("/Users/kamausykes/Documents/GOALI Project/Process Flow Results")
plt.savefig('Northeast_daily_flow_Mar.jpg')
