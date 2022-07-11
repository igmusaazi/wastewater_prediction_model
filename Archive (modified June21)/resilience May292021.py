"""
This script calculates the time equivalent performance reduction, maximum performance
reduction and recovery time for a given storm event defined by a rising limb, crest, and
recession limb. 

Author: 
    Isaac Musaazi
Created:
    May 10, 2021
    
Latest update: 
    May 29, 2021
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



os.getcwd()

# os.chdir('D:\Howard\Modelling\Python')

storm = pd.read_csv('StormDataAnalysis_Isaac.csv')
storm = storm.drop([0]).reset_index(drop=True) #remove first row with parameter units
storm = storm.loc[:,['t','srtSRTtotal','q11','vss11','scod11','cod11','bod11','snh11','tkn11']]
# storm = storm[['t','srtSRTtotal','q11','vss11','scod11','cod11','bod11','snh11','tkn11']]
storm.describe()
storm= storm.astype(float)

plt.scatter(storm.t, storm.bod11, color="blue", label = 'BOD', alpha = 1, s = 2, marker = 'v')
plt.scatter(storm.t, storm.cod11, color="red", label = 'COD',alpha= 1, s= 2, marker = '*')
plt.scatter(storm.t, storm.tkn11, color="black", s = 2, alpha = 1, label = 'TKN')
plt.scatter(storm.t, storm.vss11, color="yellow", label = 'VSS',alpha= 1, s= 2, marker = '8')
plt.xlabel('Time in days', size = 15)
plt.ylabel('concentration (mg/L)', size = 15)
plt.title('Pollutograph', fontweight ='bold')
plt.legend()
plt.savefig('Pollutograph.png',dpi = 1200)
    

# def pollutant_baseline(pollutant):
#     baseline = np.polyfit(storm.t, pollutant, 0) ##fit data to a zero degree polynomial assuming that pollutant concentration doesnot change over time
#     baseValue = np.poly1d(baseline)[0]
#     print(baseValue)
    
# base = pollutant_baseline(storm.bod11) #specify the type of pollutant and obtain the baseline pollutant value value 

baseline = np.polyfit(storm.t, storm.bod11, 0) ##fit data to a zero degree polynomial assuming that pollutant concentration doesnot change over time
baseValue = np.poly1d(baseline)[0]

lower_limit = []
upper_limit = []
time = np.array(storm.t)
def recovery_time0(conc):
    for i in range(0, 36): 
         if conc[i] <= baseValue and conc[i+1] >= baseValue and conc[i+2] >= conc[i+1]:
            lower_limit = time[i]
            print(f"Lower Limit: {lower_limit}")
            
    for i in range(36, len(conc)): 
        if conc[i] <= baseValue and conc[i-1] >= baseValue and (conc[i-2] > conc[i-1] or conc[i - 2] == conc[i-1]): 
            upper_limit = time[i]
            print(f"Upper Limit: {upper_limit}")
            RT = upper_limit-lower_limit
            f = open('test.csv','w')
            f.write(str(lower_limit) +'\n')
            f.write(str(upper_limit) + '\n')            
            f.write(str(upper_limit) + '\n')
            f.close()
            return RT
            
recovery_time0(storm.bod11)
 
def recovery_time1(conc):
    time = np.array(storm.t)
    lower_limit = 0
    upper_limit = 0
    for i in range(20, 36): 
        if conc[i-1] <= conc[i] and conc[i+1] <= conc[i] and conc[i+2] <= conc[i+1]:
            lower_limit = time[i]
            print(f"Lower Limit: {lower_limit}")

    for i in range(36, len(conc)):
        if conc[i] <= baseValue and conc[i-1] >= baseValue and (conc[i-2] > conc[i-1] or conc[i - 2] == conc[i-1]):
            upper_limit = time[i]
            print(f"Upper Limit: {upper_limit}")
            RT = upper_limit-lower_limit
            return RT
recovery_time1(storm.bod11)

def tepr(bod11, t):
    I = np.trapz(bod11, t) - max(storm.t)*storm.bod11[1]
    return I
I = tepr(storm.bod11,storm.t)

print("The Time Equivalent Performance Reduction is {}".format(I))
    
def timeline(time, pollutant):
    lower_limit = 0
    for i in range(0, len(pollutant)):
        if pollutant[i] <= 1.9 and pollutant[i+1] >= 1.9:          
            lower_limit = pollutant[i]
            print(lower_limit)
            
timeline(storm.t, storm.bod11)
        
# storm['bod_removal'] = (storm.bod11 - abs(storm.bod11 - fit_coeff[0]))/storm.bod11 ####pollutant removal performance 
# sns.lineplot(x=storm.t,y=storm.bod_removal).set(title='pollutant removal', xlabel='time in days',ylabel ='pollutant removal');sns.set_theme(style="darkgrid") ###lineplot of pollutant removal over time


for i in storm:
    if storm[i]>=storm[i+1] and storm[i+1]>=storm[i+2] and storm[i+2] == storm[i+3]:
        upper_limit = storm[i+2]
        print(upper_limit)


###time equivalent performance reduction#########

#####maximum performance reduction ##############
data_fit = x 
curve = np.polyfit(storm.t,storm.bod11,2) ##fit data to a second degree polynomial
poly= np.poly1d(curve) ###equation that describes change 0.0004099 x**2 - 0.05048 x + 3.142
bod_predict = np.polyval(data_fit,storm.t[:150])

mpr = storm.bod11.max() - np.mean(poly(storm.t))
print("The Maximum Performance Reduction is {}".format(mpr))

