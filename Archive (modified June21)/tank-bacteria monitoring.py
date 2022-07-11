#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:32:12 2020

@author: kamausykes
"""


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir("/Users/kamausykes/Documents/GOALI Project/Modeling")

day = input("Enter desired day: ")

tank_no = input('Enter desired tank: ')

section = [1,2,3,4,5,6]

# For heterotrophs relative abundance

df = pd.read_excel('/Users/kamausykes/Documents/GOALI Project/Modeling/Simulations/MABR_Storm5(AutoRecovered).xlsx',sheet_name=1)

result = df["t"].isin([int(day)]) # produces a boolean dataframe that has true at the location of the given value

for col in df[['t']]:
    
    column = df['t']
    
    rows = list(result[result == True].index) # gets the row index of the true value in the boolean dataframe
    
my_list_df = list(df) # creates a list of column names

conc_array_het = np.empty(6,int)

for i in range(1,len(my_list_df)): # iterates through column names to find the right column
    
    col_name = my_list_df[i]
    
    if col_name[-4] == tank_no:
        
        for col in df[[col_name]]: # once the right column is found, it loops through the values to find the ones that correspond to the specified day
            
            column = df[col_name]
            
            conc = column[rows[0]]
            
            conc_array_het[int(col_name[-2])-1] = conc
            
print ("Heterotroph Relative Abundance: ", conc_array_het)

# For AOB relative abundance

df = pd.read_excel('/Users/kamausykes/Documents/GOALI Project/Modeling/Simulations/MABR_Storm5(AutoRecovered).xlsx',sheet_name=2)

result = df["t"].isin([int(day)]) # produces a boolean dataframe that has true at the location of the given value

for col in df[['t']]:
    
    column = df['t']
    
    rows = list(result[result == True].index) # gets the row index of the true value in the boolean dataframe
    
my_list_df = list(df)

conc_array_AOB = np.empty(6,int)

for i in range(1,len(my_list_df)): # iterates through column names to find the right column
    
    col_name = my_list_df[i]
    
    if col_name[-4] == tank_no:
        
        for col in df[[col_name]]: # once the right column is found, it loops through the values to find the ones that correspond to the specified day
            
            column = df[col_name]
            
            conc = column[rows[0]]
            
            conc_array_AOB[int(col_name[-2])-1] = conc
            
print ("AOB Relative Abundance: ", conc_array_AOB)

# For NOB relative abundance

df = pd.read_excel('/Users/kamausykes/Documents/GOALI Project/Modeling/Simulations/MABR_Storm5(AutoRecovered).xlsx',sheet_name=3)

result = df["t"].isin([int(day)]) # produces a boolean dataframe that has true at the location of the given value

for col in df[['t']]:
    
    column = df['t']
    
    rows = list(result[result == True].index) # gets the row index of the true value in the boolean dataframe
    
my_list_df = list(df)

conc_array_NOB = np.empty(6,int)

for i in range(1,len(my_list_df)): # iterates through column names to find the right column
    
    col_name = my_list_df[i]
    
    if col_name[-4] == tank_no:
        
        for col in df[[col_name]]: # once the right column is found, it loops through the values to find the ones that correspond to the specified day
            
            column = df[col_name]
            
            conc = column[rows[0]]
            
            conc_array_NOB[int(col_name[-2])-1] = conc
            
print ("NOB Relative Abundance: ", conc_array_NOB)


# plotting graphs

plt.plot(section, conc_array_het, linewidth=2, linestyle="solid", marker="o", markersize=4, label='Heterotroph Relative Abundance')

plt.plot(section, conc_array_AOB, linewidth=2, linestyle="solid", marker="D", markersize=4, label='AOB Relative Abundance')

plt.plot(section, conc_array_NOB, linewidth=2, linestyle="solid", marker="8", markersize=4, label='NOB Relative Abundance')

plt.title('Day '+ day + ': Tank '+ tank_no, )

plt.xlabel('Sections', fontsize = 20, weight='bold')

plt.ylabel('Conc. (mg COD/L)', fontsize = 20, weight='bold')

plt.grid(b= True, which = 'major', axis='y', linestyle=':', linewidth=1.5)

legend_properties = {'weight':'bold', 'size':15}

plt.legend(prop=legend_properties, bbox_to_anchor = (0.75, 1.15))

plt.xticks(fontsize=15, weight='bold')

plt.yticks(fontsize=15, weight='bold')

plt.show()