# -*- coding: utf-8 -*-
"""
Created on Tue May  4 19:58:29 2021

@author: musaa
"""

import os
import pandas as pd 
import numpy as np
import seaborn as sns #for enhanced matplotlib graphics
import matplotlib.pyplot as plt
import scipy.integrate as integrate 

# %whos print variables in memory

os. chdir('D:/PhD research/Modelling/Python Modelling/training')
##general guide followed when dataset is loaded
df = pd.read_csv('../training/gapminder.tsv',sep ='\t') #read file from current directory
df.shape  #number of rows and columns
df.info() #more information about the dataset i.e, column names, data types for data in each row
df.head() # first five rows of the dataset
df.columns #shows the columns in the dataset
df.index #indcates the row numbers
df.values # provides numpy array, using pandas if talking to other libraries
country = df['country'] #extension of an numpy array
country = df[['country']] # need dataframe as an output, better print out 
drop_col = df.drop(['country', 'continent'], axis = 'columns') #removes columns
subset_rows = df.loc[[4, 5]] #string match and find output specific rows
subset_row = df.iloc[[-4,5]] #output specific rows based on position not the string identity
subset = df.loc[:, ['year','pop']] #select all rows with specific columns
subset = df.iloc[:, [2,4]] #select all rows with specific columns
spec_country = df.loc[df['country']=='United States']
life_mean = df.groupby(['year'])['lifeExp'].mean() #based on the pandas library
life_mean = df.groupby(['year'])['lifeExp'].agg(np.mean) #using another library
life_gdp_mean = df.groupby(['year','continent'])[['lifeExp', 'gdpPercap']].agg(np.mean)
life_gdp_mean = df.groupby(['year','continent'])[['lifeExp', 'gdpPercap']].agg(np.mean).reset_index() #flat dataset

#exercise 
tips = tips = sns.load_dataset('tips')
bill_ave_total = tips.groupby(['smoker','day', 'time'])['total_bill'].mean()
sm_bill = tips.loc[(tips['smoker']=='No')&(tips['total_bill']>=10)] #filter smoker no and total bill greater than 10


pew = pd.read_csv('../training/pew.csv')
pew_tidy = pew.melt(id_vars='religion',var_name = 'income', value_name = 'count')

billboard = pd.read_csv('../training/billboard.csv')
artist_mean = billboard\
    .melt(id_vars=['year','artist', 'track', 'time', 'date.entered']\
                             ,var_name='week', value_name='rank').groupby\
        ('artist')['rank'].mean() #reorganizes dataset and finds rank means based on artist
time = time.astype(np.float)
ebola = pd.read_csv('../training/country_timeseries.csv')
ebola_long = ebola.melt(id_vars=['Date', 'Day'],value_vars='cd_country',value_name='count')
ebola_split = ebola_long['cd_country'].str.split('_', expand = True) #two words joined by _ separated to individual columns
col_new = ebola_long[['status', 'country']] = ebola_split  

weather = pd.read_csv('../training/weather.csv')
weather_long = weather.melt(id_vars=['id','year','month','element'], var_name='day', value_name='temp')
weather_long\
    .pivot_table(index=['id','year', 'month']\
                 ,columns='element',values='temp') #temp from rows to columns
pf = pd.DataFrame(\
                  {'a': [1,1,3,5]\
                   ,'b':[1,1,3,5]}) #create dataframe
def my_sq(x):
    return x**2
pf['a'].apply(my_sq) 

#############simple if statements##########
a = 24  ##assign the number

if a%2 == 0: ####set the if condition
    print("{} is an even number".format(a)) ###determine the output
else:
    print("{} is odd".format(a))
    
boys = ("Kevin", "Jacob", "Levi")
candidate = ("Levi")

if candidate in boys:
    print("{} is a boy".format(candidate))
else:
    print("{} is not a boy".format(candidate))
    
a = 24 
if a%2==0:  ########nested if statements########
    print("{} is  divisible by 2".format(a)) 
    if a%3 ==0:
        print("{} is also divisible by 3".format(a))
else:
    print("{} is not divisible by 3".format(a))
    
########another loop to test#########
i = 1
result = 1

while i <= 100:
    result *= i
    if i == 42:
        print("Magic number 42 reached")
        break
    i +=1
print('i:',i)
print('result:', result)  


#####code prints out Banana is not good while orange and apple as my fruit ###########
for fruit in ['Banana', 'Apple', 'Orange']:
    if fruit == "Banana":
        print("This fruit is not good:", fruit)
    else:
        print("Here is my fruit:",fruit) 

days = ['Mon', 'Tue', 'Wed', 'Thurs', 'Fri']
for i in range (len(days)):
 print("The value at position {} is {}.".format(i,days[i]))
 
###  functions ################ 
def filter_even(number_list):
    result_list = []
    for number in number_list:
        if number%2 == 0:
            result_list.append(number)
    return(result_list)
even_list = filter_even([3,5,7,9,13])   

###loan payment options ### try and except condition works when the calculation at some point returns a zero in the denominator
def emi(amount,duration, rate, downpayment=0):
    """ calculates the monthly loan repayment based on interest rate
    loan duration and down payment""" ###docustring used to describe the function
    
    loan_amount = amount - downpayment
    try:
        emi = loan_amount*rate*((1+rate)**duration)/(((1+rate)**duration)-1)
    except ZeroDivisionError:
        emi = loan_amount/duration
    return emi

emi1 = emi(1_260_000, 8*12, 0.1/12, 300_000)   
emi2 = emi(1_260_000, 10*12, 0.08/12) 

# storm['bod11'] = pd.to_numeric(storm.bod11) - convert dtype(object) to numeric
##########plotting###########
tips = sns.load_dataset('tips')
tips.head()
tips.tip.plot(kind='hist')
cts = tips.smoker.value_counts()
cts.plot(kind= 'bar')
sns.countplot(x='smoker', data= tips)
sns.displot(tips.total_bill) #distribution plot
sns.lmplot(x='total_bill',y='tip',data = tips)

tit = pd.read_csv('../Python Modelling')
tit.head()
fig, ax = plt.subplots()  ##call figure to display
_ = ax.plot(np.sort(tit['age']), marker ='o',markersize =1) ###np.sort ignores blanks in dataset
_ =ax.set_title("Titanic") #_ means results from assignment doesn't matter
_ = ax.set_ylabel("Ages")
_ =ax.set_yticks([15, 25,55,64]
  
fig, (ax1,ax2) = plt.subplots(ncols=2)   ###two figures
fig.suptitle('horray') #####title in the middle of the 2 plots
fig, axes = plt.subplots(ncols=2, nrows=2, constrained_layout = True) ### 4 figures, layout does the magic  
axes[0,0].plot() ###specifics the location of the plot
##########writing final dataset to a file ############
storm_final = storm5[['t','bod11','tkn11','totbod','totkn']] ###selects information that needs to be captured
storm5.to_csv("stormfinal.csv",index=None) ##writing .csv file minus without including the index

storm_final.bod11.plot() ##simple line plot
storm_final.set_index('t',inplace = True) ###change index for the x-axis in the plot function for pandas

storm5.bod11.plot(); storm5.tkn11.plot() ##bod and tkn plots in pandas
plt.plot(storm5.bod11);plt.plot(storm5.tkn11); plt.xlabel('Time in days'); plt.ylabel('BOD and TKN');plt.legend(['bod','tkn']) ##matplotlib library
output_plot = plt.plot(storm5.bod11, marker = 'o',ms = 1);plt.plot(storm5.tkn11, marker = '*', ms = 2); plt.xlabel('Time in days'); plt.ylabel('BOD and TKN');plt.legend(['bod','tkn']);plt.title('Change in pollutant over time')

plt.plot(storm5.bod11,'s-b');plt.plot(storm5.tkn11, '*--y');\
    plt.xlabel('Time in days'); plt.ylabel('BOD and TKN');\
        plt.legend(['bod','tkn']);\
            plt.title('Change in pollutant over time') ##fmt style helps specify the marker, line and color in the plot
            
sns.set_style('darkgrid')
sns.scatterplot(x = storm.t,y = storm.bod11, s = 100)
plt.title('BOD changes over time');sns.scatterplot(x ='t', y = 'bod11', data= storm)

plt.title('Distribution of TKN'); plt.hist(storm.tkn11, bins =np.arange(0,4, 2)) ##specifies the bins that need to be used

sns.barplot(x ='day', y = 'total_bill', data =tips_df)###computes the average and can be used to avoid using the plt.plt function that requires having to first calculate the average values that will be passed to the plot function
sns.barplot(x ='day', y = 'total_bill',hue = 'sex', data =tips_df); plt.xticks(rotation = 80) ###categorize based on sex in the data

flight_df.pivot('year','month','passengers') #pivot converts the data into a matrix here year(row) and month(column)

plt.title('No of passengers in 000s');sns.heatmap(flight2,cmap= 'Blues') ###heatmap use "annot" to print numbers 

from PIL from Image ###module helps in reading images
pics = Image.open('Photo ISAAC MUSAAZI.jpeg')

def multiply(*numbers): #variable number of arguments to a function
    total = 1
    for number in numbers:
        total *=number
    return total

multiply(2,3,4,5,6,7,8,9)

def save_user(**user):
    print(user['name'])
save_user(id = 1,name = "John", age =22)

def fizz_buzz(input):
    if (input%3 == 0) and (input%5 == 0):
        return "FizzBuzz"
    if input%3 == 0:
        return "Fizz"
    if input%5 == 0:
        return "Buzz"
    return input

np.c_[np.array([1,2,3]),np.array([0]*3)] ###concantenate columns

def vect_add(a,b):  ##function that can add arrays
    c=[None]*len(a)
    for k in range(0,len(a)):
        c[k] = a[k]+b[k]
    return c

def vect_add(a,b): ####add arrays using the append function
    c=[]
    for k in range(0,len(a)):
        c.append(a[k]+b[k])
    return c

##########iterations and loops#########

user_1 = {'username':'Mitchel','id':2,'educlevel':'secondary'}

user_2 = {'username':'Fred','id':1,'educlevel':'primary', 'email':"let@yahoo.com"}

users = [user_1, user_2]

for user in users: ###print user id
    print(user['id'])
    
for user in users: ###print user with email
    if 'email' in user:
        print(user['email'])

select_user = {}

user_look_up = 1

for user in users:
    if 'id' in user:
        if user['id'] ==user_look_up:
            select_user = user
            
abc = [1,2,3,4,5,6]

abc_sq = []

for num in abc: ####loop squares abc_sq
    new_number = num**2
    abc_sq.append(new_number)