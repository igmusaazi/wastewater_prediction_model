#############################################
##Develop ML models that can predict influent flows to a plant based on precipitation, annual seasons,
water quality measurements
#############################################
Author: Isaac Musaazi 
Latest version:  Oct 20, 2023
############################################
import pandas as pd
import numpy as np
import scipy.stats
import smogn  ##synthetic minority over-sampling for  regression 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
import shap

data = pd.read_csv('your_data_file.csv')  # assume that data for developing a prediction model is from a CSV file
data['date'] = pd.to_datetime(data['date']) # time series data requires to convert the date column to datetime
data.set_index('date', inplace=True)
data = pd.get_dummies(data, prefix=None, drop_first=True) #drop one of the dummy variables. season of the year was the categorial variable

#Time based train-test split################
test_data = 290    #split the data to preserve the temporal order. Testing data comes from later time periods. For Plant I, 290 data points were considered in the testing set
train_data = data[:-test_data] #training data comes from the earlier time periods
test_data = data[-test_data:]

train_labels = train_data['flow'] # extract labels -  in this study 'flow'
test_labels = test_data['flow']


###some data preprocessing and handling censored values for training set###
train_data['censoring'] = train_data['variable'].apply(lambda x: 0 if '<' not in x and '>' not in x else (1 if '<' in x else 2))
train_data['variable'] = train_data['variable'].str.replace('[<>]', '', regex=True).astype(float) #variable was NOx for Plant I
### mean and standard deviation values obtained from R using the enorm censored package 
mean_value = 0.03
std_value = 0.01

censored_positions = (train_data.censoring == 1) # determine censored value positions
num_censored = censored_positions.sum() # calculate the number of censored values that are needed
random_values = np.random.normal(mean_value, std_value, num_censored) # generate those random values to replace the censored numbers. Assume these random values come from a normal distribution

###creating additional features from rainfall data to provide more information to the prediciton models and potentially improve predictive performance. 
#here the  count resets when the rainfall exceeds 0.04, and it increments for each day without significant rainfall
##This should result in a continuous progression of antecedent dry day (ADD)
#the "ADD" column is created in the training set. It counts the number of days before each rainfall event and resets the count when a rainfall event is observed. 

count_rain = 0
ADD = []

for rainfall in train_data['rainfall']:  # Iterate through the 'rainfall' column
    if rainfall <= 0.04:
        count_rain += 1  # Increment the count for each day without significant rainfall
    else:
        count_rain = 0  # Reset the count when rainfall exceeds 0.04
    ADD.append(count_rain)

train_data['ADD'] = ADD ##we have an additional variable in the training set


######handling missing data, particularly Plant II the imputation on train and test set is separate#####################
cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target', 'ADD', 'season_1', 'season_2', 'season_3'] # define columns to be used for imputation

imputation = IterativeImputer(estimator=KNeighborsRegressor(), max_iter=1000, tol=1e-1) # Create an IterativeImputer

for data in [train_data]:
    # Extract the columns from the data for imputation and perform imputation and update the train data
    X = data[cols] 
    imputed_values = imputation.fit_transform(X)
    imputed_df = pd.DataFrame(imputed_values, columns=X.columns) # Create a DataFrame with the imputed values
    for col in cols:
        data[col] = imputed_df[col]

for data in [test_data]:
    X = data[cols]
    imputed_values = imputation.fit_transform(X)
    imputed_df = pd.DataFrame(imputed_values, columns=X.columns)

    for col in cols:
        data[col] = imputed_df[col]    


####combine the train and test sets and perform some preliminary analysis#####
###here the correlation coefficients using spearman are completed########
###Table SX and Table SXX########
combined_data = pd.concat([train_data, test_data], axis=0)
spear_corr = combined_data.corr(method='spearman', min_periods=2)
correlation_matrix, p_values = scipy.stats.spearmanr(a=combined_data, b=None, axis=0)

coefficients_matrix = pd.DataFrame(correlation_matrix)
p_values = pd.DataFrame(p_values)

####################################
## streamline process for model selection and evaluation using pyCaret### 
##remember to install this package on your local machine#########
#################################
##check perfomrmance of different regression models based on training time and OAI. OAI is computed separately based on R2, RMSE, MAE (see Equation 1 in the paper)
model_selection = setup(data =combined_data , target = 'flow',preprocess=False,session_id = 123, normalize= True) #no preprocessing is required
best_models = compare_models()
model_results = pull()   ####ranks best to worst model based on the R2


###SMOGN algorithm is used to handle the skewed distribution of the target variable ('flow') 
##want to improve the ability of the modelsto predict rare cases effectively
num = 50 ###create 50 synthetic datasets and select the best set based on the lowest mse value and the lowest zero count for the dummy variables
best_mse = float('inf')  # initialize to a high value
synthetic_set = None
columns = ['seasonSpring', 'seasonSummer', 'seasonWinter']

for i in range(num):
    k_values = range(1, 3)  #k-values specifies the number of neighbors to consider for interpolation used in over-sampling
    pertubs =np.linspace(0.1, 1, 20)  #the amount of perturbation to apply to the introduction of Gaussian Noise.
    samp_methods = ['extreme', 'balance'] #less over/under sampling or more/over undersampling
    rel_thres_values = np.linspace(0.1, 1, 20)  # The higher the threshold (values close to 1), the higher the over/under-sampling boundary. 
    rel_xtrm_types = ['high', 'both'] #specifies region of the response variable y should be considered rare. When high oversampling is done 
    rel_coef_values = np.linspace(0.1, 5, 30)  # box plot coefficient used to automatically determine extreme and therefore rare "minority" values in y
    results = []

    k = randrange(1, 5)
    pertub =uniform(0.1, 0.4)
    samp_method = choice(['extreme', 'balance'])
    rel_thres = uniform(0, 1)
    rel_xtrm_type = choice(['high', 'both'])
    rel_coef = uniform(0.01, 0.4)
    synthetic_sets = [] 
    data_train = smogn.smoter(data=train_data, y='flow', k=k, samp_method=samp_method, rel_thres=rel_thres,
                              rel_xtrm_type=rel_xtrm_type,pert=pertub, rel_coef=rel_coef)
    
    data_train = data_train.dropna() ###some missing data may be produced; drop it to allow the synthetic generation to complete

    if not data_train.empty: # avoids having an empty data file after dropping some missing information   # Define the features (X) and target (y)
        X = data_train.drop(columns=['flow'])
        y = data_train['flow']
    
    
    # Split data into training and test sets
        XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=0.7, test_size=0.3, shuffle=False, stratify=None)


        yTestActual = yTest.values         # calculate MSE for the test set
        yTestSynthetic = data_train.loc[XTest.index, 'flow'].values
        if yTestActual.shape != yTestSynthetic.shape:
            min_length = min(len(yTestActual), len(yTestSynthetic))
            yTestActual = yTestActual[:min_length]
            yTestSynthetic = yTestSynthetic[:min_length]
            mse = np.mean((yTestActual - yTestSynthetic) ** 2)
            zero_count = data_train[columns].apply(lambda col: col.tolist().count(0)).sum()
             
            synthetic_sets.append((data_train.copy(), mse, zero_count))

if synthetic_sets:
    synthetic_sets.sort(key=lambda x: x[2]) # sort based on the number of zeros in the columns of interest (dummy variables - seasons)

    best_synthetic_data, best_mse, best_zero_count = synthetic_sets[0] # choose set with the lowest zero count as the best synthetic dataset
    
    def remove_zeros(data): # remove columns with zero counts
        columns = [col for col in data.columns if data[col].sum() == 0]
        data.drop(columns=columns, inplace=True)

    remove_zeros(best_synthetic_data)  


########model development using train and test data#####################
###use the train set without resampling and the train set with synthetic data separately####
##get the performance of the models for these different set by testing on the same test set###############
models = [
    ('Linear Regression', LinearRegression(), {}),
    
    ('k Nearest Neighbors',  KNeighborsRegressor(),
     {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']}),
    
    ('Random Forest', RandomForestRegressor(), {
        'n_estimators': [10, 20, 30, 40],
    'max_depth': [1, 2, 3, 4],
    }),
    
    ('BayesianRidge', BayesianRidge(), {
        'alpha_1': [1e-3, 1e-2,1e-1, 1, 10, 100],
    'alpha_2': [1e-3, 1e-2,1e-1, 1, 10, 100],
    'n_iter': [10,50, 100],
    'tol': [1e-3, 1e-2, 1e-1]
}),
    ('XGBoost', xgb.XGBRegressor(),
     {'n_estimators': [10, 20, 30, 50, 100],
    'max_depth': [1, 2, 3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
      })
]
results  = pd.DataFrame()
results_parameters = pd.DataFrame() ##different model hyperparameters

scaler = StandardScaler()  #use standard scaler for the data
X_train_scaled = scaler.fit_transform(train_data)
X_test_scaled = scaler.transform(test_data)

for model_name, model, param_grid in models:

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

    grid_search.fit(X_train_scaled, train_labels)
    best_params = grid_search_resampled.best_params_
    best_model = grid_search_resampled.best_estimator_
    predictions = best_model_resampled.predict(X_test_scaled)

    best_params_str = ', '.join([f"{key}: {val}" for key, val in best_params.items()])
    results_parameters[model_name + ' Best Parameters'] = [best_params_str]


    prediction_std = np.std(predictions)
    z_score = 1.96  # Corresponds to a 95% confidence interval
    prediction_interval = z_score * (prediction_std / np.sqrt(len(test_labels)))
    upper_bound = y_pred_resampled + prediction_interval
    lower_bound = y_pred_resampled - prediction_interval
    r2_resampled = r2_score(test_labels, y_pred_resampled)
    mse_resampled = mean_squared_error(test_labels, y_pred_resampled)

    results[model_name + ' Predictions'] = y_pred_resampled
    results[model_name + ' Upper Bound'] = upper_bound
    results[model_name + ' Lower Bound'] = lower_bound
    results[model_name, 'MSE'] = mse_resampled
    results[model_name, 'R2'] = r2_resampled


########figure SX################################
def cum_plot(data):
    '''plot the cumulative distribution function to determine the flow rate that constitutes
    a rare events. The 99th percentile is used as the cutoff'''
    
    counts, bins = np.histogram(combined_data['flow'], bins = 10)
    pdf = counts / sum(counts)  
    cdf = np.cumsum(pdf)
            
    plt.xlabel('Flow in MGD', size = 15)
    plt.ylabel('Cumulative Probability', size = 15)
    plt.title('Cumulative Probability and Flow') 
    plt.plot(bins[1:],pdf, color="red", label="PDF")
    plt.plot(bins[1:], cdf, label="CDF")
    plt.axhline(y = 0.99, color='k', linestyle='dashed',label = '99th percentile')
    plt.axvline(x=(np.percentile(combined_data['flow'],99)), color='r', linestyle='dashdot', label = 'Influent Flow Threshold')
    plt.legend()
    return plt.savefig('figure SXx.png', dpi = 1200)


#########Figure SX############################
model = 'XGBoost'  # Replace with your specific model
def y_formatter(y, pos):
  return abs(y)  # Custom y-axis label formatter to remove the negative sign

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10, 5))
ax1.plot('date', 'flow', label='measured flow', linestyle='solid', data=predictions_resampled, color='black')
ax1.plot('date', model, ls='solid', label=f'baseline flow', data=predictions, color='orange')
ax1.plot('date', model, ls='--', label=f'predicted flow (resampled)', data=predictions_resampled, color='green')
ax1.set_ylabel('influent flow (million gallons per day)')
ax1.set_xlim([date.date(2020, 2, 1), date.date(2020, 12, 31)])
date_form = mdates.DateFormatter('%m-%d')
ax1.xaxis.set_major_formatter(date_form)
week = mdates.WeekdayLocator(interval=3)
day = mdates.DayLocator()
ax1.xaxis.set_major_locator(week)
ax1.tick_params(axis='x', labelsize=10, rotation=45)
#ax1.legend(prop={"size": 8},loc='center left',bbox_to_anchor=(0.4, 0.5),frameon=False)
ax1.set_title('Plant I', fontsize=12)
ax2_1 = ax1.twinx()
ax2_1.bar(predictions['date'], -predictions['rainfall'], color='tab:blue', alpha=0.8, label='rainfall')
#ax2_1.legend(prop={"size": 8},loc='center left',bbox_to_anchor=(0.4, 0.4),frameon=False)
#ax2_1.set_ylabel('Rainfall')
ax2_1.yaxis.set_major_formatter(FuncFormatter(y_formatter))
rainfall_max = -max(predictions['rainfall'])
flow_max = max(predictions_resampled['flow'])
ax2_1.set_ylim(rainfall_max * 2, 0)
ax1.set_ylim(25, flow_max * 1.5)

ax2.plot('date', 'flow', label='measured flow', linestyle='solid', data=predictions_resampled_houston, color='black')
ax2.plot('date', model, ls='solid', label=f'baseline flow', data=predictions_houston, color='orange')
ax2.plot('date', model, ls='--', label=f'predicted flow (resampled)', data=predictions_resampled_houston, color='green')
#ax2.set_ylabel('influent flow (million gallons per day)')
ax2.set_xlim([date.date(2018, 9, 24), date.date(2019, 6, 30)])
date_form = mdates.DateFormatter('%m-%d')
ax2.xaxis.set_major_formatter(date_form)
week = mdates.WeekdayLocator(interval=3)
day = mdates.DayLocator()
ax2.xaxis.set_major_locator(week)
ax2.tick_params(axis='x', labelsize=10, rotation=45)
ax2.legend(prop={"size": 8},loc='center left',bbox_to_anchor=(0.2, 0.5),frameon=False)
ax2.set_title('Plant II', fontsize=12)

# Create a shared secondary y-axis for the bar plot
ax2_2= ax2.twinx()

# Bar plot on the secondary y-axis without the negative sign
ax2_2.bar(predictions_houston['date'], -predictions_houston['rainfall'], color='tab:blue', alpha=0.8, label='rainfall')
ax2_2.legend(prop={"size": 8},loc='center left',bbox_to_anchor=(0.2, 0.4),frameon=False)
ax2_2.set_ylabel('Rainfall')
ax2_2.yaxis.set_major_formatter(FuncFormatter(y_formatter))
rainfall_max = -max(predictions_houston['rainfall'])
flow_max = max(predictions_resampled_houston['flow'])
ax2_2.set_ylim(rainfall_max * 2.5, 0)
ax2.set_ylim(0.5, flow_max * 2)
plt.savefig('Fig XX.png',dpi=1200)



#########Figure 2 ###############################
model_colors = {
    'linearReg': 'orange',
    'kNearest': 'blue',
    'RandomForest': 'red',
    'BayesianRidge': 'green',
    'XGBoost': 'purple'
}

fig, ((ax_main1, ax_zoom1), (ax_main2, ax_zoom2)) = plt.subplots(2, 2, figsize=(18, 10), sharex='col')

r2_values_col1 = {model: [] for model in model_colors}
mse_values_col1 = {model: [] for model in model_colors}
r2_values_col2 = {model: [] for model in model_colors}
mse_values_col2 = {model: [] for model in model_colors}

def calculate_regression_stats(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value**2
    mse = np.mean((y - (slope * x + intercept))**2)
    return slope, intercept, r_squared, mse

x_min, x_max, y_min, y_max = 20, 100, 20, 100
col1 = ['linearReg', 'kNearest', 'RandomForest', 'BayesianRidge', 'XGBoost']
col2 = ['linearReg','kNearest', 'RandomForest', 'BayesianRidge', 'XGBoost']

# Loop through the columns of the predictions data for the first set of regressions (col1)
for col in col1:
    x = predictions.flow
    y = predictions[col]

    slope, intercept, r_squared, mse = calculate_regression_stats(x, y)
    r2_values_col1[col].append(r_squared)
    mse_values_col1[col].append(mse)
    color = model_colors[col]
    marker_size = 10

    # Plot the scatter plots and regression lines on the first main subplot
    ax_main1.scatter(x, y, label=f"{col}: R²={r_squared:.2f}", c=color, s=marker_size)
    #ax_main1.scatter(x, y, label=f"{col}: mse={mse:.2f}", c=color, s=marker_size)
    ax_main1.plot(x, slope * x + intercept, c=color)
    ax_main1.plot([0, 100], [0, 100], linestyle='--', color='black')
    ax_main1.set_ylabel("predicted flow (million gallons per day)")
    ax_main1.set_title("Plant I -baseline ")
    ax_main1.legend(frameon=False, fontsize=10)
    ax_main1.set_xlim(x_min, x_max)
    ax_main1.set_ylim(y_min, y_max)
    ax_zoom1.set_xlim(50, 80)
    ax_zoom1.set_ylim(50, 80)
    zoom_rect1 = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='0.3', linewidth=2)
    ax_main1.add_patch(zoom_rect1)
    ax_main1.indicate_inset_zoom(ax_zoom1)
    ax_zoom1.scatter(x, y,c=color)
    ax_zoom1.plot(x, slope * x + intercept,label=f"{col}", linestyle='-', linewidth=3,c=color)
    ax_zoom1.legend(frameon=False, fontsize=10)
    ax_zoom1.plot([50, 80], [50, 80], linestyle='--', color='black')
    ax_zoom1.text(74, 78, '1:1 Line', fontsize=12, color='black')

for col in col2:
    x = predictions_resampled.flow
    y = predictions_resampled[col]

    slope, intercept, r_squared, mse = calculate_regression_stats(x, y)

    r2_values_col2[col].append(r_squared)
    mse_values_col2[col].append(mse)
    color = model_colors[col]
    marker_size = 10

    # Plot the scatter plots and regression lines on the second main subplot
    ax_main2.scatter(x, y, label=f"{col}: R²={r_squared:.2f}", c=color, s=marker_size)
    #ax_main2.scatter(x, y, label=f"{col}: mse={mse:.2f}", c=color, s=marker_size)
    ax_main2.plot(x, slope * x + intercept, c=color)
    ax_main2.plot([0, 100], [0, 100], linestyle='--', color='black')
    # Customize the main plots for both sets of regressions (same as before)
    #ax_main1.set_xlabel("measured flow (million gallons per day)")


    ax_main2.set_xlabel("measured flow (million gallons per day)")
    ax_main2.set_ylabel("predicted flow (million gallons per day)")
    ax_main2.set_title("Plant I  - resampled")

    # Add legends to the main plots for both sets of regressions

    ax_main2.legend(frameon=False, fontsize=10)

# Set the zoomed-in portions for both main plots (same as before)
    ax_main2.set_xlim(x_min, x_max)
    ax_main2.set_ylim(y_min, y_max)


    # Customize the zoomed-in subplots for both sets of regressions (same as before)
    #ax_zoom1.set_xlabel("measured flow")
    #ax_zoom1.set_ylabel("predicted flow")
    #ax_zoom1.set_title("Zoom In - Set 1")

    ax_zoom2.set_xlabel("measured flow (million gallons per day)")
    #ax_zoom2.set_ylabel("predicted flow")
    #ax_zoom2.set_title("Zoom In - Set 2")
    ax_zoom2.set_xlim(50, 80)
    ax_zoom2.set_ylim(50, 80)

    # Add rectangles to indicate the zoomed-in areas for both sets of regressions (same as before)
    zoom_rect2 = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='0.3', linewidth=2)
    ax_main2.add_patch(zoom_rect2)
    ax_main2.indicate_inset_zoom(ax_zoom2)
    # Plot the 1:1 lines on both zoomed-in subplots (same as before)
    ax_zoom2.scatter(x, y,c=color)
    ax_zoom2.plot(x, slope * x + intercept, label=f"{col}",linestyle='-', linewidth=3,c=color)
    ax_zoom2.plot([50, 80], [50, 80], linestyle='--', color='black', label='1:1 Line')
#plt.show()
plt.savefig('FigX.png',bbox_inches='tight',dpi=2000)

#########these model hyperparameters are obtained from the grid search. These should be different for the baseline and resampled training data#########
#best_knn_model = KNeighborsRegressor(metric=manhattan, n_neighbors=10, weights=distance)
best_random_forest_model = RandomForestRegressor(max_depth=4, n_estimators=10)
best_bayesian_ridge_model = BayesianRidge(alpha_1=100, alpha_2=0.001, n_iter= 10, tol= 0.001)
best_xgboost_model = xgb.XGBRegressor(learning_rate=0.2, max_depth=4, n_estimators=100)


#################Figure XX #########################3
models = [
    ('Random Forest', best_random_forest_model),
    ('Bayesian Ridge', best_bayesian_ridge_model),
    ('XGBoost', best_xgboost_model)
]
shap_values_list = []
for model_name, model in models:
    model.fit(X_train_scaled, train_labelsBal)

    explainer = shap.Explainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    shap_values_list.append(shap_values)
    feature_names = train_alexRenew.columns.tolist()
for idx, (model_name, shap_values) in enumerate(zip([model[0] for model in models], shap_values_list)):
    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, title=f'{model_name} Shapley Summary Plot')
    # Save the plot with a unique name (optional)
    plt.savefig(f'shap_summary_plot_{idx}.png', bbox_inches='tight',dpi=2000)
    plt.close('all')
