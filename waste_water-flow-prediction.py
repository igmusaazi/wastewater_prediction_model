"""
This script is used to develop a random forest model to predict influent flows based on precipitation, annual seasons, time of the day,
effluent water quality measurements. 
...

Author: 
    Isaac Musaazi 
Latest version: 
    March 20, 2022 @ 2p.m
"""
import pandas as pd
import numpy as np
import scipy.stats
import smogn  ##synthetic minority over-sampling for  regression 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.tree import export_graphviz
import pydot


data = pd.read_csv('trainAlex.csv')

# Define the SMOGN parameters
num_synthetic_datasets = 100 ###we create 50 synthetic datasets for plant I and select the best set based on the lowest mse value
best_mse = float('inf')  # Initialize with a high value
best_synthetic_data = None

for i in range(num_synthetic_datasets):
    # Define SMOGN parameters
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

    # Apply SMOGN
    data_train = smogn.smoter(data=data, y='flow', k=k, samp_method=samp_method, rel_thres=rel_thres,
                              rel_xtrm_type=rel_xtrm_type,pert=pertub, rel_coef=rel_coef)
    
    data_train = data_train.dropna() ###some missing data may be produced; drop it to allow for further analysis

    if not data_train.empty: # avoid having an empty data file after drop some missing information         # Define the features (X) and target (y)
        X = data_train.drop(columns=['flow'])
        y = data_train['flow']
    
    
    # Split data into training and test sets
        XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=0.7, test_size=0.3, shuffle=False, stratify=None)

        # Calculate MSE for the test set
        yTestActual = yTest.values
        yTestSynthetic = data_train.loc[XTest.index, 'flow'].values
        if yTestActual.shape != yTestSynthetic.shape:
            min_length = min(len(yTestActual), len(yTestSynthetic))
            yTestActual = yTestActual[:min_length]
            yTestSynthetic = yTestSynthetic[:min_length]
            mse = np.mean((yTestActual - yTestSynthetic) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_synthetic_data = data_train.copy()  
best_synthetic_data.to_csv("best_synthetic_data.csv", index=False)  # save the best synthetic dataset to a CSV file

####preliminary correlation between variable to be used in the model - spearman's correlation################
dcBlue = pd.read_csv('../DC Water Data October 2021/dcWaterStorms(October 2021)/BluePlainsFinal_Update(March22).csv') ##effluent quality and precipitation
dcBlue['datetime'] = pd.to_datetime(dcBlue['datetime'])

dcBlue = pd.get_dummies(dcBlue, prefix=None)
dcBlueCorr = dcBlue[['mean_prcp', 'influent_flow', 
'ammonia_ppm', 'nitrate_ppm', 'nitrite_ppm', 'season_Fall', 'season_Spring', 'season_Summer', 'season_Winter','timeDay_Afternoon', 'timeDay_Evening', 'timeDay_Morning', 'timeDay_Night']] 
spear = dcBlueCorr.corr(method='spearman', min_periods =2) #min periods is the minimum pair of observations for a valid result
dcBlueCorrMtrx = scipy.stats.spearmanr(a=dcBlueCorr,b=None, axis=0)
coefMatrx = pd.DataFrame(dcBlueCorrMtrx[0])
pvalues = pd.DataFrame(dcBlueCorrMtrx[1])
#coefMatrx.to_csv('dcBluespearman(March2022).csv')
# =============================================================================
########cross correlation###########
dcBlue = dcBlue.drop(['Unnamed: 0','Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11'], axis =1)

fields = ['datetime','influent_flow_imp','mean_prcp','ammonia_ppm_imp', 'nitrate_ppm_imp', 'nitrite_ppm_imp']
x =dcBlue[fields]
def df_derived_by_shift(df,lag=0,NON_DER=[]):
    df = df.copy()
    if not lag:
        return df
    cols ={}
    for i in range(1,lag+1):
        for x in list(df.columns):
            if x not in NON_DER:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k,v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i+=1
        df = pd.concat([df, dfn], axis=1).reindex(df.index)
    return df

NON_DER = ['datetime',]

df_new = df_derived_by_shift(x, 6, NON_DER)
df_new = df_new.dropna()
crosstest = df_new.corr()
laggedvalues = ['datetime','influent_flow_imp','mean_prcp_3','ammonia_ppm_imp_3', 'nitrate_ppm_imp_3', 'nitrite_ppm_imp_3']
dcBlue2=df_new[laggedvalues]
dcNew = pd.merge(left =dcBlue,right = dcBlue2, left_on = 'datetime', right_on ='datetime') #column merge based on 'DateRead'
d2New = dcNew.loc[:, ['datetime', 'influent_flow_imp_x', 'mean_prcp_3','ammonia_ppm_imp_3', 'nitrate_ppm_imp_3', 'nitrite_ppm_imp_3','event', 'season']]

##define duplicates and plot ############
dcBlue2= dcBlue.groupby(['ammonia_ppm_imp', 'nitrate_ppm_imp', 'nitrite_ppm_imp','Time']).size().reset_index(name='count')


### random forest regressor ###############
dcBlue = pd.read_csv('../DC Water Data October 2021/dcWaterStorms(October 2021)/BluePlainsFinal_Lag(March22).csv') ##effluent quality and precipitation
dcBlue['datetime'] = pd.to_datetime(dcBlue['datetime'])
dcBlue = pd.get_dummies(dcBlue, prefix=None)

####train test split not random accuracy testdata = 10% ################
test_data = 1389
train_dcBlue = dcBlue[:-test_data]
valid_dcBlue=train_dcBlue.iloc[-500:]
train_dcBlue= pd.concat([train_dcBlue, valid_dcBlue]).drop_duplicates(keep=False).copy()
test_dcBlue=dcBlue[-test_data:]

train_labels = train_dcBlue['influent_flow']
valid_labels = valid_dcBlue['influent_flow']
test_labels = test_dcBlue['influent_flow']

valid_dcBlue = valid_dcBlue.drop(['Unnamed: 0','datetime','influent_flow'], axis =1)


#######suitable set of hyperparametersusing grid search balanced#########
params = {
    'n_estimators': [1000, 2000, 3000, 4000, 5000],
    'min_samples_split': [100, 200, 500, 1000, 2000],
    'max_features': [2, 3, 4, 5, 6, 8]
}

rf_gridsearch = GridSearchCV(
    estimator = RandomForestRegressor(random_state=42),
    param_grid=params,
    cv=5,
    n_jobs = -1,
    scoring='neg_mean_absolute_error',
    verbose=1,
    error_score='raise'
)

rf_gridsearch.fit(valid_dcBlue, valid_labels)
rf_gridsearch.best_estimator_
rf_gridsearch.best_score_

###training dataset not balanced ##########
rf = RandomForestRegressor(max_features=8, min_samples_split=100, n_estimators=2000, random_state=42).fit(train_dcBlue, train_labels) ##not balanced

train_predictions = rf.predict(train_dcBlue)
train_errors = abs(train_predictions - train_labels)
train_mape = 100 * (train_errors / train_labels)   #performance metrics
train_mae = np.mean(train_errors) #mean absolute error
train_accuracy = 100 - np.mean(train_mape)

test_predictions = rf.predict(test_dcBlue)
test_errors = abs(test_predictions - test_labels)
test_mape = 100 * (test_errors / test_labels)   #performance metrics
test_mae = np.mean(test_errors) #mean absolute error
test_accuracy = 100 - np.mean(test_mape)

#####balancing dataset SMOGN ##########
train_dcBlue = train_dcBlue.drop(['Unnamed: 0','datetime'], axis =1)

influent_smogn = smogn.smoter(data = train_dcBlue,  y = 'influent_flow',\
                      k=7,samp_method = 'balance',rel_thres =0.8,rel_method='auto',rel_xtrm_type = 'both', rel_coef = 1.5).copy() 

trainBal = pd.read_csv('../DC Water Data October 2021/TrainingBalanced(March22).csv')
trainBal_labels = trainBal['influent_flow']
trainBal = trainBal.drop(['Unnamed: 0','influent_flow'], axis =1)

###training dataset balanced ##########
rfBal = RandomForestRegressor(max_features=8, min_samples_split=1000, n_estimators=2000, random_state=42).fit(trainBal, trainBal_labels) ##balanced


trainBal_predictions = rfBal.predict(trainBal)
trainBal_errors = abs(trainBal_predictions - trainBal_labels)
trainBal_mape = 100 * (trainBal_errors / trainBal_labels)   #performance metrics
trainBal_mae = np.mean(trainBal_errors) #mean absolute error
trainBal_accuracy = 100 - np.mean(trainBal_mape)

test_dcBlue = test_dcBlue.drop(['Unnamed: 0','datetime','influent_flow'], axis =1)
test_predictions = rfBal.predict(test_dcBlue)
test_errors = abs(test_predictions - test_labels)
test_mape = 100 * (test_errors / test_labels)   #performance metrics
test_mae = np.mean(test_errors) #mean absolute error
test_accuracy = 100 - np.mean(test_mape)


##################prediction interval (80%)############# https://andrewpwheeler.com/2022/02/04/prediction-intervals-for-random-forests/
test_pred = pd.DataFrame(test_predictions, columns = ['influent_pred']) #convert array to dataframe
resid = trainBal_labels - rfBal.oob_prediction_
lowq = resid.quantile(0.1)
higq = resid.quantile(0.9)


####contribution of input variables on the prediction of influent. caution: use small sample size##########
test_dcBlue4 = test_dcBlue.rename(columns = {'mean_prcp':'rainfall','ammonia_ppm':'effluent_ammonia','nitrate_ppm':'effluent_nitrate', 'nitrite_ppm':'effluent_nitrite','timeDay_Morning':'morning_time'
                                            ,'timeDay_Afternoon':'afternoon_time', 'timeDay_Evening':'evening_time','timeDay_Night':'night_time'})

explainer = shap.TreeExplainer(rfBal, feature_perturbation='tree_path_dependent')
shap_values = explainer.shap_values(test_dcBlue4)
shap_obj = explainer(test_dcBlue4)
shap.plots.bar(shap_obj, show= 'True')
plt.savefig('beeswarmplotNoBal.png', dpi =1200)

rf = RandomForestRegressor(max_features=8, min_samples_split=100, n_estimators=2000, max_depth = 3, random_state=42).fit(train_dcBlue, train_labels) ##not balanced
tree_small = rf.estimators_[5]

dcBluefeatures_list = list(train_dcBlue.columns) ##save features names
export_graphviz(tree_small, out_file = 'small_tree_quality.dot', feature_names = dcBluefeatures_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree_quality.dot')
graph.write_png('small_tree_quality.png')

importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(dcBluefeatures_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# =============================================================================
# baseline_preds = test_dcBlue[:, dcBluefeatures_list.index('average')]  ##baseline to check quality of predictions
# baseline_errors = abs(baseline_preds - test_labels)
#  regressor = RandomForestRegressor(n_estimators=20, random_state=0)
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# 
# =============================================================================
######separate out night and wet#########
dcNight = dcBlue[(dcBlue['influent_flow'] >= 0) & (dcBlue['timeDay'] == 'Night')]
dcWet = dcBlue[(dcBlue['influent_flow'] >= 0) & (dcBlue['event_Wet'] == 1)]
