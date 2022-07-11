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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics 
from sklearn.tree import export_graphviz
import pydot
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier

### random forest regressor ###############
#dcBlue = pd.read_csv('../dcWaterStorms(October 2021)/dcStormsFinal(November2021)/april2018.csv')
dcBlue = pd.read_csv('../DC Water Data October 2021/dcWaterStorms(October 2021)/BluePlainsFinal(October2021)_Update(Nov23).csv')
#dcBlue = pd.read_csv('../DC Water Data October 2021/dcWaterStorms(October 2021)/BluePlainsFinal(October2021)_Update(Dec14).csv') ##effluent quality and precipitation
dcBlue['datetime'] = pd.to_datetime(dcBlue['datetime'])
dcBlue = pd.get_dummies(dcBlue, prefix='season')
labels = np.array(dcBlue['influent_flow_imp'])
dcBlue = dcBlue.drop(['Unnamed: 0','datetime', 'influent_flow_imp'], axis =1)
dcBluefeatures_list = list(dcBlue.columns) ##save features names
dcBlue = np.array(dcBlue)

train_dcBlue, test_dcBlue, train_labels, test_labels = train_test_split(dcBlue, labels, test_size = 0.25, random_state = 42)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_dcBlue, train_labels)
predictions = rf.predict(test_dcBlue)
errors = abs(predictions - test_labels)
mape = 100 * (errors / test_labels)   #performance metrics
accuracy = 100 - np.mean(mape)

rf_small = RandomForestRegressor(n_estimators=1000, max_depth = 3)
rf_small.fit(train_dcBlue, train_labels)
tree_small = rf_small.estimators_[5]

export_graphviz(tree_small, out_file = 'small_tree_quality.dot', feature_names = dcBluefeatures_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree_quality.dot')
graph.write_png('small_tree-quality.png')

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
# =============================================================================

############extra trees regressor ##########
#dcBlue = pd.read_csv('../DC Water Data October 2021/dcWaterStorms(October 2021)/BluePlainsFinal(October2021)_Update(Nov23).csv')
dcBlue = pd.read_csv('../DC Water Data October 2021/dcWaterStorms(October 2021)/BluePlainsFinal(October2021)_Update(Dec14).csv')
dcBlue['datetime'] = pd.to_datetime(dcBlue['datetime'])

labels = np.array(dcBlue['influent_flow_imp'])
dcBlue = dcBlue.drop(['Unnamed: 0','datetime', 'influent_flow_imp'], axis =1)
dcBluefeatures_list = list(dcBlue.columns) ##save features names
dcBlue = np.array(dcBlue)

train_dcBlue, test_dcBlue, train_labels, test_labels = train_test_split(dcBlue, labels, test_size = 0.25, random_state = 42)
et = ExtraTreesRegressor(n_estimators = 1000, random_state = 42)
et.fit(train_dcBlue, train_labels)
predictions = et.predict(test_dcBlue)
errors = abs(predictions - test_labels)
mape = 100 * (errors / test_labels)   #performance metrics
accuracy = 100 - np.mean(mape)

et_small = ExtraTreesRegressor(n_estimators=1000, max_depth = 3)
et_small.fit(train_dcBlue, train_labels)
tree_small = et_small.estimators_[4]

export_graphviz(tree_small, out_file = 'extra_tree_quality.dot', feature_names = dcBluefeatures_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('extra_tree_quality.dot')
graph.write_png('extra_tree_quality.png')

importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(dcBluefeatures_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
