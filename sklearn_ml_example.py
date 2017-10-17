#import modules
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# @ G. Gibson

#import libraries
import numpy as np
import pandas as pd

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url)
#make data pretty
data = pd.read_csv(dataset_url, sep=';')
print data.head()
print data.shape
print data.describe()

#split data into training and testing sets
y = data.quality
x = data.drop('quality', axis=1)
#implement scikit-learn train_test_split function
#random_state is a seed file number so we can reproduce results
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 123, stratify = y)

#stratify--make sure training set looks similar to test set to make eval more reliable
#now standardize (subtract mean and divide difference by SD)
#scikit-learn simple scalin = X_train_scaled = preprocessing.scale(X_train)
#only works on training set not test set
#transformer api in scikit allows for fitting current and future data sets

#fit transformer to training set
scaler = preprocessing.StandardScaler().fit(X_train)
#apply transformer to training set
X_train_scaled = scaler.transform(X_train)
print X_train_scaled.mean(axis=0)
print X_train_scaled.std(axis=0)
#apply transformer to test set
X_test_scaled = scaler.transform(X_test)
print X_test_scaled.mean(axis=0)
print X_test_scaled.std(axis=0)
#modeling pipeling = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=1000))
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))
#set hyperparameters
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
#cross-validation across preprocessing and model algorithm
#10 is the typical number of folds in which data is broken up
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
#fit and tune model
clf.fit(X_train, y_train)
#predict a new data set
y_pred = clf.predict(X_test)
#evaluate model performance
print r2_score(y_test,y_pred)
print mean_squared_error(y_test,y_pred)
#is the performance good enough?
#improve model by:
#1.  try other regression models (reg regression, boosted tree, etc)
#2.  collect more data
#3.  engineer smarter features after looking into exploratory analysis
#4.  speak to a domain expert (wine tasting)

#save model to a pkl file
joblib.dump(clf, 'rf_regressor.pkl')
#load model
#clf2 = joblib.loead('rf_regressor.pkl')
#predict data set using loaded model
#clf2.predict(X_test)
