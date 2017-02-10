import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForest
from sklearn.ensemble import GradientBoostingRegressor as GradientBoosting
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
#from rdkit import Chem
"""
Read in train and test as Pandas DataFrames
"""
df_train = pd.read_csv("C://Users//wiljin//Documents//Harvard//Junior Year//2nd Semester//Classes//CS 181//practicals//GoG_P1//train2.csv")
df_test = pd.read_csv("C://Users//wiljin//Documents//Harvard//Junior Year//2nd Semester//Classes//CS 181//practicals//GoG_P1//test2.csv")

#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)

#DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)

#Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values
X_train = vals[:test_idx]
X_test = vals[test_idx:]
print("Train features:", X_train.shape)
print("Train gap:", Y_train.shape)
print("Test features:", X_test.shape)
del df_all
del df_test
del df_train
def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

#Dividing test data in 80/20 test train split
X_train_2,X_val_2,Y_train_2,Y_val_2 = train_test_split(X_train, Y_train,test_size=0.2)
#%%
#Modified Original Code to get MSE
LR = LinearRegression()
LR.fit(X_train_2, Y_train_2)
LR_pred = LR.predict(X_test)
#%%
#Random Forest
RF = RandomForest(n_jobs=-1,oob_score=True)
RF.fit(X_train, Y_train)
#RF_pred = RF.predict(X_test)
print("done loading")
#%%
#Tuning RF using exhaustive grid search
rf_param_test = {'n_estimators':[10,30,50,70,90,110],'max_depth':['None',3,5,7,9]}
RF_init = RandomForest(n_jobs=-1, max_features='auto')
RF_search = GridSearchCV(estimator = RF_init, param_grid = rf_param_test,scoring='mean_squared_error',n_jobs=-1,iid=False)
print("done with GridSearch")
RF_search.fit(X_train, Y_train)
print("done with fit")
RF_search.best_params_
#%%
RF = RandomForest(n_jobs=-1, n_estimators=30)
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)
write_to_file("rf4_added.csv",RF_pred)
print("done with rf")
#%%
GB = GradientBoosting(learning_rate = 0.1, n_estimators = 200, subsample = 0.8, random_state=10, min_samples_leaf = 4, max_depth = 5, min_samples_split = 50956)
GB.fit(X_train,Y_train)
ypred = GB.predict(X_test)
print("done fitting")
write_to_file("test2gb_added.csv", ypred)
print("done with gb")

