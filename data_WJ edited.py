import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as GradientBoosting

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

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

RF = RandomForestRegressor(n_jobs=-1,oob_score=True)
RF.fit(X_train, Y_train)
#RF_pred = RF.predict(X_test)
print("done loading")
#%%
RF = RandomForestRegressor(n_jobs=-1, n_estimators=30)
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
