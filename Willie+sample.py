
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as RandomForest
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.cross_validation import train_test_split



# In[3]:

from sklearn.ensemble import GradientBoostingRegressor as GradientBoosting
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV



# In[4]:

"""
Read in train and test a?s Pandas DataFrames
"""
df_train = pd.read_csv("train2.csv")
df_test = pd.read_csv("test2.csv")




# In[5]:

#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)


# In[6]:

#DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)
df_all.head()


# In[12]:

"""
Example Feature Engineering

this calculates the length of each smile string and adds a feature column with those lengths
Note: this is NOT a good feature and will result in a lower score!
"""
#smiles_len = np.vstack(df_all.smiles.astype(str).apply(lambda x: len(x)))
#df_all['smiles_len'] = pd.DataFrame(smiles_len)


# In[8]:

#Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values
#print "Train features:", X_train.shape
#print "Train gap:", Y_train.shape
#print "Test features:", X_test.shape


# In[14]:

X_train = df_all.values[:test_idx]
X_test = df_all.values[test_idx:]

test2 = GradientBoosting(learning_rate = 0.1, n_estimators = 200, subsample = 0.8, random_state=10, min_samples_leaf = 4, max_depth = 5, min_samples_split = 50956)
test2.fit(X_train,Y_train)
ypred = GB.predict(X_test)

# In[26]:

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")


# In[ ]:




# In[72]:

write_to_file("gb2.csv", ypred)


# In[ ]:



