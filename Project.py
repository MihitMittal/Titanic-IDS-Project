#%%
#Importing libraries
import numpy as np
import pandas as pd
import sklearn.metrics
# %%
#Importing datasets
testing_ds = pd.read_csv("titanic_test.csv")
training_ds = pd.read_csv("titanic_train.csv")
# %%
#View the top few rows of training dataset
training_ds.head()
# %%
