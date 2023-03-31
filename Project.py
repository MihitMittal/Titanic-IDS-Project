# %%
# Importing libraries
import numpy as np
import pandas as pd
import sklearn.metrics
from matplotlib import pyplot as plt
# %%
# Importing datasets
testing_ds = pd.read_csv("titanic_test.csv")
training_ds = pd.read_csv("titanic_train.csv")
# %%
# View the top few rows of training dataset
training_ds.head()
# %%
# Check for Missing data (Age / Cabin)
print(training_ds[pd.isnull(training_ds["Age"])])
print(training_ds[pd.isnull(training_ds["Cabin"])])
# %%
# Data Cleaning: Impute missing values in Age based Pclass (take average of age in Pclass)
avg_age = training_ds["Age"].mean()
training_ds["Age"].fillna(avg_age, inplace=True)
# %%
# Data Cleaning: Drop the Cabin Column
training_ds.drop("Cabin", axis=1)
# %%
# Data Cleaning: Drop the row in Embarked column that is NaN.
