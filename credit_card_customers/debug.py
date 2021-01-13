# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
df = pd.read_csv("./BankChurners.csv")
print("The shape of current dataset:")
print(df.shape)
df.sample(5)


# %%
# those three columns are not useful for analyzing, we need to drop them
to_drop = ["CLIENTNUM","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"]
for colname in to_drop:
    df.drop(colname, inplace = True, axis = 1)


# %%
print("The shape of current dataset:")
print(df.shape)
df.sample(5)


# %%
print(df.info())


# %%
print(df.describe())

# %% [markdown]
# Lets descover the non-numerical values in the given table

# %%
first_line = df.iloc[0]
print(first_line)

# %% [markdown]
# As what we see in the table, the first line of dataframe only contains values like "existing customer", "45", "M", "3".... and the first column we print above contains the name of the column.
# 
# So it must be a hash-table type of data (in Python, we call it dictionary) with keys and values.
# 
# Therefore, we can use its values to decide if it is a categorical value, or numerical value.
# 
# But this is not a safe way, if we print out the one column, like this:

# %%
print(df["Attrition_Flag"])
print(type(df["Attrition_Flag"]))

# %% [markdown]
# You can see a feature names "dtype".
# If you are using ide for python debugging instead of jupyter notebook and put a break point on a variable x (x = df\["Attrition_Flag"\]), you can see that dtype is a class method for andas.core.series.Series

# %%
# first mentioned method ,not safe but it is a way if you are not familiar with pandas
# categorical_vars = []
# for key, val in first_line.items():
#     try:
#         num = float(val)
#     except:
#         categorical_vars.append(key)
# print(categorical_vars)


# %%
# second method
categorical_vars = []
for col_name in df.columns:
    if df[col_name].dtype == "object":
        categorical_vars.append(col_name)
print(categorical_vars)

a = df["Attrition_Flag"]
# %%
import collections

def plot_cat(df, cat_name):
    """
    @type df: pandas dataframe
    @type cat: string, a column with categorical value in df
    """
    f, ax = plt.subplots()
    val_dict = collections.Counter([val for val in df[cat_name].values()])


