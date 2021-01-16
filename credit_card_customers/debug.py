# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# CSV data from https://www.kaggle.com/sakshigoyal7/credit-card-customers

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# this line is not really needed, just make plots looks nicer
plt.style.use('bmh')

# %% [markdown]
# ## 1. Data Overview and Cleaning

# %%
# replace the path to your own downloaded csv file path if you are a beginner
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

# %% [markdown]
# Fortunately there is no missing values in this dataset, we do not have to deal with missing values

# %%
print(df.describe())

# %% [markdown]
# It is clear that this dataset contains both numerical and categorical values
# %% [markdown]
# ## 2. Data Visualization
# ### 2.1 Simple Visualization
# %% [markdown]
# Lets descover the non-numerical values in the given table first

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
print("Categorical Variables: " + str(categorical_vars))


# %%
import collections

def plot_cat(df, cat_name):
    """
    This function is to plot single categorical value in a plot
    @type df: pandas dataframe
    @type cat: string, a column with categorical value in df
    """
    f, ax = plt.subplots()
    # val_dict = {cat1: count_cat_2, cat2:count_cat_2}
    val_dict = collections.Counter(df[cat_name].values)
    names = [key for key in val_dict]
    heights = [val for val in val_dict.values()]
    ax.bar(names, heights)
    # if you just want to show the image, you don't have to return f
    # if you want the image item for further usage, like to save if ,you can return f, which is the figure itself
    return f

def plot_all(df, cat_names, l = 3, fsize = (18, 10)):
    """
    Plot all categorical value in one figure
    @type df: pandas dataframe
    @type cat_names: List[String], list contains categorical values
    @type l: int, how many subplots will be in one line of the plot (default 3)
    @type fsize: tuple, length and height of the plot
    """
    h = len(cat_names) // l if len(cat_names) % l == 0 else len(cat_names) // l + 1
    f, ax = plt.subplots(h, l, figsize = fsize)
    for i in range(h):
        for j in range(l):
            idx = i * l + j
            if idx >= len(cat_names):
                break
            cat_name = cat_names[idx]
            val_dict = collections.Counter(df[cat_name].values)
            names = [key for key in val_dict]
            for name in names:
                # if the value in current plot is too long, rotate the x label in the current subplot
                # for cagetorical variable with less category, we do not have to rotate them. You can modify 50 to other values to see what will happend
                if len(name) * len(names) > 50:
                    ax[i][j].tick_params(axis='x', rotation=30)
                    break
            heights = [val for val in val_dict.values()]      
            # use random rgb value to plot, you can delete color = if you just want the default blue color.
            # seaborn library also available to apply color scheme on plot
            rects = ax[i][j].bar(names, heights)
            ax[i][j].set_title(cat_name)
            # percentage label
            cnt = sum(heights)
            for rect in rects:
                w = rect.get_width()
                x = rect.get_x()
                h = rect.get_height()
                ax[i][j].text(rect.get_x() + rect.get_width()/2, h, str(round(h/cnt * 100, 3)) + "%", ha = "center", va = "bottom")
    for extra in range(j, l):
        f.delaxes(ax[i][extra])
    return f


# %%
# not really needed, only because these random seed make the generated figure looks nicer
cat_plots = plot_all(df, categorical_vars)

# %% [markdown]
# The other columns contains numerical values

# %%
num_vals = []
for item in df.columns:
    if item not in categorical_vars:
        num_vals.append(item)
print("Numerical Variables: " + str(num_vals))


# %%
def plot_all_num(df, num_names, l = 4,  fsize = (18, 15)):
    h = len(num_names) // l if len(num_names) % l == 0 else len(num_names) // l + 1
    f, ax = plt.subplots(h, l, figsize = fsize)
    for i in range(h):
        for j in range(l):
            idx = i * l + j
            if idx >= len(num_names):
                break
            num_name = num_names[idx]
            cnt = collections.Counter(df[num_name])
            key_num = len(cnt)
            ax[i][j].hist(df[num_name], bins = min(50, key_num))
            ax[i][j].set_title(num_names[idx])
    return f


# %%
num_plots = plot_all_num(df, num_vals)


# %%
corrs = df.loc[: ,num_vals].corr().abs()
print(corrs)

# %% [markdown]
# Seaborn is the best way to draw heatmap.
# You can draw density plot and bar plots by seaborn as well.

# %%
import seaborn as sns
plt.figure(figsize = (15, 15))
sns.heatmap(corrs, annot = True, fmt = '.3f', cmap = 'coolwarm')

# %% [markdown]
# Let's print out the pairs of variables which have too high correlation coefficient(corr > 0.9)

# %%
relation_dict = dict()
for colname in corrs.columns:
    for rowname in corrs.index:
        pair = tuple(sorted([colname, rowname]))
        if pair not in relation_dict and corrs[colname][rowname] > 0.9 and colname != rowname:
            relation_dict[pair] = corrs[colname][rowname]
for key, val in relation_dict.items():
    print(str(key) + ':' + str(val))

# %% [markdown]
# Now we only have two valiables which has high correlation coefficient
# 
# We have to determine which to remove to avoid multicollinearity

# %%
import statistics as st


# %%
corrs_avg_to_buy = corrs["Avg_Open_To_Buy"]
corrs_credit_limit = corrs["Credit_Limit"]
mean_avg_to_buy = st.mean(corrs_avg_to_buy)
mean_credit_limit = st.mean(corrs_credit_limit)
mid_avg_to_buy = st.median(corrs_avg_to_buy)
mid_credit_limit = st.median(corrs_credit_limit)
print("Mean of correlations in variable Avg_Open_To_Buy: " + str(mean_avg_to_buy))
print("Mean of correlations in variable Credit_Limit: " + str(mean_credit_limit)) 
print("Median of correlations in variable Avg_Open_To_Buy" + str(mid_avg_to_buy))
print("Median of correlations in variable Credit_Limit:" + str(mid_credit_limit))

# %% [markdown]
# Simply drop Avg_Open_To_Buy according to the mean and medium of its correlation coefficients and the maximum correlation coefficient from the heat map.

# %%
df.drop("Avg_Open_To_Buy", inplace = True, axis = 1)
print("The shape of current dataset:")
print(df.shape)

# %% [markdown]
# ## 2.2 Explore Relationships
# 
# We are also interested in customer and non-customer in categorical values.
# 
# One example of our interest is:
# if this person in our dataset is a male, what is the probability of him already become our customer?

# %%
def plot_in_cats(dt, cat_vals, to_explore, l = 3, figsize = (20, 15)):
    # you cannot compare yourself with yourself
    if to_explore in cat_vals:
        cat_vals.remove(to_explore)
    h = len(cat_vals) // l if len(cat_vals) % l == 0 else len(cat_vals) // l + 1
    f, a = plt.subplots(h, l, figsize = figsize)
    plt.rc('font', size=7)
    for i in range(h):
        for j in range(l):
            idx = i * l + j
            if idx >= len(cat_vals):           
                break
            cat_name = cat_vals[idx]
            ax = sns.countplot(cat_name, data = dt, hue = to_explore, ax = a[i][j])
            cat_strs = set(dt[cat_name].values)
            for item in cat_strs:
                if len(item) * len(cat_strs) > 50:
                    ax.tick_params(axis='x', rotation=30)
                    break
            for c in ax.containers:
                s = sum([p.get_height for p in c.patches])
                for p in c.patches:
                    h = p.get_height()
                    x = p.get_x()
                    w = p.get_width()
                    ax.text(x + w / 2 , h, str(round(h/s * 100, 3)) + "%", ha = "center")
    for extra in range(j, l):
        f.delaxes(a[i][extra])
    # we cannot return f here because seaborn will draw extra plot in notebook if you do it
    # use matplotlib if it not spends too much time for you in coding, seaborn sometimes cause some unpredictable problems(but not bugs).


# %%
plot_in_cats(df, categorical_vars, "Attrition_Flag")

# %% [markdown]
# ## 2.3 Feature Engineering

# %%



