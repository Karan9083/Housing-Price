#!/usr/bin/env python
# coding: utf-8

# In[57]:


#Import the all libaries

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


# In[58]:


#Read the dataset and print the data set
df_house=pd.read_csv("housing.csv")
df_house.head()


# In[14]:


import math
print(math.log(452600))


# In[15]:


df_house.columns


# In[59]:


#Check the null value
df_house.isnull().sum()


# In[60]:


#Fill the missing values
df_house.total_bedrooms=df_house.total_bedrooms.fillna(df_house.total_bedrooms.mean())
df_house.isnull().sum()


# In[61]:


#Encoding data
le = LabelEncoder()
df_house['ocean_proximity']=le.fit_transform(df_house['ocean_proximity'])


# In[62]:


#Standardize data

# Get column names first
names = df_house.columns
# Create the Scaler object
scaler = StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_house)
scaled_df = pd.DataFrame(scaled_df, columns=names)
scaled_df.head()


# In[63]:


#Visualize relationship between features and target

fig,axs=plt.subplots(1,3,sharey=True)
scaled_df.plot(kind='scatter',x='longitude',y='median_house_value',ax=axs[0],figsize=(16,8))
scaled_df.plot(kind='scatter',x='latitude',y='median_house_value',ax=axs[1],figsize=(16,8))
scaled_df.plot(kind='scatter',x='housing_median_age',y='median_house_value',ax=axs[2],figsize=(16,8))

#plot graphs
fig,axs=plt.subplots(1,3,sharey=True)
scaled_df.plot(kind='scatter',x='total_rooms',y='median_house_value',ax=axs[0],figsize=(16,8))
scaled_df.plot(kind='scatter',x='total_bedrooms',y='median_house_value',ax=axs[1],figsize=(16,8))
scaled_df.plot(kind='scatter',x='population',y='median_house_value',ax=axs[2],figsize=(16,8))

#plot graphs
fig,axs=plt.subplots(1,3,sharey=True)
scaled_df.plot(kind='scatter',x='households',y='median_house_value',ax=axs[0],figsize=(16,8))
scaled_df.plot(kind='scatter',x='median_income',y='median_house_value',ax=axs[1],figsize=(16,8))
scaled_df.plot(kind='scatter',x='ocean_proximity',y='median_house_value',ax=axs[2],figsize=(16,8))


# In[21]:


for column in scaled_df:
    plt.figure()
    sns.boxplot(x=scaled_df[column])


# In[64]:


#Extract X and Y data

X_Features=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'ocean_proximity']
X=scaled_df[X_Features]
Y=scaled_df['median_house_value']

print(type(X))
print(type(Y))


# In[65]:


print(df_house.shape)
print(X.shape)
print(Y.shape)


# In[68]:


#Split the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)


# In[69]:


#Perform Linear Regression
linreg=LinearRegression()
linreg.fit(x_train,y_train)


# In[70]:


LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)


# In[71]:


y_predict = linreg.predict(x_test)


# In[72]:


print(sqrt(mean_squared_error(y_test,y_predict)))
print((r2_score(y_test,y_predict)))


# In[73]:


#Perform Decision Tree Regression
dtreg=DecisionTreeRegressor()
dtreg.fit(x_train,y_train)


# In[74]:


DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
                      max_leaf_nodes=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      presort=False, random_state=None, splitter='best')


# In[75]:


y_predict = dtreg.predict(x_test)
print(sqrt(mean_squared_error(y_test,y_predict)))
print((r2_score(y_test,y_predict)))


# In[77]:


from sklearn.ensemble import RandomForestRegressor
RFregressor = RandomForestRegressor()
RFregressor.fit(x_train, y_train)


# In[81]:


predictionRF = RFregressor.predict(x_test)


# In[79]:


from sklearn.metrics import mean_squared_error
mseRF = mean_squared_error(y_test, predictionRF)
print('Root mean squared error from Random Forest Regression = ')
print(mseRF)


# In[82]:


#Hypothesis testing and P values

lm=smf.ols(formula='median_house_value ~ longitude+latitude+housing_median_age+total_rooms+total_bedrooms+population+households+median_income+ocean_proximity',data=scaled_df).fit()


# In[83]:


lm.summary()


# In[47]:


x_train_Income=x_train[['median_income']]
x_test_Income=x_test[['median_income']]


# In[48]:


print(x_train_Income.shape)
print(y_train.shape)


# In[49]:


linreg=LinearRegression()
linreg.fit(x_train_Income,y_train)
y_predict = linreg.predict(x_test_Income)


# In[50]:


#print intercept and coefficient of the linear equation
print(linreg.intercept_, linreg.coef_)
print(sqrt(mean_squared_error(y_test,y_predict)))
print((r2_score(y_test,y_predict)))


# In[51]:


#plot least square line
scaled_df.plot(kind='scatter',x='median_income',y='median_house_value')
plt.plot(x_test_Income,y_predict,c='red',linewidth=2)


# In[52]:


lm=smf.ols(formula='median_house_value ~ median_income',data=scaled_df).fit()


# In[53]:


lm.summary()


# In[ ]:


##The P value is 0.000 indicates strong evidence against the null hypothesis, so you reject the null hypothesis.
 ##so, there is a strong relationship between median_house_value and median_income

