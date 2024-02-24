# A. Problem Statement

A retail store that has multiple outlets across the country are facing issues in managing the
inventory - to match the demand with respect to supply. You are a data scientist, who has to
come up with useful insights using the data and make prediction models to forecast the sales for
X number of months/years.

# B. Project Objective

The objective of this project is to predict Sales of store for the next 12 weeks. As in dataset, size and time related data are given as feature, so analyze if sales are impacted by time-based factors and space- based factor. Most importantly how inclusion of holidays in a week soars the sales in store?

# C. Data Discription

In this dataset, there are historical sales data of 45 Walmart stores based on store location and week. There are certain events and holidays which impact sales on each day. The business is facing a challenge due to unforeseen demands and runs out of stock some times. Walmart would like to predict the sales and demand accurately. The objective is to determine the factors affecting the sales and to analyze the impact of markdowns around holidays on the sales.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from math import sqrt

import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

walmart = pd.read_csv('Walmart.csv')
walmart.head()

"""# D. Data Pre-processing Steps and Inspiration

The Pre-processing of the data includes the following steps:
1. Data Cleaning: Cleaning the data by removing missing values, outliers and other inconsistencies.
2. Data Exploration: Exploring the data to gain insights and understanding the data.
3. Data Visualization: Visualizing the data for better understanding.
"""

walmart.shape

walmart.info()

# Converting 'Date' column to datetime and adding 'Year', 'Month' and 'Week' column

walmart["Date"] = pd.to_datetime(walmart["Date"])
walmart['Year'] =walmart['Date'].dt.year
walmart['Month'] =walmart['Date'].dt.month
walmart['Week'] =walmart['Date'].dt.week

walmart.head()

walmart.info()

walmart.describe()

#Checking Null

walmart.isnull().sum()

#Checking Duplicates

walmart.duplicated().sum()

walmart.groupby('Month')['Weekly_Sales'].mean()

walmart.groupby('Year')['Weekly_Sales'].mean()

# Data Visualization

# Analyzing the distribution of target variable
plt.figure(figsize = (10, 5))
sns.distplot(walmart['Weekly_Sales'], hist_kws=dict(edgecolor="black"))
plt.title('Weekly Sales Distribution', fontsize= 15)
plt.grid()
plt.show()

walmart['Holiday_Flag'].value_counts()

sns.countplot(x = 'Holiday_Flag', data = walmart);

plt.figure(figsize=(20,8))
sns.barplot(walmart['Store'], walmart['Weekly_Sales'])
plt.title('Weekly Sales by Store', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Store', fontsize=16)
plt.grid()
plt.show()

#This function plots the graph relation between a categorized feature and the Weekly_Sales

def graph_relation_to_weekly_sale(col_relation, df, x='Week', palette=None):
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.relplot(
        x=x,
        y='Weekly_Sales',
        hue=col_relation,
        data=df,
        kind='line',
        height=5,
        aspect=2,
        palette=palette
    )
    plt.show()

graph_relation_to_weekly_sale('Year', walmart, x='Date', palette='Set2')

plt.figure(figsize = (20, 7))
sns.barplot(walmart['Week'], walmart['Weekly_Sales'])
plt.title('Average Weekly Sales', fontsize=18)
plt.ylabel('Weekly Sales', fontsize=16)
plt.xlabel('Week', fontsize=16)
plt.grid()
plt.show()

plt.figure(figsize = (20,10))
sns.heatmap(walmart.corr(), cmap = 'PuBu', annot = True)
plt.show()

walmart.drop(['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'], axis = 1, inplace = True)

"""# E. Choosing the Algorithm for the Project

The choice of algorithm for a machine learning project is depends upon the type of problem we are trying to solve. Generally, supervised learning algorithms are used for classification and regression problems, while unsupervised learning algorithms are used for clustering and dimensionality reduction tasks. Some of the most popular algorithms used in machine learning includes Random Forests, Support Vector Machines (SVMs), Extra Trees, k-Nearest Neighbors (kNN), Decision Trees and xgboost.
"""

x = walmart.drop(['Date','Weekly_Sales'], axis=1)
x

y = walmart['Weekly_Sales']

rf = RandomForestRegressor(n_estimators = 100)
rf.fit(x, y)

# checking the feature importance

plt.figure(figsize = (15, 5))
plt.bar(x.columns, rf.feature_importances_)
plt.title("Feature Importance", fontsize = 15)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 0)

"""### Linear Regression"""

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

plt.scatter(y_test, y_pred)

print("R2 Score: ", r2_score(y_test, y_pred))
print("MSE Score: ", mean_squared_error(y_test, y_pred))
print("RMSE : ", sqrt(mean_squared_error(y_test, y_pred)))

"""### Decision Tree"""

dtree = DecisionTreeRegressor()
dtree.fit(x_train, y_train)

y_pred1 = dtree.predict(x_test)

plt.scatter(y_test, y_pred1)

print("R2 Score: ", r2_score(y_test, y_pred1))
print("MSE Score: ", mean_squared_error(y_test, y_pred1))
print("RMSE : ", sqrt(mean_squared_error(y_test, y_pred1)))

"""### Random Forest"""

rf1 = RandomForestRegressor(n_estimators = 100)
rf1.fit(x_train, y_train)

y_pred2 = rf1.predict(x_test)

plt.scatter(y_test, y_pred2)

print("R2 Score: ", r2_score(y_test, y_pred2))
print("MSE Score: ", mean_squared_error(y_test, y_pred2))
print("RMSE : ", sqrt(mean_squared_error(y_test, y_pred2)))

"""### KNN"""

knn = KNeighborsRegressor()
knn.fit(x_train, y_train)

y_pred3 = knn.predict(x_test)

plt.scatter(y_test, y_pred3)

print("R2 Score: ", r2_score(y_test, y_pred3))
print("MSE Score: ", mean_squared_error(y_test, y_pred3))
print("RMSE : ", sqrt(mean_squared_error(y_test, y_pred3)))

"""### Getting Average of Best Models"""

y_pred_final = (y_pred1 + y_pred2  + y_pred4)/3.0

plt.scatter(y_test, y_pred_final)

print("R2 Score: ", r2_score(y_test, y_pred_final))
print("MSE Score: ", mean_squared_error(y_test, y_pred_final))
print("RMSE : ", sqrt(mean_squared_error(y_test, y_pred_final)))

"""# F. Motivation and Reasons For Choosing the Algorithm

I have tried Linear Regression, Decision Tree, Random Forest, KNN and XGBoost algorithms. We can see above Linear Regression And KNN model is not fitting for this dataset but Random Forest, Decision Tree and is working good for this dataset.

# G. Assumptions

It is not possible to accurately forecast the sales for each store for the next 12 weeks using machine learning without additional information. Machine learning algorithms require data to be able to make predictions. This data could include historical sales data, customer demographics, store location, and other factors. Without this data, it is not possible to accurately forecast sales for each store for the next 12 weeks.

# H. Model Evaluation and Techniques

The most accurate way to forecast sales for each store using machine learning is to use a time series forecasting model. This type of model takes into account the historical sales data for each store and uses it to predict future sales. The model can be trained using a variety of techniques, such as neural networks. Once the model is trained, it can be used to make predictions about future sales for each store. Additionally, the model can be evaluated using a variety of metrics.

# I. Inferences from the Same

Walmart can use machine learning to forecast sales for each store. By leveraging historical sales data, Walmart can use predictive analytics to identify patterns and trends in sales and use them to make accurate predictions about future sales. Walmart can also use machine learning to identify factors that influence sales, such as weather, seasonality, and customer demographics. By incorporating these factors into their forecasting models, Walmart can make more accurate predictions about future sales.

# J.Future Possibilities of the Project

The sales for each store can be forecasted using machine learning algorithms such as regression, decision trees, and neural networks. These algorithms can be used to predict the sales for each store based on historical data, such as sales figures from previous years, customer demographics, and other factors. The predictions can then be used to inform decisions about inventory, pricing, and marketing strategies. Additionally, the predictions can be used to identify trends and opportunities for growth.
"""



