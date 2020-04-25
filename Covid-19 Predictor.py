#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:03:18 2020

@author: shoumik
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ban=pd.read_csv('COVID-19_in_bangladesh.csv')

# Taking confirmed cases since first case appear in 3/08/2020
cases = ban['cases'].groupby(ban['Date']).sum().sort_values(ascending=True)
cases = cases[cases>0].reset_index().drop('Date',axis=1)

deaths = ban['deaths'].groupby(ban['Date']).sum().sort_values(ascending=True)
deaths = deaths[deaths>0].reset_index().drop('Date',axis=1)

# add new 3 days here
cases = cases[0:70]
deaths = deaths[0:70]

# Converting our data into a array
days_since_first_case = np.array([i for i in range(len(cases.index))]).reshape(-1, 1)
bd_cases = np.array(cases).reshape(-1, 1)

days_since_first_death = np.array([i for i in range(len(deaths.index))]).reshape(-1, 1)
bd_deaths = np.array(deaths).reshape(-1, 1)

#Preparing indexes to predict next 3  days
days_in_future = 3
future_forcast = np.array([i for i in range(len(cases.index)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-3]

future_forcast_deaths = np.array([i for i in range(len(deaths.index)+days_in_future)]).reshape(-1, 1)
adjusted_dates_deaths = future_forcast_deaths[:-3]

#Splitting data into train and test to evaluate our model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(days_since_first_case , bd_cases , test_size= 3, shuffle=False , 
                                                    random_state = 42) 

X_train_death, X_test_death, y_train_death, y_test_death = train_test_split(days_since_first_death, bd_deaths, test_size= 3 ,
                                                                            shuffle=False, random_state = 42) 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error


 # Transform our cases data for polynomial regression
poly = PolynomialFeatures(degree=4)
poly_X_train = poly.fit_transform(X_train)
poly_X_test = poly.fit_transform(X_test)
poly_future_forcast = poly.fit_transform(future_forcast)
   
# polynomial regression cases
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train, y_train)
test_linear_pred = linear_model.predict(poly_X_test)
linear_pred = linear_model.predict(poly_future_forcast)

# evaluating with MAE and MSE
print('MAE:', mean_absolute_error(test_linear_pred, y_test))   

cases_pred_vsl=plt.figure(figsize=(13, 7))
plt.plot(adjusted_dates , bd_cases , label = "Real cases")
plt.plot(future_forcast , linear_pred , label = "Polynomial Regression Predictions", linestyle='dashed', color='orange')

plt.title('Cases in Bangladesh over the time: Predicting Next 3 days', size=20)
plt.xlabel('Days Since 3/08/20', size=20)
plt.ylabel('Cases', size=20)
plt.xticks(size=15)
plt.yticks(size=15)
plt.axvline(len(X_train), color='black', linestyle="-.", linewidth=1)
plt.axvline(len(X_train)+2, color='green', linestyle="--", linewidth=2.5)
plt.text(18, 5000 , "model training", size = 15, color = "black")
plt.text((len(X_train)+0.2), 15000, "prediction", size = 15, color = "black")

# defyning legend config
plt.legend(loc = "upper left" , frameon = True, ncol = 2 , fancybox = True, framealpha = 0.95
           , shadow = True , borderpad = 1 , prop={'size': 15})

plt.show();

# Transform our death data for polynomial regression
poly_death = PolynomialFeatures(degree=4)
poly_X_train_death = poly_death.fit_transform(X_train_death)
poly_X_test_death = poly_death.fit_transform(X_test_death)
poly_future_forcast_death = poly_death.fit_transform(future_forcast_deaths)    

# polynomial regression deaths
linear_model_death = LinearRegression(normalize=True, fit_intercept=False)
linear_model_death.fit(poly_X_train_death, y_train_death)
test_linear_pred_death = linear_model_death.predict(poly_X_test_death)
linear_pred_death = linear_model_death.predict(poly_future_forcast_death)

# evaluating with MAE and MSE
print('MAE:', mean_absolute_error(test_linear_pred_death, y_test_death)) 
 
deaths_pred_vsl=plt.figure(figsize=(13,7))
plt.plot(adjusted_dates_deaths, bd_deaths, label = "Real deaths")
plt.plot(future_forcast_deaths , linear_pred_death , label = "Polynomial Regression Predictions" ,
         linestyle='dashed' , color='red')

plt.title('Deaths in Bangladesh over the time: Predicting Next 3 days', size=20)
plt.xlabel('Days Since 03/08/20', size=20)
plt.ylabel('Deaths', size=20)
plt.xticks(size=15)
plt.yticks(size=15)
plt.axvline(len(X_train_death), color='black', linestyle="-.", linewidth=1)
plt.axvline(len(X_train_death)+2, color='green', linestyle="--", linewidth=2.5)
plt.text(10, 200 , "model training" , size = 15, color = "black")
plt.text((len(X_train_death)+0.2), 600, "prediction", size = 15 , color = "black")

# defyning legend config
plt.legend(loc = "upper left" , frameon = True, ncol = 2 , fancybox = True, framealpha = 0.95, shadow = True, borderpad = 1, prop={'size': 15})

plt.show();       

#picking for api
import pickle
pickle.dump(linear_pred[-3:],open('model_cases.pkl','wb'))
pickle.dump(linear_pred_death[-3:],open('model_deaths.pkl','wb'))
pickle.dump(cases_pred_vsl,open('cases_figure.pkl','wb'))
pickle.dump(deaths_pred_vsl,open('deaths_figure.pkl','wb'))

#loading the dumped files
loaded_cases=pickle.load(open('model_cases.pkl','rb'))
loaded_deaths=pickle.load(open('model_deaths.pkl','rb'))
cases_pred_vsl=pickle.load(open('cases_figure.pkl','rb'))
deaths_pred_vsl=pickle.load(open('deaths_figure.pkl','rb'))
                                                        