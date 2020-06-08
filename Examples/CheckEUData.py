# This file checks the reporting delay in the series of EU data as downloaded from
# https://data.europa.eu/euodp/en/data/dataset/covid-19-coronavirus-data

# The data is in the local folder .\Data\ECDC
import os
from os.path import sep
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

basePath = os.getcwd()
if basePath.endswith('Examples'):
    basePath = basePath[:-9]  # to remove the Examples bit
basePath = basePath + sep+'Data'+sep+'ECDC'+sep
df1 = pd.read_excel(basePath + r"COVID-19-geographic-disbtribution-worldwide-2020-05-15.xlsx")  # support information about the population
df2 = pd.read_excel(basePath + r"COVID-19-geographic-disbtribution-worldwide-2020-05-14.xlsx")  # support information about the population
print(df1.keys())
GER1 = df1['cases'][df1['countriesAndTerritories']=='Germany'].to_numpy()
Dates1 = df1['dateRep'][df1['countriesAndTerritories']=='Germany']
GER2 = df2['cases'][df2['countriesAndTerritories']=='Germany'].to_numpy()
Dates2 = df2['dateRep'][df2['countriesAndTerritories']=='Germany']

GER2 = np.concatenate((np.array([0]),GER2),0)

plt.figure(10)
plt.plot(Dates1, GER1-GER2)
#plt.plot(Dates1, GER1)
#plt.plot(Dates1, GER2)

# CONCLUSION: The ECDC does NOT update cases retrospectively.
# How about death numbers?

GER1 = df1['deaths'][df1['countriesAndTerritories']=='Germany'].to_numpy()
Dates1 = df1['dateRep'][df1['countriesAndTerritories']=='Germany']
GER2 = df2['deaths'][df2['countriesAndTerritories']=='Germany'].to_numpy()
Dates2 = df2['dateRep'][df2['countriesAndTerritories']=='Germany']

GER2 = np.concatenate((np.array([0]),GER2),0)

plt.figure(10)
plt.plot(Dates1, GER1-GER2)

# They also do not update death numbers retrospectively!
