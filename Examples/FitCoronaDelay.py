# This example is written for the new interface
# This is the full COVID-19 model to be fitted to the RKI data
# see the PPT for details of the model design

import StateModeling as stm
import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from Corona.LoadData import loadData, preprocessData
from Corona.CoronaModel import CoronaDelayModel, plotTotalCases
from bokeh.io import push_notebook, show, output_notebook
from correct_deaths_new import PreprocessDeaths
from correct_deaths_new import PreprocessDeaths
from Deaths_RKI_Format import reformatDeaths
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
# sys.path.append(r'../RKI_COVID19')
# sys.path.append(r'C:\\Users\\pi96doc\\Documents\\Programming\\RKI_COVID19')
#import RKI_COVID19_DB
#from RKI_COVID19 import RKI_COVID19_DB
import os
from os import sep
basePath = os.getcwd()

DataStruct = 'Michael'

if DataStruct == 'Rainer':
    DataDir = r'C:\Users\pi96doc\Documents\Programming\PythonScripts\FromWeb\CoronaData\CSV-Dateien-mit-Covid-19-Infektionen-'
    # PreprocessDeaths(DataDir)
    reformatDeaths(DataDir, NumThreads=8)

usePreprocessed = True # use the specially preprocessed RKI Data
SumAges = False
SumGender = True
TimeRange =  [30,90] # TimeRange = None

Filename = basePath + sep +'..'+ sep + r'Data' + sep + 'PreprocessedMeasured_A'+str(SumAges)+'_G'+str(SumGender)
if True:  # reload data (or use preprocessed)
    if True:
        AllMeasured = loadData(useThuringia = False, pullData=False, usePreprocessed=True)
        if False:
            ExampleRegions = ['SK Gera', 'SK Jena', 'LK Nordhausen', 'SK Erfurt', 'SK Suhl', 'LK Weimarer Land', 'SK Weimar', 'LK Greiz',
                              'LK Schmalkalden-Meiningen', 'LK Eichsfeld', 'LK Sömmerda', 'LK Hildburghausen',
                              'LK Saale-Orla-Kreis', 'LK Saale-Holzland-Kreis', 'LK Kyffhäuserkreis', 'LK Saalfeld-Rudolstadt', 'LK Ilm-Kreis',
                              'LK Unstrut-Hainich-Kreis', 'LK Gotha', 'LK Sonneberg', 'SK Eisenach', 'LK Altenburger Land',
                              'LK Wartburgkreis']
        else:
            ExampleRegions = None
            # ExampleRegions = ['SK Jena', 'LK Greiz', 'SK Gera', 'LK Sonneberg']  # 'SK Gera',
        AllMeasured = preprocessData(AllMeasured, ReduceDistricts=ExampleRegions, SumDistricts=True, SumAges=False, SumGender=True, TimeRange=TimeRange)
        # AllMeasured['Cases'] = np.transpose(AllMeasured['Cases'],(0,2,3,1))
        # AllMeasured['Dead'] = np.transpose(AllMeasured['Dead'],(0,2,3,1))

    else:
        AllMeasured = loadData(r"COVID-19 Linelist 2020_05_11.xlsx", useThuringia = True, pullData=False, lastDate='09.05.2020')
        if True:
            ExampleRegions = ['SK Jena', 'LK Greiz', 'SK Gera', 'LK Sonneberg'] # 'SK Gera',
            AllMeasured = preprocessData(AllMeasured, ReduceDistricts=ExampleRegions, SumDistricts=False, SumAges=True, SumGender=True, TimeRange=TimeRange)
        else:
            AllMeasured = preprocessData(AllMeasured, ReduceDistricts=None, SumDistricts=True, SumAges=True, SumGender=True, TimeRange=TimeRange)

    np.save(Filename, AllMeasured)
else:
    AllMeasured = np.load(Filename + '.npy', allow_pickle=True).item()

AllMeasured['Cases'] = AllMeasured['Cases'][:,np.newaxis,:,:,:]  # to account for the (empty) Disease Progression axis
AllMeasured['Dead'] = AllMeasured['Dead'][:,np.newaxis,:,:,:]
# AllMeasured['Population'] = np.squeeze(AllMeasured['Population'])
print(AllMeasured['Cases'].shape)

lossWeights = {'cases':0.1,'hospitalization':0.1,'deaths': 0.1}
M = CoronaDelayModel(AllMeasured, Tmax = AllMeasured['Cases'].shape[0], lossWeight=lossWeights)

tf.config.experimental_run_functions_eagerly(True)
M.plotMatplotlib = True
M.doFit(0)
M.showStates(MinusOne=['S'])
M.DataDict={}
g = M.getGUI(showResults=M.showSimRes, doFit=M.doFit)

# M.showResultsBokeh()

