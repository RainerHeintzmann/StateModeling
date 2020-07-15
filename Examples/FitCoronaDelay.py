# This example is written for the new interface
# This is the full COVID-19 model to be fitted to the RKI data
# see the PPT for details of the model design

import StateModeling as stm
import numpy as np
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
DataStruct = 'Rainer'

if DataStruct == 'Rainer':
    DataDir = r'C:\Users\pi96doc\Documents\Programming\PythonScripts\FromWeb\CoronaData\CSV-Dateien-mit-Covid-19-Infektionen-'
    # PreprocessDeaths(DataDir)
    reformatDeaths(DataDir, NumThreads=8)


if True:
    AllMeasured = loadData(useThuringia = False, pullData=False)
    ExampleRegions = ['SK Jena', 'LK Greiz', 'SK Gera', 'LK Sonneberg']  # 'SK Gera',
    AllMeasured = preprocessData(AllMeasured, ReduceDistricts=ExampleRegions, SumDistricts=False, SumAges=True, SumGender=True)
    # AllMeasured['Cases'] = np.transpose(AllMeasured['Cases'],(0,2,3,1))
    # AllMeasured['Dead'] = np.transpose(AllMeasured['Dead'],(0,2,3,1))

else:
    AllMeasured = loadData(r"COVID-19 Linelist 2020_05_11.xlsx", useThuringia = True, pullData=False, lastDate='09.05.2020')
    if True:
        ExampleRegions = ['SK Jena', 'LK Greiz', 'SK Gera', 'LK Sonneberg'] # 'SK Gera',
        AllMeasured = preprocessData(AllMeasured, ReduceDistricts=ExampleRegions, SumDistricts=False, SumAges=True, SumGender=True)
    else:
        AllMeasured = preprocessData(AllMeasured, ReduceDistricts=None, SumDistricts=True, SumAges=True, SumGender=True)

AllMeasured['Cases'] = np.squeeze(AllMeasured['Cases'])[:,np.newaxis,np.newaxis,:]
AllMeasured['Dead'] = np.squeeze(AllMeasured['Dead'])[:,np.newaxis,np.newaxis,:]
AllMeasured['Population'] = np.squeeze(AllMeasured['Population'])
print(AllMeasured['Cases'].shape)

M = CoronaDelayModel(AllMeasured, Tmax = AllMeasured['Cases'].shape[0], lossWeight={'cases':1.0, 'deaths': 10.0})

tf.config.experimental_run_functions_eagerly(True)
M.doFit(0)
M.showStates(MinusOne=['S'])
M.DataDict={}
g = M.getGUI(showResults=M.showSimRes, doFit=M.doFit)