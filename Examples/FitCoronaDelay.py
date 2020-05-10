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
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

if False:
    AllMeasured = loadData(useThuringia = False, pullData=False)
    AllMeasured = preprocessData(AllMeasured, ReduceDistricts=None, SumDistricts=True, SumAges=True, SumGender=True)
else:
    AllMeasured = loadData(r"COVID-19 Linelist 2020_05_09.xlsx", useThuringia = True, pullData=False, lastDate='09.05.2020')
    # ExampleRegions = ['SK Jena', 'LK Greiz', 'SK Gera'] # 'SK Gera',
    # AllMeasured = preprocessData(AllMeasured, ReduceDistricts=ExampleRegions, SumDistricts=False, SumAges=True, SumGender=True)
    AllMeasured = preprocessData(AllMeasured, ReduceDistricts=None, SumDistricts=True, SumAges=True, SumGender=True)

M = CoronaDelayModel(AllMeasured, Tmax = 150, lossWeight={'cases':1.0, 'deaths': 10.0})

tf.config.experimental_run_functions_eagerly(True)
M.doFit(0)
M.showStates(MinusOne=['S'])
M.DataDict={}
g = M.getGUI(showResults=M.showSimRes, doFit=M.doFit)