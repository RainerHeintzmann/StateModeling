# This example is written for the new interface
# This is the full COVID-19 model to be fitted to the RKI data
# see the PPT for details of the model design

import StateModeling as stm
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from Corona.LoadData import loadData, preprocessData
from Corona.CoronaModel import CoronaModel, plotTotalCases, plotRaw

#data = loadData(r"COVID-19 Linelist 2020_04_23.xlsx", useThuringia = True, pullData=False)
# data = loadData(r"COVID-19 Linelist 2020_04_30.xlsx", useThuringia = True, pullData=False)
# AllMeasured = preprocessData(AllMeasured)
AllMeasured = loadData(useThuringia = False, pullData = False)
ExampleRegions = ['SK Jena', 'LK Greiz'] # 'SK Gera',
AllMeasured = preprocessData(AllMeasured, ReduceDistricts=ExampleRegions, SumDistricts=False, SumAges=True, SumGender=True)
# plotRaw(AllMeasured)

# M = CoronaDelayModel(AllMeasured, Tmax = AllMeasured['Cases'].shape[0], lossWeight=lossWeights)
M = CoronaModel(AllMeasured)

if False:
    mobdat = AllMeasured['mobility']
    mobdate = mobdat['date'].to_numpy()
    plt.figure('Retail and recreation');plt.plot(mobdat['retail_and_recreation_percent_change_from_baseline'].to_numpy())
    offsetDay=0; plt.xticks(range(offsetDay, len(mobdate), 7), [date for date in mobdate[offsetDay:-1:7]], rotation="vertical")
    plt.ylabel('Percent Change'); plt.tight_layout()

Tmax = 120

# M.toFit(['r0', 'hr', 'ht0', 'I0'])
# M.toFit(['r0', 'I0'])
M.toFit(['r0', 'h', 'aT0', 'aBase', 'I0', 'd', 'rd', 'T0', 'q']) # 'q',
# M.toFit(['r0'])

# if Cases.shape[-1] > 1:
#     M.toFit.append()
if AllMeasured['Cases'].shape[-2] > 1:
    M.appendToFit(['Age Border', 'Age Sigma'])

# g = M.getGUI()

lossWeights = {'cases':0.1,'deaths': 0.1}

M.DataDict={}
g = M.getGUI(showResults=M.showSimRes, doFit=M.doFit, Dates = list(AllMeasured['Dates']))


PopSum = np.sum(AllMeasured['Population'])
measured = AllMeasured['Cases'][:,  np.newaxis, :, :, :] / PopSum
measuredDead = AllMeasured['Dead'][:, np.newaxis, :, :, :] / PopSum
NIter = 500 # 200

xlim = None  # (60,100)
# fittedVars, fittedRes = M.fit({'detected': measured}, Tmax, otype=otype, oparam=oparam, NIter=NIter, verbose=True, lossScale=lossScale)
FitDict = {'cases': measured}
if "Hospitalized" in AllMeasured.keys():
    FitDict['hospitalization'] = AllMeasured['Hospitalized'][:, np.newaxis, :, :, :]/ PopSum
FitDict['deaths'] = measuredDead

# SimDict = {'cases': None, 'cumul_cases': None, 'cumul_dead':None}
if True:
    simulated = M.simulate('simulated', FitDict, Tmax=Tmax)
    M.showResults(title=AllMeasured['Region'], Scale=PopSum, ylabel='occupancy', xlim=xlim, dims=("District"), Dates=AllMeasured['Dates'], legendPlacement='upper right', styles=['.','-','--'])
    M.showStates(MinusOne=('S'), dims2d=None, Dates = AllMeasured['Dates'])

if True:
    otype = "L-BFGS"
    lossScale = 1.0  # 1e4
    oparam = {"normFac": 'max'}
else:
    lossScale = None
    otype = "nesterov"  # "adagrad"  "adadelta" "SGD" "nesterov"  "adam"
    learnrate = {"nesterov": 1000.0, "adam": 7e-7}
    oparam = {"learning_rate": tf.constant(learnrate[otype], dtype=stm.CalcFloatStr)}
# oparam['noiseModel'] = 'Poisson'
oparam['noiseModel'] = 'Gaussian'
# oparam['noiseModel'] = 'ScaledGaussian'  # is buggy? Why the NaNs?

# tf.config.experimental_run_functions_eagerly(True)

fittedVars, fittedRes = M.fit(FitDict, Tmax, otype=otype, oparam=oparam, NIter=NIter, verbose=True, lossScale=lossScale)
# YMax =np.max(np.sum(AllMeasured['Cases']) / sum(AllMeasured['Population']))
# M.showResults(title=AllMeasured['Region'], ylabel='occupancy', xlim=xlim, ylim = [1e-6,YMax], dims=("District"), Dates=AllMeasured['Dates'], legendPlacement='upper right')
M.showResults(title=AllMeasured['Region'], Scale=PopSum, ylabel='occupancy', xlim=xlim, dims=("District"), Dates=AllMeasured['Dates'], legendPlacement='upper right', styles=['.','-','--'])
# plt.ylim(1e-7*PopSum,1e-3*PopSum)

M.showStates(MinusOne=('S'), dims2d=None, Dates = AllMeasured['Dates'], legendPlacement='upper right')

if measured.shape[-2] > 1:
    M.showResults(title="Age Distribution", ylabel='occupancy', xlim=xlim, dims=("Age"), Dates=AllMeasured['Dates'], legendPlacement='upper right')

# np.sum(measured[-1,:,:,:],(0,2))*PopSum / Pop  # detected per population

# if 'T0' in fittedVars:
#     print("mean(T0) = " + str(np.mean(fittedVars['T0'])))
# print("mean(r0) = " + str(np.mean(fittedVars['r0'])))
# print("h = " + str(fittedVars['h']))
# print("aT0 = " + str(fittedVars['aT0']))
# print("aBase = " + str(fittedVars['aBase']))
# print("d = " + str(fittedVars['d']))
# if 'rd' in fittedVars:
#     print("rd = " + str(fittedVars['rd']))
# if 'q' in fittedVars:
#     print("q = " + str(fittedVars['q']))

M.compareFit(fittedVars=fittedVars)

plotTotalCases(AllMeasured)

if False:
    plt.figure("Awareness reduction")
    plt.plot(awareness(np.arange(0, 100)))
