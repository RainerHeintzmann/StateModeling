# This example is written for the new interface
import StateModeling as stm
import numpy as np
import matplotlib.pyplot as plt
import fetch_data
import pandas as pd
import tensorflow as tf

basePath = r"C:\Users\pi96doc\Documents\Programming\PythonScripts\StateModeling"
if False:
    data = fetch_data.DataFetcher().fetch_german_data()
    data_np = data.to_numpy()
    df = pd.read_excel(basePath + r"\Examples\bev_lk.xlsx")  # support information about the population
    MeasDetected, MeasDead, SupportingInfo = stm.cumulate(data, df)
    np.save(basePath + r'\Data\MeasDetected', MeasDetected)
    np.save(basePath + r'\Data\MeasDead', MeasDead)
    np.save(basePath + r'\Data\SupportingInfo', SupportingInfo)
else:
    MeasDetected = np.load(basePath + r'\Data\MeasDetected.npy')
    MeasDead = np.load(basePath + r'\Data\MeasDead.npy')
    SupportingInfo = np.load(basePath + r'\Data\SupportingInfo.npy', allow_pickle=True)
(IDs, LKs, PopM, PopW, Area, Ages, Gender) = SupportingInfo

# fit,data = stm.DataLoader().get_new_data()
# axes = data.keys()
# datp = data.pivot_table(values=['cases','deaths'], index=['id','day'], aggfunc=np.sum, fill_value=0)
# data_np = datp.to_numpy()
# NumIDs = data['id'].unique().shape
# NumDays = data['day'].unique().shape

ReduceDistricts = True
if ReduceDistricts:
    DistrictStride = 50
    MeasDetected = MeasDetected[:, 0:-1:DistrictStride, :, :]
    PopM = PopM[0:-1:DistrictStride]
    PopW = PopW[0:-1:DistrictStride]
    IDs = IDs[0:-1:DistrictStride]
Tmax = 120

M = stm.Model()
M.addAxis("Gender", entries=len(Gender) - 1)
M.addAxis("Age", entries=len(Ages))
M.addAxis("District", entries=len(IDs))
M.addAxis("Disease Progression", entries=20, queue=True)

Pop = 1e6 * np.array([(3.88 + 0.78), 6.62, 2.31 + 2.59 + 3.72 + 15.84, 23.9, 15.49, 7.88, 1.0], stm.CalcFloatStr)
AgeDist = (Pop / np.sum(Pop))

InitAge = M.Axes['Age'].init(AgeDist)

PopSum = np.sum(PopM) + np.sum(PopW)

InitPopulM = M.Axes['District'].init(PopM / PopSum)
InitPopulW = M.Axes['District'].init(PopW / PopSum)
InitPopul = InitPopulM + InitPopulW
MRatio = np.sum(PopM) / PopSum
M.newState(name='S', axesInit={"Age": InitAge, "District": InitPopul, "Gender": [MRatio, 1 - MRatio]})
I0 = M.newVariables({'I0': 0.000055 * InitPopulM})  # a district dependent variable of initially infected
InitProgression = lambda: I0 * M.Axes['Disease Progression'].initDelta()  # variables to fit have to always be packed in lambda functions!
M.newState(name='I', axesInit={"Disease Progression": InitProgression, "District": None, "Age": None, "Gender": None})
M.newState(name='H', axesInit={"Disease Progression": 0, "District": 0, "Age": 0, "Gender": 0})
M.newState(name='R', axesInit={"District": 0, "Age": 0, "Gender": 0})  # undetected recovered
M.newState(name='Rd', axesInit={"District": 0, "Age": 0, "Gender": 0})  # detected recovered
ht0 = M.newVariables({'ht0': 3.0})
hr = M.newVariables({'hr': 0.02})  # rate of hospitalization
hospitalization = lambda: hr * M.Axes['Disease Progression'].initGaussian(ht0, 3.0)
influx = M.newVariables({'influx': 0.0001})  # a district dependent variable of initially infected
# infectionRate = lambda I: (I + influx) * M.Var['r0']
r0 = M.newVariables({'r0': 0.11 / InitPopul})
infectionRate = lambda: M.Var['r0']
M.addRate(('S', 'I'), 'I', infectionRate, queueDst="Disease Progression")  # S ==> I[0]
M.addRate('I', 'H', hospitalization)  # I[t] -> H[t]
M.addRate('H', 'Rd', 1.0, queueSrc="Disease Progression")  # H[t] -> R[t]  this is a dequeuing operation and thus the rate needs to be one!
M.addRate('I', 'R', 1.0, queueSrc="Disease Progression")  # H[t] -> R[t]  this is a dequeuing operation and thus the rate needs to be one!
M.addResult('detected', lambda State: tf.reduce_sum(State['H'], 1) + State['Rd'])  # ('I', 'S')

# M.toFit(['r0', 'hr', 'ht0', 'I0'])
M.toFit(['r0', 'I0'])

# simulated = M.simulate('simulated', {'detected': None}, Tmax=Tmax)
# M.showResults(ylabel='occupancy')
# M.showStates(MinusOne=('S'))

if True:
    otype = "L-BFGS"
    lossScale = 1  # 1e4
    oparam = {"normFac": 'max'}
else:
    # ToDo the local normFac is not yet recognized for the below methods
    lossScale = None
    otype = "nesterov"  # "adagrad"  "adadelta" "SGD" "nesterov"  "adam"
    learnrate = {"nesterov": 1e-10, "adam": 7e-7}
    oparam = {"learning_rate": learnrate[otype]}
# oparam['noiseModel'] = 'Poisson'
oparam['noiseModel'] = 'Gaussian'
# oparam['noiseModel'] = 'ScaledGaussian'  # is buggy?

NIter = 150
fittedVars, fittedRes = M.fit({'detected': MeasDetected[:, np.newaxis, :, :, 0:1] / PopSum}, Tmax, otype=otype, oparam=oparam, NIter=NIter, verbose=True, lossScale=lossScale)
M.showResults(ylabel='occupancy', dims=("District"))
M.showStates(MinusOne=('S'))
