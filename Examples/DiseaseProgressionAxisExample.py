# This example simulates a simply infection process and its progression to hospitalization
# day to start of new transmission: 2.5 days, days to maximal infection: 6 days (source: Leopoldina statement)
# https://www.leopoldina.org/uploads/tx_leopublication/2020_04_03_Leopoldina_Stellungnahme_Gesundheitsrelevante_Ma%C3%9Fnahmen_Corona.pdf
import StateModeling as stm
import tensorflow as tf

M = stm.Model()  # creates a new Model instance
ProgressionTime = 20
M.addAxis("Disease Progression", ProgressionTime, queue=True)

I0 = M.newVariables({"I0": 1.0})  # initially infected
InitProgression = lambda: I0 * M.Axes['Disease Progression'].initDelta()  # variables to fit have to always be packed in lambda functions!
M.newState(name='progression', axesInit={'Disease Progression': InitProgression})  # initially someone infected
M.newState(name='hospitalized', axesInit={'Disease Progression': 0.0})  # just the total in hospital
M.newState(name='recovered', axesInit=0.0)  # the ones that made it through the time series
toHospitalRate = M.newVariables({"toHospitalTime": 7.0, "toHospitalSigma": 3.0, "toHospitalRate": 0.1})  # average time when sent to hospital, sigma and rate
hospitalization = lambda: toHospitalRate * M.Axes['Disease Progression'].initGaussian(M.Var['toHospitalTime'],
                                                                                      M.Var['toHospitalSigma'])
M.addRate('progression', 'hospitalized', hospitalization)  # susc*infec --> infec second order rate
M.addRate('hospitalized', 'recovered', 1.0, queueSrc='Disease Progression')  # When you made it though the queue, you are recovered
M.addRate('hospitalized', 'hospitalized', 0.05, queueSrc='total', queueDst='Disease Progression')  # add some chance to start over
Popul = 100000.0
M.toFit(['I0', 'toHospitalTime', 'toHospitalRate'])  # 'toHospitalTime',  fit the infectin rate and the initial situation
to_distort = {'toHospitalTime': 1.3, 'I0': 1.2, 'toHospitalRate': 0.5}

M.addResult('det. hosp.', lambda State: tf.reduce_sum(Popul * State['hospitalized']))
# simulate data

Tmax = 80
sigma = 300
measured = M.simulate('measured', {'det. hosp.': None}, Tmax=Tmax, applyPoisson=True)

# Fit with distorted starting values
M.relDistort(to_distort)
distorted = M.simulate('distorted', {'det. hosp.': None}, Tmax=Tmax)

if True:
    otype = "L-BFGS"
    lossScale = 1e6  # 1e4
    oparam = {"normFac": None}
else:
    lossScale = None
    otype = "adam"  # "adagrad"  "adadelta" "SGD" "nesterov"  "adam"
    learnrate = {"nesterov": 1e-9, "adam": 7e-6}
    oparam = {"learning_rate": learnrate[otype]}
oparam['noiseModel'] = 'Poisson'

fittedVars, fittedRes = M.fit({'det. hosp.': measured}, Tmax, otype=otype, oparam=oparam, NIter=150, verbose=True, lossScale=lossScale)

M.compareFit()
M.showResults(ylabel='occupancy')
M.showStates()
