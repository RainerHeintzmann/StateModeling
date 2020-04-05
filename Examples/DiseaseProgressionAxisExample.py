# This example simulates a simply infection process and its progression to hospitalization
# day to start of new transmission: 2.5 days, days to maximal infection: 6 days (source: Leopoldina statement)
# https://www.leopoldina.org/uploads/tx_leopublication/2020_04_03_Leopoldina_Stellungnahme_Gesundheitsrelevante_Ma%C3%9Fnahmen_Corona.pdf
import StateModeling as stm

M = stm.Model()  # creates a new Model instance
ProgressionTime=20
M.addAxis("Disease Progression", ProgressionTime, queue=True)

I0 = M.newVariables({"I0": 1.0})  # initially infected
InitProgression = lambda: I0 * M.Axes['Disease Progression'].initDelta()
M.newState(name='progression', axesInit={'Disease Progression': InitProgression})  # initially someone infected
M.newState(name='hospitalized', axesInit=0.0)  # just the total in hospital
toHospitalTime = M.newVariables({"toHospitalTime": 7.0})  # average time when sent to hospital
toHospitalSigma = M.newVariables({"toHospitalSigma": 3.0})  # average time when sent to hospital
toHospitalRate = M.newVariables({"toHospitalRate": 0.1})  # average time when sent to hospital
hospitalization = lambda: toHospitalRate * M.Axes['Disease Progression'].initGaussian(toHospitalTime, toHospitalSigma)

M.addRate('progression', 'hospitalized', hospitalization)  # susc*infec --> infec second order rate
Popul = 100000.0
M.toFit(['I0', 'toHospitalTime'])  # 'toHospitalTime',  fit the infectin rate and the initial situation
to_distort = {'toHospitalTime': 1.3, 'I0': 1.2}

M.addResult('det. hosp.', lambda State: Popul * State['hospitalized'])
# simulate data

Tmax = 80
sigma = 300
measured = M.simulate('measured', {'det. hosp.': 0}, Tmax=Tmax, applyGaussian=sigma)

# Fit with distorted starting values
M.relDistort(to_distort)
distorted = M.simulate('distorted', {'det. hosp.': 0}, Tmax=Tmax)

if True:
    otype = "L-BFGS"
    lossScale = 1e6 # 1e4
    oparam = {"normFac":None}
else:
    lossScale = None
    otype = "adam" # "adagrad"  "adadelta" "SGD" "nesterov"  "adam"
    learnrate = {"nesterov": 1e-9, "adam": 7e-6}
    oparam = {"learning_rate": learnrate[otype]}
fittedVars, fittedRes = M.fit({'det. hosp.': measured}, Tmax, otype=otype, oparam=oparam, NIter=150, verbose=True, lossScale=lossScale)

M.compareFit()
M.showResults(ylabel='occupancy')
M.showStates()
