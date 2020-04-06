# A simple two state example, but with several separate decay rates and starting values, which are fitted individually
import StateModeling as stm
import numpy as np

NumMolecules = 16
M = stm.Model()  # creates a new Model instance
M.addAxis("Different Molecules", NumMolecules)

sigma = 0.2
true_I0 = 1200 * np.random.normal(np.ones(NumMolecules), sigma)
I0 = M.newVariables({'I0': true_I0})
k = M.newVariables({'k': 0.02 * 1. / np.random.normal(np.ones(NumMolecules), sigma)})
M.newState(name='S0', axesInit={'Different Molecules': M.Axes['Different Molecules'].init(0)})  # ground state
M.newState(name='S1', axesInit={'Different Molecules': M.Axes['Different Molecules'].init(I0)})  # excited state. Systems starts in the excited state

M.addRate('S1', 'S0', 'k')  # S1 --> S0  first order decay leading to a single exponential decay
M.addResult('detected', lambda State: State['S1'])  # ('I', 'S')
M.toFit(['k', 'I0'])  # fitting S1 works, but not fitting I0 !
# M.toFit(['k'])

# simulate data

Tmax = 80
measured = M.simulate('measured', {'detected': None}, Tmax=Tmax, applyPoisson=True)

# Fit with distorted starting values
M.relDistort({'k': 0.8, 'I0': 1.2})
distorted = M.simulate('distorted', {'detected': None}, Tmax=Tmax)

oparam={'noiseModel': 'Gaussian', "normFac": "max"} #
if True:
    otype = "L-BFGS"
    lossScale = None
else:
#    lossScale = None
    otype = "adagrad"  # "adadelta" "SGD" "nesterov"  "adam"

fittedVars, fittedRes = M.fit({'detected': measured}, Tmax, otype=otype, oparam=oparam, NIter=150, verbose=True, lossScale=lossScale)

M.compareFit()
M.showResults(ylabel='Intensity')
M.showStates()
