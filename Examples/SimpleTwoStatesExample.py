# This example is written for the new interface
import StateModeling as stm

NumMolecules = 4
M = stm.Model()  # creates a new Model instance
M.newState(name='S0', axesInit=0.0)  # ground state
S1 = M.newVariables({'S1': 1.0}, forcePos=True, normalize=True)  # transition rate
M.newState(name='S1', axesInit=S1)  # excited state. Systems starts in the excited state
true_k = 0.135
true_I0 = 1200.0 / true_k
M.newVariables({'k': true_k}, forcePos=True)  # transition rate
I0 = M.newVariables({'I0': true_I0})  # transition rate
M.addRate('S1', 'S0', 'k', resultTransfer='emission')  # S1 --> S0  first order decay leading to a single exponential decay

M.addResult('detected', lambda State: I0() * M.Var['k']() * State['S1'])  # ('I', 'S')
M.toFit(['k', 'I0'])
# M.toFit(['k'])

# simulate data

Tmax = 80
measured = M.simulate('measured', {'detected': None}, Tmax=Tmax, applyPoisson=True)

# Fit with distorted starting values
M.relDistort({'k': 0.8, 'I0': 1.8})
distorted = M.simulate('distorted', {'detected': None}, Tmax=Tmax)

if True:
    otype = "L-BFGS"
else:
    otype = "adagrad"  # "adadelta" "SGD" "nesterov"  "adam"
fittedVars, fittedRes = M.fit({'detected': measured}, Tmax, otype=otype, NIter=150, verbose=True)

M.showResults(ylabel='Intensity', logY=False)
M.showStates()
M.compareFit()
