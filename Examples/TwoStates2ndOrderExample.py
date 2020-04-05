# This example simulates a simply infection growth with saturation and a detection effciency
import StateModeling as stm

M = stm.Model()  # creates a new Model instance
I0 = M.newVariables({'I0': 0.05})  # initially infected
M.newState(name='susceptible', axesInit=1.0 - I0)  # ground state
M.newState(name='infected', axesInit=I0)  # excited state. Systems starts in the excited state
true_r0 = 0.2  # rate of infection
R0 = M.newVariables({'R0': true_r0})  # transition rate
M.addRate(('susceptible', 'infected'), 'infected', R0)  # susc*infec --> infec second order rate
true_D = 0.2  # rate of detection
D = M.newVariables({'D': true_D})
Popul = 100000
M.addResult('detected', lambda State: D * Popul * State['infected'])
M.toFit(['R0', 'I0', 'D'])  # fit the infectin rate and the initial situation
to_distort = {'R0': 1.1, 'I0': 1.2, 'D': 0.2} #,
# simulate data

Tmax = 80
sigma = 300
measured = M.simulate('measured', {'detected': 0}, Tmax=Tmax, applyGaussian=sigma)

# Fit with distorted starting values
M.relDistort(to_distort)
distorted = M.simulate('distorted', {'detected': 0}, Tmax=Tmax)

if True:
    otype = "L-BFGS"
    lossScale = 1e6 # 1e4
    oparam = {"normFac":None}
else:
    lossScale = None
    otype = "adam" # "adagrad"  "adadelta" "SGD" "nesterov"  "adam"
    learnrate = {"nesterov": 1e-9, "adam": 7e-6}
    oparam = {"learning_rate": learnrate[otype]}
fittedVars, fittedRes = M.fit({'detected': measured}, Tmax, otype=otype, oparam=oparam, NIter=150, verbose=True, lossScale=lossScale)

M.compareFit()
M.showResults(ylabel='occupancy')
M.showStates()
