# This example is written for the new interface
# This is the full COVID-19 model to be fitted to the RKI data
# see the PPT for details of the model design

import StateModeling as stm
import numpy as np
import matplotlib.pyplot as plt
# import csv
import fetch_data
import pandas as pd
import tensorflow as tf

basePath = r"C:\Users\pi96doc\Documents\Programming\PythonScripts\StateModeling"
useThuringia = True
if useThuringia:
    Region = "Thuringia"
    # Thuringia = pd.read_excel(r"C:\Users\pi96doc\Documents\Anträge\Aktuell\COVID_Dickmann_2020\COVID-19 Linelist 2020_04_06.xlsx")
    Thuringia = pd.read_excel(r"C:\Users\pi96doc\Documents\Anträge\Aktuell\COVID_Dickmann_2020\PetraDickmann\COVID-19 Linelist 2020_04_21.xlsx")
    df = pd.read_excel(basePath + r"\Examples\bev_lk.xlsx")  # support information about the population
    AllMeasured = stm.binThuringia(Thuringia, df)
    Hospitalized = AllMeasured['Hospitalized']
else:
    Region = "Germany"
    if False:
        data = fetch_data.DataFetcher().fetch_german_data()
        # with open(r"C:\Users\pi96doc\Documents\Anträge\Aktuell\COVID_Dickmann_2020\Global_Mobility_Report.csv", 'r', encoding="utf8") as f:
        #     mobility = list(csv.reader(f, delimiter=","))
        # mobility = np.array(mobility[1:], dtype=np.float)

        googleData = pd.read_excel(r"C:\Users\pi96doc\Documents\Anträge\Aktuell\COVID_Dickmann_2020\COVID-19 Linelist 2020_04_06.xlsx")
        data_np = data.to_numpy()
        df = pd.read_excel(basePath + r"\Examples\bev_lk.xlsx")  # support information about the population
        AllMeasured = stm.cumulate(data, df)
        np.save(basePath + r'\Data\AllMeasured', AllMeasured)

        # can be checked with
        # https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Situationsberichte/2020-04-16-de.pdf?__blob=publicationFile
    else:
        AllMeasured = np.load(basePath + r'\Data\AllMeasured.npy', allow_pickle=True).item()
        RawCumulCases = AllMeasured['CumulCases']
        RawCumulDead = AllMeasured['CumulDead']

RawCases = AllMeasured['Cases']
RawDead = AllMeasured['Dead']
RawCured = AllMeasured['Cured']
RawCumulCases = np.cumsum(RawCases, 0)
RawCumulDead = np.cumsum(RawDead, 0)
RawIDs = AllMeasured['IDs']
LKs = AllMeasured['LKs']
RawPopM = AllMeasured['PopM']
RawPopW = AllMeasured['PopW']
Area = AllMeasured['Area']
Ages = AllMeasured['Ages']
Gender = AllMeasured['Gender']
Dates = AllMeasured['Dates']


# only to 11.04.2020
mobility = pd.read_csv(r"C:\Users\pi96doc\Documents\Anträge\Aktuell\COVID_Dickmann_2020\Global_Mobility_Report.csv", low_memory=False)
mobdat = mobility[mobility['sub_region_1'] == "Thuringia"]
mobdate=mobdat['date'].to_numpy()
plt.figure('Retail an recreation');plt.plot(mobdat['retail_and_recreation_percent_change_from_baseline'].to_numpy())
offsetDay=0; plt.xticks(range(offsetDay, len(mobdate), 7), [date for date in mobdate[offsetDay:-1:7]], rotation="vertical")
plt.ylabel('Percent Change')
plt.tight_layout()

# fit,data = stm.DataLoader().get_new_data()
# axes = data.keys()
# datp = data.pivot_table(values=['cases','deaths'], index=['id','day'], aggfunc=np.sum, fill_value=0)
# data_np = datp.to_numpy()
# NumIDs = data['id'].unique().shape
# NumDays = data['day'].unique().shape
CorrectWeekdays = False
if CorrectWeekdays:
    RawCases = stm.correctWeekdayEffect(RawCases)
    RawDead = stm.correctWeekdayEffect(RawDead)

ReduceDistricts = True
ReduceAges = False
SumAges = True
SumDistricts = False
SumGender = True
# ReduceGerman = True
Pop = 1e6 * np.array([(3.88 + 0.78), 6.62, 2.31 + 2.59 + 3.72 + 15.84, 23.9, 15.49, 7.88], stm.CalcFloatStr)
SelectedGender = slice(0, 2)  # remove the "unknown" part
GenderLabels = Gender[SelectedGender]

if ReduceDistricts:
    # DistrictStride = 50
    # SelectedIDs = slice(0,MeasDetected.shape[1],DistrictStride)
    # IDLabels = LKs[SelectedIDs]
    # SelectedIDs = (0, 200, 250, 300, 339, 340, 341, 342)  # LKs.index('SK Jena')
    if useThuringia:
        # SelectedIDs = (12,13,17,19)
        SelectedIDs = (17, 19)
    else:
        SelectedIDs = (352,342,167,332,399, 278, 403, 82, 230, 55, 251, 102, 221, 122, 223, 110, 263, 80, 240, 330, 3, 276)
    # LKs.index('SK Jena'), SK Gera, LK Nordhausen, SK Erfurt, Sk Suhl, LK Weimarer Land, SK Weimar
    # LK Greiz, LK Schmalkalden-Meiningen, LK Eichsfeld, LK Sömmerda, LK Hildburghausen,
    # LK Saale-Orla-Kreis, LK Kyffhäuserkreis, LK Saalfeld-Rudolstadt, LK Ilm-Kreis,
    # LK Unstrut-Hainich-Kreis, LK Gotha, LK Sonneberg, SK Eisenach, LK Altenburger Land, LK Wartburgkreis
    # SelectedIDs = (0, 200)
    IDLabels = [LKs[index] for index in SelectedIDs]
    CumulCases = RawCumulCases[:, SelectedIDs, :, SelectedGender]
    CumulDead = RawCumulDead[:, SelectedIDs, :, SelectedGender]
    Cases = RawCases[:, SelectedIDs, :, SelectedGender]
    Dead = RawDead[:, SelectedIDs, :, SelectedGender]
    Hospitalized = Hospitalized[:, SelectedIDs, :, SelectedGender]
    Cured = RawCured[:, SelectedIDs, :, SelectedGender]
    PopM = [RawPopM[index] for index in SelectedIDs]  # PopM[0:-1:DistrictStride]
    PopW = [RawPopW[index] for index in SelectedIDs]  # PopW[0:-1:DistrictStride]
    IDs = [RawIDs[index] for index in SelectedIDs]  # IDs[0:-1:DistrictStride]
else:
    IDLabels = LKs
    CumulCases = RawCumulCases
    CumulDead = RawCumulDead
    Cases = RawCases
    Dead = RawDead
    Cured = RawCured
    AgeLabels = Ages
    GenderLabels = Gender
    PopM = RawPopM
    PopW = RawPopW
    IDs = RawIDs

if ReduceAges:
    SelectedAges = slice(0, RawCumulCases.shape[2] - 1)  # remove the "unknown" part
    AgeLabels = Ages[SelectedAges]
    CumulCases = CumulCases[:, :, SelectedAges, :]
    CumulDead = CumulDead[:, :, SelectedAges, :]
    Cases = Cases[:, :, SelectedAges, :]
    Dead = Dead[:, :, SelectedAges, :]
    Cured = Cured[:, :, SelectedAges, :]

if SumAges:
    AgeLabels = ['summed Ages']
    CumulCases = np.sum(CumulCases, 2, keepdims=True)
    CumulDead = np.sum(CumulDead, 2, keepdims=True)
    Cases = np.sum(Cases, 2, keepdims=True)
    Dead = np.sum(Dead, 2, keepdims=True)
    Hospitalized= np.sum(Hospitalized, 2, keepdims=True)
    Pop = np.sum(Pop)

if SumDistricts:
    IDLabels = [Region]
    CumulCases = np.sum(CumulCases, 1, keepdims=True)
    CumulDead = np.sum(CumulDead, 1, keepdims=True)
    Cases = np.sum(Cases, 1, keepdims=True)
    Dead = np.sum(Dead, 1, keepdims=True)
    Hospitalized= np.sum(Hospitalized, 1, keepdims=True)

if SumGender:
    GenderLabels = ['Both Genders']
    CumulCases = np.sum(CumulCases, 3, keepdims=True)
    CumulDead = np.sum(CumulDead, 3, keepdims=True)
    Cases = np.sum(Cases, 3, keepdims=True)
    Dead = np.sum(Dead, 3, keepdims=True)
    Hospitalized= np.sum(Hospitalized, 3, keepdims=True)
Tmax = 120

M = stm.Model()
M.addAxis("Gender", entries=len(GenderLabels), labels=GenderLabels)
M.addAxis("Age", entries=len(AgeLabels), labels=AgeLabels)
M.addAxis("District", entries=len(IDLabels), labels=IDLabels)
M.addAxis("Disease Progression", entries=20, queue=True)
M.addAxis("Quarantine", entries=24, queue=True)

AgeDist = (Pop / np.sum(Pop))

InitAge = M.Axes['Age'].init(AgeDist)

PopSum = np.sum(PopM) + np.sum(PopW)

if Cases.shape[3] > 1:
    InitPopulM = M.Axes['District'].init(PopM / PopSum)
    InitPopulW = M.Axes['District'].init(PopW / PopSum)
    InitPopul = InitAge * np.concatenate((InitPopulM, InitPopulW), -1)
else:
    InitPopul = InitAge * M.Axes['District'].init(1.0)

# InitGender = [MRatio, 1 - MRatio]
# MRatio = np.sum(PopM) / PopSum

# susceptible
M.newState(name='S', axesInit={"Age": 1.0, "District": InitPopul, "Gender": 1.0})
# I0 = M.newVariables({'I0': 0.000055 * InitPopulM}, forcePos=False)  # a district dependent variable of initially infected
# assume 4.0 infected at time 0
#  (2.0/323299.0) * InitPopul
I0Start = 5e-7 # 3.0e-7
I0 = M.newVariables({'I0': I0Start}, forcePos=False)  # a global variable of initial infection probability
# InitProgression = lambda: I0 * M.Axes['Disease Progression'].initDelta()  # variables to fit have to always be packed in lambda functions!
# M.newState(name='I', axesInit={"Disease Progression": InitProgression, "District": None, "Age": None, "Gender": None})
# infected (not detected):
M.newState(name='I', axesInit={"Disease Progression": 0, "District": 0, "Age": 0, "Gender": 0})
# cured (not detected):
M.newState(name='C', axesInit={"District": 0, "Age": 0, "Gender": 0})
T0Start = Dates.index('14.02.2020')+0.2   #14.0
T0 = M.newVariables({"T0": T0Start * np.ones(M.Axes['District'].shape, stm.CalcFloatStr)}, forcePos=False)  # time at which a delta is injected into the start of the progression axis
# the initial infection is generated by "injecting" a Gaussian (to be differentiable)
M.addRate('S', 'I', lambda t: I0() * M.initGaussianT0(T0(), t), queueDst='Disease Progression', hasTime=True)
# Age-dependent base rate of infection
r0 = M.newVariables({'r0': 2.5 * np.ones(M.Axes['Age'].shape, stm.CalcFloatStr)}, forcePos=True)
aT0Start = Dates.index('22.03.2020')+0.0 # 65.0
aT0 = M.newVariables({'aT0': aT0Start}, forcePos=True)  # awarenessTime
aSigma = 4.0
aBase = M.newVariables({'aBase': 0.54}, forcePos=True)  # residual relative rate after awareness effect
awareness = lambda t: M.initSigmoidDropT0(aT0(), t, aSigma, aBase())  # 40% drop in infection rate
it0 = M.newVariables({'it0': 3.5}, forcePos=True) # day of most probably infection
sigmaI = 3.0
infectiveness = M.Axes['Disease Progression'].initGaussian(it0(), sigmaI)
InitPupulDistrictOnly = np.sum(InitPopul,(-1,-2), keepdims=True)
M.addRate(('S', 'I'), 'I', lambda t: (r0()* awareness(t) * infectiveness) / InitPupulDistrictOnly,
          queueDst="Disease Progression", hasTime=True, hoSumDims=['Age', 'Gender'])  # S ==> I[0]
M.addRate('I', 'C', 1.0, queueSrc="Disease Progression")  # I --> C when through the queue

# --- The (undetected) quarantine process:
# susceptible, quarantined
M.newState(name='Sq', axesInit={"Age": 0, "Quarantine": 0, "District": 0, "Gender": 0})
# infected, quarantined
M.newState(name='Iq', axesInit={"Age": 0, "Disease Progression": 0, "District": 0, "Gender": 0})
q = M.newVariables({"q": 0.7}, forcePos=True)  # quarantine ratio (of all ppl.) modeled as a Gaussian (see below)
LockDown = Dates.index('20.03.2020')+0.0  # This is when retail changed! 79
sigmaQ = 1.5
lockDownFct = lambda t: q() * M.initGaussianT0(LockDown, t, sigmaQ)
M.addRate('S', 'Sq', lockDownFct, queueDst='Quarantine', hasTime=True)
# M.addRate('S', 'Sq', q, queueDst="Quarantine")  # S -q-> Sq
M.addRate('Sq', 'S', 1.0, queueSrc="Quarantine")  # Sq --> S
M.addRate('I', 'Iq', lockDownFct, hasTime=True)  # S -q-> Sq
M.addRate('Iq', 'C', 1.0, queueSrc="Disease Progression")  # Iq --> C when through the infection queue. Quarantine does not matter any more
# ---------- detecting some of the infected:
# detected quarantine state:
M.newState(name='Q', axesInit={"Age": 0, "Disease Progression": 0, "District": 0, "Gender": 0})  # no quarantine axis is needed, since the desease progression takes care of this
d = M.newVariables({'d': 0.027}, forcePos=True)  # detection rate
M.addRate('I', 'Q', d, resultTransfer=('cases','Disease Progression'))  # S -q-> Sq
M.addRate('Iq', 'Q', d, resultTransfer=('cases','Disease Progression'))  # detection by testing inside the quarantine
# ---- hospitalizing the ill
# hospitalized state:
M.newState(name='H', axesInit={"Disease Progression": 0, "District": 0, "Age": 0, "Gender": 0})
ht0 = M.newVariables({'ht0': 5.5}, forcePos=False)  # most probable time of hospitalization
h = M.newVariables({'h': 0.08})  # rate of hospitalization, should be age dependent
# influx = M.newVariables({'influx': 0.0001})  # a district dependent variable of initially infected
# infectionRate = lambda I: (I + influx) * M.Var['r0']
AgeBorder = M.newVariables({'AgeBorder': 2.3}, forcePos=False, normalize=None)  # rate of hospitalization, should be age dependent
AgeSigma = M.newVariables({'AgeSigma': 0.5}, forcePos=False, normalize=None)  # rate of hospitalization, should be age dependent
hospitalization = lambda: h() * M.Axes['Disease Progression'].initGaussian(ht0(), 3.0) * \
                          M.Axes['Age'].initSigmoid(AgeBorder(), AgeSigma())
M.addRate('I', 'H', hospitalization, resultTransfer=(('cases', 'Disease Progression'),('hospitalization', 'Disease Progression')))  # I[t] -> H[t]
M.addRate('Q', 'H', hospitalization, resultTransfer=('hospitalization', 'Disease Progression'))  # Q[t] -> H[t]
M.addRate('Iq', 'H', hospitalization, resultTransfer=(('cases', 'Disease Progression'),('hospitalization', 'Disease Progression')))  # Iq[t] -> H[t]
# cured (detected):
M.newState(name='CR', axesInit={"District": 0, "Age": 0, "Gender": 0})
M.addRate('H', 'CR', 1.0, queueSrc="Disease Progression")  # H[t] -> CR[t]  this is a dequeueing operation and thus the rate needs to be one!
M.addRate('Q', 'CR', 1.0, queueSrc="Disease Progression")  # H[t] -> R[t]  this is a dequeueing operation and thus the rate needs to be one!
# ---- intensive care:
# in intensive care
M.newState(name='HIC', axesInit={"Disease Progression": 0, "District": 0, "Age": 0, "Gender": 0})
# dead
M.newState(name='D', axesInit={"District": 0, "Age": 0, "Gender": 0})
hic = 0.05  # should be age dependent
M.addRate('H', 'HIC', hic)
M.addRate('HIC', 'H', 1.0, queueSrc="Disease Progression")  # HIC[t] -> H[t] If intensive care was survived, start over in hospital
# rate to die from intensive care:
r = 0.05  # should be age dependent
M.addRate('HIC', 'D', r, resultTransfer='deaths')

# cumulative total detected (= measured) cases:
M.addResult('cumul_cases', lambda State: tf.reduce_sum(State['H'], 1) + tf.reduce_sum(State['Q'], 1, keepdims=True) + tf.reduce_sum(State['HIC'], 1, keepdims=True) + State['CR'] + State['D'])  # ('I', 'S')
M.addResult('cumul_dead', lambda State: State['D'])  # ('I', 'S')

# M.toFit(['r0', 'hr', 'ht0', 'I0'])
# M.toFit(['r0', 'I0'])
M.toFit(['T0', 'r0', 'h', 'aT0', 'aBase', 'I0', 'd'])  # 'q',
# M.toFit(['r0'])

# if Cases.shape[-1] > 1:
#     M.toFit.append()
if Cases.shape[-2] > 1:
    M.toFit.append(['Age Border', 'Age Sigma'])

simulated = M.simulate('simulated', {'cases': None}, Tmax=Tmax)
# M.showResults(ylabel='occupancy')
# M.showStates(MinusOne=('S'))

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

measured = Cases[:,  np.newaxis, np.newaxis, :, :, :] / PopSum
measuredDead = Dead[:, np.newaxis, np.newaxis, :, :, :] / PopSum
NIter = 200 # 200

# tf.config.experimental_run_functions_eagerly(True)

xlim = None  # (60,100)
# fittedVars, fittedRes = M.fit({'detected': measured}, Tmax, otype=otype, oparam=oparam, NIter=NIter, verbose=True, lossScale=lossScale)
FitDict = {'cases': measured}
if "Hospitalized" in AllMeasured.keys():
    FitDict['hospitalization'] = Hospitalized[:, np.newaxis, np.newaxis, :, :, :]/ PopSum
FitDict['deaths'] = measuredDead

fittedVars, fittedRes = M.fit(FitDict, Tmax, otype=otype, oparam=oparam, NIter=NIter, verbose=True, lossScale=lossScale)
if measured.shape[-3] > 1:
    M.showResults(title="District Distribution", ylabel='occupancy', xlim=xlim, dims=("District"), Dates=Dates)
else:
    M.showResults(title=Region, ylabel='occupancy', xlim=xlim, dims=("District"), Dates=Dates)
    # M.showResults(title=Region+"_2", ylabel='occupancy', xlim=xlim, dims=("District"))
# M.showStates(MinusOne=('S'), dims2d=("time", "District"))
M.showStates(MinusOne=('S'), dims2d=None, Dates = Dates)

if measured.shape[-2] > 1:
    M.showResults(title="Age Distribution", ylabel='occupancy', xlim=xlim, dims=("Age"), Dates=Dates)

# np.sum(measured[-1,:,:,:],(0,2))*PopSum / Pop  # detected per population
plt.figure('hospitalization')
toPlot = np.squeeze(hospitalization())
if toPlot.ndim > 1:
    plt.imshow(toPlot)
else:
    plt.plot(toPlot)

print("mean(T0) = " + str(np.mean(fittedVars['T0'])))
print("mean(r0) = " + str(np.mean(fittedVars['r0'])))
print("h = " + str(fittedVars['h']))
print("aT0 = " + str(fittedVars['aT0']))
print("aBase = " + str(fittedVars['aBase']))
print("d = " + str(fittedVars['d']))
print("q = " + str(fittedVars['q']))

plt.figure("Neuinfektionen")
#plt.plot((np.sum(RawCumulCases[1:, :, :, :], (1, 2, 3)) - np.sum(RawCumulCases[0:-1, :, :, :], (1, 2, 3))) / np.sum(RawPopM + RawPopW) * 100000)
factor = 1.0
if False:
    factor = 100000.0 / np.sum(RawPopM + RawPopW)
    plt.ylabel("Cases / 100.000 und Tag")
else:
    plt.ylabel("Cases")
plt.plot(factor*np.sum(RawCases[1:, :, :, :], (1, 2, 3)))
plt.plot(factor*np.sum(10.0 * RawDead[1:, :, :, :], (1, 2, 3)))
if "Hospitalized" in AllMeasured.keys():
    plt.plot(factor * np.sum(Hospitalized[1:, :, :, :], (1, 2, 3)))
    plt.legend(('New Infections', 'Deaths (*10)', 'Hospitalized'))
else:
    plt.legend(('New Infections', 'Deaths (*10)'))
plt.xlabel("Tag")
offsetDay = 0  # being sunday
plt.xticks(range(offsetDay, len(Dates), 7), [date for date in Dates[offsetDay:-1:7]], rotation="vertical")
# plt.xlim(45, len(Dates))
plt.tight_layout()
plt.hlines(0.25, 0, len(Dates), linestyles="dashed")
# plt.vlines(11*7, 0, 5, linestyles="dashed")

plt.figure("Awareness reduction")
plt.plot(awareness(np.arange(0, 100)))

plt.figure("All_" + Region)
plt.semilogy(np.sum(RawCumulCases, (1, 2, 3)), 'g')
plt.semilogy(np.sum(RawCumulDead, (1, 2, 3)), 'm')
plt.semilogy(np.sum(RawCases, (1, 2, 3)), 'g.-')
plt.semilogy(np.sum(RawDead, (1, 2, 3)), 'm.-')
plt.semilogy(np.sum(RawCured, (1, 2, 3)), 'b')
if "Hospitalized" in AllMeasured.keys():
    plt.semilogy(np.sum(Hospitalized, (1, 2, 3)))
    plt.legend(['CumulCases', 'CumulDead', 'Cases', 'Deaths', 'Cured', 'Hospitalized'])
else:
    plt.legend(['CumulCases', 'CumulDead', 'Cases', 'Deaths', 'Cured'])
offsetDay = 0  # being sunday
plt.xticks(range(offsetDay, len(Dates), 7), [date for date in Dates[offsetDay:-1:7]], rotation="vertical")
# plt.xlim(45, len(Dates))
plt.tight_layout()
