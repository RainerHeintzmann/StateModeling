import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.core.protobuf import config_pb2
import numpy as np
import os
# from fit_model import load_data
import matplotlib.pyplot as plt
import time
import numbers
import pandas as pd
import tf_keras_tfp_lbfgs as funfac
from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timedelta


class DataLoader(object):
    def __init__(self):
        load_dotenv()

    def pull_data(self, uri='http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/daily_data'):
        return requests.get(uri).json()

    #        return requests.get('http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/daily_data').json()
    def get_new_data(self):
        uri = "http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/data"
        json_data = self.pull_data(uri)
        table = np.array(json_data["rows"])
        column_names = []
        for x in json_data["fields"]:
            column_names.append(x["name"])
        df = pd.DataFrame(table, columns=column_names)
        df["day"] = [datetime.fromtimestamp(x["$date"] / 1000) for x in df["day"].values]
        df["id"] = df["latitude"].apply(lambda x: str(x)) + "_" + df["longitude"].apply(lambda x: str(x))
        unique_ids = df["id"].unique()
        regions = {}
        for x in unique_ids:
            regions[x] = {}
            regions[x]["data_fit"] = df[df["id"] == x]
        return regions, df


NumberTypes = (int, float, complex, np.ndarray, np.generic)

# The aim is to build a SEIR (Susceptible → Exposed → Infected → Removed)
# Model with a number of (fittable) parameters which may even vary from
# district to district
# The basic model is taken from the webpage
# https://gabgoh.github.io/COVID/index.html
# and the implementation is done in Tensorflow 1.3
# The temporal dimension is treated by unrolling the loop

CalcFloatStr = 'float32'
if False:
    defaultLossDataType = "float64"
else:
    defaultLossDataType = "float32"
defaultTFDataType = "float32"
defaultTFCpxDataType = "complex64"

def addDicts(dict1, dict2):
    """Merge dictionaries and keep values of common keys in list"""
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            val2 = dict1[key]
            if equalShape(value.shape, val2.shape):
                dict3[key] = value + val2
            else:
                print('Shape 1: '+str(value.shape)+", shape 2:"+str(val2.shape))
                raise ValueError('Shapes of transfer values to add are not the same')
    return dict3

def Init(noCuda=False):
    """
    initializes the tensorflow system
    """
    if noCuda is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    tf.compat.v1.reset_default_graph()  # currently just to shield tensorflow from the main program


# Init()
### tf.compat.v1.disable_eager_execution()
# sess = tf.compat.v1.Session()


# tf.device("/gpu:0")


# Here some code from the inverse Modeling Toolbox (Rainer Heintzmann)

def iterativeOptimizer(myTFOptimization, NIter, loss, verbose=False):
    if NIter <= 0:
        raise ValueError("NIter has to be positive")
    for n in range(NIter):
        myTFOptimization()  # summary?
        myloss = loss().numpy()
        if np.isnan(myloss):
            raise ValueError("Loss is NaN. Aborting iteration.")
        if verbose:
            print(str(n) + "/" + str(NIter) + ": " + str(myloss))
    return myloss  # , summary


def optimizer(loss, otype='L-BFGS-B', NIter=300, oparam={'gtol': 0, 'learning_rate': None}, var_list=None, verbose=False):
    """
    defines an optimizer to be used with "Optimize"
    This function combines various optimizers from tensorflow and SciPy (with tensorflow compatibility)

    Parameters
    ----------
    loss : the loss function, which is a tensor that has been initialized but contains variables
    otype (default: L-BFGS : The method of optimization to be used the following options exist:
        from Tensorflow:
            sgrad
            nesterov
            adadelta
            adam
            proxgrad
        and from SciPy all the optimizers in the package tf.contrib.opt.ScipyOptimizerInterface
    NIter (default: 300) : Number of iterations to be used
    oparam : a dictionary to be passed to the detailed optimizers containing optimization parameters (e.g. "learning-rate"). See the individual documentation
    var_list (default: None meaning all) : list of tensorflow variables to be used during minimization
    verbose (default: False) : prints the loss during iterations if True

    Returns
    -------
    an optimizer funtion (or lambda function)

    See also
    -------

    Example
    -------
    """
    if NIter < 0:
        raise ValueError("NIter has to be positive or zero")

    optimStep = 0
    if (var_list is not None) and not np.iterable(var_list):
        var_list = [var_list]
    # these optimizer types work strictly stepwise
    elif otype == 'SGD':
        learning_rate = oparam["learning_rate"]
        if learning_rate == None:
            learning_rate = 0.00003
        print("setting up sgrad optimization with ", NIter, " iterations.")
        optimStep = lambda loss: tf.keras.optimizers.SGD(learning_rate).minimize(loss, var_list=var_list)  # 1.0
    elif otype == 'nesterov':
        learning_rate = oparam["learning_rate"]
        if learning_rate == None:
            learning_rate = 0.00002
        print("setting up nesterov optimization with ", NIter, " iterations.")
        optimStep = lambda loss: tf.keras.optimizers.SGD(learning_rate, nesterov=True, momentum=1e-4).minimize(loss, var_list=var_list)  # 1.0
    elif otype == 'adam':
        learning_rate = oparam["learning_rate"]
        if learning_rate == None:
            learning_rate = 0.0013
        print("setting up adam optimization with ", NIter, " iterations, learning_rate: ", learning_rate, ".")
        optimStep = lambda loss: tf.keras.optimizers.Adam(learning_rate, 0.9, 0.999).minimize(loss, var_list=var_list)  # 1.0
    elif otype == 'adadelta':
        learning_rate = oparam["learning_rate"]
        if learning_rate == None:
            learning_rate = 0.0005
        print("setting up adadelta optimization with ", NIter, " iterations.")
        optimStep = lambda loss: tf.keras.optimizers.Adadelta(learning_rate, 0.9, 0.999).minimize(loss, var_list=var_list)  # 1.0
    elif otype == 'adagrad':
        learning_rate = oparam["learning_rate"]
        if learning_rate == None:
            learning_rate = 0.0012
        print("setting up adagrad optimization with ", NIter, " iterations.")
        optimStep = lambda loss: tf.keras.optimizers.Adagrad(learning_rate).minimize(loss, var_list=var_list)  # 1.0
    if optimStep != 0:
        myoptim = lambda: optimStep(loss)
        myOptimizer = lambda: iterativeOptimizer(myoptim, NIter, loss, verbose=verbose)
    # these optimizers perform the whole iteration
    elif otype == 'L-BFGS':
        # normFac = None
        # if "normFac" in oparam:  # "max", "mean" or None
        #     normFac = oparam["normFac"]
        func = funfac.function_factory(loss, var_list)  # normFactors=normFac
        # convert initial model parameters to a 1D tf.Tensor
        init_params = func.initParams()  # retrieve the (normalized) initialization parameters
        # use the L-BFGS solver
        myOptimizer = lambda: LBFGSWrapper(func, init_params, NIter)
        # myOptimizer = lambda: tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func,
        #                                                    initial_position=init_params,
        #                                                    tolerance=1e-8,
        #                                                    max_iterations=NIter)
        # # f_relative_tolerance = 1e-6,
    else:
        raise ValueError('Unknown optimizer: ' + otype)

    return myOptimizer  # either an iterative one or 'L-BFGS'


def LBFGSWrapper(func, init_params, NIter):
    optim_results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func,
                                                 initial_position=init_params,
                                                 tolerance=1e-6,
                                                 num_correction_pairs=5,
                                                 max_iterations=NIter)
    # f_relative_tolerance = 1e-6
    # converged, failed,  num_objective_evaluations, final_loss, final_gradient, position_deltas,  gradient_deltas
    if not optim_results.converged:
        tf.print("WARNING: optimization did not converge")
    if optim_results.failed:
        tf.print("WARNING: lines search failed during iterations")
    res = optim_results.position
    func.assign_new_model_parameters(res)
    return optim_results.objective_value


def doNormalize(val, normalize, reference):
    if normalize == "max":
        val = val * tf.reduce_max(reference)
    elif normalize == "mean":
        val = val * tf.reduce_mean(reference)
    return val


def invNormalize(val, normalize, reference):
    if normalize == "max":
        val = val / tf.reduce_max(reference)
    elif normalize == "mean":
        val = val / tf.reduce_mean(reference)
    return val


@tf.custom_gradient
def monotonicPos(val, b2=1.0):
    """
    applies a monotonic transform mapping the full real axis to the positive half space

    This can be used to implicitely force the reconstruction results to be all-positive. The monotonic function is derived from a hyperboloid:

    The function is continues and differentiable.
    This function can also be used as an activation function for neural networks.

    Parameters
    ----------
    val : tensorflow array
        The array to be transformed

    Returns
    -------
    tensorflow array
        The transformed array

    Example
    -------
    """
    mysqrt = tf.sqrt(b2 + tf.square(val) / 4.0)

    def grad(dy):
        return dy * (0.5 + val / mysqrt / 4.0), None  # no abs here!

    #    return mysqrt + val / 2.0, grad   # This is the original simple equation, but it is numerically very unstable for small numbers!
    # slightly better but not good:
    #    return val * (0.5 + tf.sign(val) * tf.sqrt(b2/tf.square(val)+0.25)), grad
    taylor1 = b2 / (2.0 * mysqrt)
    diff = val / 2.0 + mysqrt  # for negative values this is a difference
    # print('diff: ' + str(diff)+", val"+str(val)+" taylor:"+str(taylor1))
    # if tf.abs(diff/val) < 2e-4:   # this seems a good compromise between finite subtraction and taylor series
    Order2N = val * tf.where(tf.abs(diff / val) < 2e-4, taylor1, diff)
    p = taylor1 + (b2 + Order2N) / (2.0 * mysqrt), grad  # this should be numerically more stable
    return p


# This monotonic positive function is based on a Hyperbola modified that one of the branches appraoches zero and the other one reaches a slope of one
def invMonotonicPos(invinput, b2=1.0, Eps=0.0):
    # a constant value > 0.0 0 which regulates the shape of the hyperbola. The bigger the smoother it becomes.
    tfinit = tf.clip_by_value(invinput, clip_value_min=tf.constant(Eps, dtype=CalcFloatStr),
                              clip_value_max=tf.constant(np.Inf, dtype=CalcFloatStr))  # assertion to obtain only positive input for the initialization
    #    return tf.cast(tfinit - (tf.constant(b2) / tfinit), dtype=CalcFloatStr)  # the inverse of monotonicPos
    return (tf.square(tfinit) - b2) / tfinit  # the inverse of monotonicPos


# def forcePositive(self, State):
#     for varN, var in State.items():
#         State[varN] = self.monotonicPos(State[varN])
#     return State

# def Reset():
#     tf.compat.v1.reset_default_graph()  # clear everything on the GPU

# def Optimize(Fwd,Loss,tfinit,myoptimizer=None,NumIter=40,PreFwd=None):
def Optimize(myoptimizer=None, loss=None, NumIter=40, TBSummary=False, TBSummaryDir="C:\\NoBackup\\TensorboardLogs\\", resVars=None, lossScale=1.0):
    """
    performs the tensorflow optimization given a loss function and an optimizer

    The optimizer currently also needs to know about the loss, which is a (not-yet evaluated) tensor

    Parameters
    ----------
    myoptimizer : an optimizer. See for example "optimizer" and its arguments
    loss : the loss() function with no arguments
    NumIter (default: 40) : Number of iterations to be used, in case that no optimizer is provided. Otherwise this argument is NOT used but the optimizer knows about the number of iterations.
    TBSummary (default: False) : If True, the summary information for tensorboard is stored
    TBSummaryDir (default: "C:\\NoBackup\\TensorboardLogs\\") : The directory whre the tensorboard information is stored.
    Eager (default: False) : Use eager execution
    resVars (default: None) : Which tensors to evaluate and return at the end.

    Returns
    -------
    a tuple of tensors

    See also
    -------

    Example
    -------
    """
    if myoptimizer is None:
        myoptimizer = lambda loss: optimizer(loss, NIter=NumIter)  # if none was provided, use the default optimizer

    if loss != None:
        mystartloss = loss().numpy() * lossScale  # eval()

    start_time = time.time()
    if TBSummary:
        summary = myoptimizer()
    else:
        myoptimizer()
    duration = time.time() - start_time
    #        if TBSummary:
    #            tb_writer = tf.summary.FileWriter(TBSummaryDir + 'Optimize', session.graph)
    #            merged = tf.summary.merge_all()
    #            summary = session.run(merged)
    #            tb_writer.add_summary(summary, 0)
    try:
        optName = myoptimizer.optName
    except:
        optName = "unkown optimizer"
    if loss != None:
        myloss = loss().numpy() * lossScale
        print(optName + ': Exec. time:{:.4}'.format(duration), '. Start L.:{:.4}'.format(mystartloss), ', Final L.:{:.4}'.format(myloss),
              '. Relative L.:{:.4}'.format(myloss / mystartloss))
    else:
        print(optName + ': Exec. time:{:.4}'.format(duration))

    if resVars == None and loss != None:
        return myloss
    else:
        res = []
        if isinstance(resVars, list) or isinstance(resVars, tuple):
            for avar in resVars:
                if not isinstance(avar, tf.Tensor) and not isinstance(avar, tf.Variable):
                    print("WARNING: Variable " + str(avar) + " is NOT a tensor.")
                    res.append(avar)
                else:
                    try:
                        res.append(avar.eval())
                    except ValueError:
                        print("Warning. Could not evaluate result variable" + avar.name + ". Returning [] for this result.")
                        res.append([])
        else:
            res = resVars.eval()
    return res
    #    nip.view(toshow)


def datatype(tfin):
    if istensor(tfin):
        return tfin.dtype
    else:
        if isinstance(tfin, np.ndarray):
            return tfin.dtype.name
        return tfin  # assuming this is already the type


def istensor(tfin):
    return isinstance(tfin, tf.Tensor) or isinstance(tfin, tf.Variable)


def iscomplex(mytype):
    mytype = str(datatype(mytype))
    return (mytype == "complex64") or (mytype == "complex128") or (mytype == "complex64_ref") or (mytype == "complex128_ref") or (mytype == "<dtype: 'complex64'>") or (
            mytype == "<dtype: 'complex128'>")


def isNumber(val):
    return isinstance(val, numbers.Number)


def isList(val):
    return isinstance(val, list)


def isTuple(val):
    return isinstance(val, tuple)


def removeCallable(ten):
    if callable(ten):
        return ten()
    else:
        return ten


def totensor(img):
    if istensor(img) or callable(img):
        return img
    if isList(img):
        img = np.array(img, CalcFloatStr)

    if not isNumber(img) and ((img.dtype == defaultTFDataType) or (img.dtype == defaultTFCpxDataType)):
        img = tf.constant(img)
    else:
        if iscomplex(img):
            img = tf.constant(img, defaultTFCpxDataType)
        else:
            img = tf.constant(img, defaultTFDataType)
    return img


def doCheckScaling(fwd, meas):
    sF = ev(tf.reduce_mean(input_tensor=totensor(fwd)))
    sM = ev(tf.reduce_mean(input_tensor=totensor(meas)))
    R = sM / sF
    if abs(R) < 0.7 or abs(R) > 1.3:
        print("Mean of measured data: " + str(sM) + ", Mean of forward model with initialization: " + str(sF) + " Ratio: " + str(R))
        print(
            "WARNING!! The forward projected sum is significantly different from the provided measured data. This may cause problems during optimization. To prevent this warning: set checkScaling=False for your loss function.")
    return tf.debugging.check_numerics(fwd, "Detected NaN or Inf in loss function")  # also checks for NaN values during runtime


# %% this section defines a number of loss functions. Note that they often need fixed input arguments for measured data and sometimes more parameters
def Loss_FixedGaussian(fwd, meas, lossDataType=None, checkScaling=False):
    if lossDataType is None:
        lossDataType = defaultLossDataType
    if checkScaling:
        fwd = doCheckScaling(fwd, meas)

    with tf.compat.v1.name_scope('Loss_FixedGaussian'):
        #       return tf.reduce_sum(tf.square(fwd-meas))  # version without normalization
        if iscomplex(fwd.dtype.as_numpy_dtype):
            mydiff = (fwd - meas)
            return tf.reduce_mean(input_tensor=tf.cast(mydiff * tf.math.conj(mydiff), lossDataType)) / tf.reduce_mean(
                input_tensor=tf.cast(meas, lossDataType))  # to make everything scale-invariant. The TF framework hopefully takes care of precomputing this
        else:
            return tf.reduce_mean(input_tensor=tf.cast(tf.square(fwd - meas), lossDataType)) / tf.reduce_mean(
                input_tensor=tf.cast(meas, lossDataType))  # to make everything scale-invariant. The TF framework hopefully takes care of precomputing this


def Loss_ScaledGaussianReadNoise(fwd, meas, RNV=1.0, lossDataType=None, checkScaling=False):
    if lossDataType is None:
        lossDataType = defaultLossDataType
    if checkScaling:
        fwd = doCheckScaling(fwd, meas)
    offsetcorr = tf.cast(tf.reduce_mean(tf.math.log(tf.math.maximum(meas, tf.constant(0.0, dtype=CalcFloatStr)) + RNV)),
                         lossDataType)  # this was added to have the ideal fit yield a loss equal to zero

    # with tf.compat.v1.name_scope('Loss_ScaledGaussianReadNoise'):
    XMinusMu = tf.cast(meas - fwd, lossDataType)
    muPlusC = tf.cast(tf.math.maximum(fwd, 0.0) + RNV, lossDataType)  # the clipping at zero was introduced to avoid division by zero
    # if tf.reduce_any(RNV == tf.constant(0.0, CalcFloatStr)):
    #     print("RNV is: "+str(RNV))
    #     raise ValueError("RNV is zero!.")
    # if tf.reduce_any(muPlusC == tf.constant(0.0, CalcFloatStr)):
    #     print("Problem: Division by zero encountered here")
    #     raise ValueError("Division by zero HERE!.")
    Fwd = tf.math.log(muPlusC) + tf.square(XMinusMu) / muPlusC
    #       Grad=Grad.*(1.0-2.0*XMinusMu-XMinusMu.^2./muPlusC)./muPlusC;
    Fwd = tf.reduce_mean(input_tensor=Fwd)
    # if tf.math.is_nan(Fwd):
    #     if tf.reduce_any(muPlusC == tf.constant(0.0, CalcFloatStr)):
    #         print("Problem: Division by zero encountered")
    #         raise ValueError("Division by zero.")
    #     else:
    #         raise ValueError("Nan encountered.")
    return Fwd  # - offsetcorr  # to make everything scale-invariant. The TF framework hopefully takes care of precomputing this


# @tf.custom_gradient
def Loss_Poisson(fwd, meas, Bg=0.05, checkPos=False, lossDataType=None, checkScaling=False):
    if lossDataType is None:
        lossDataType = defaultLossDataType
    if checkScaling:
        fwd = doCheckScaling(fwd, meas)

    with tf.compat.v1.name_scope('Loss_Poisson'):
        #       meas[meas<0]=0
        meanmeas = tf.reduce_mean(meas)
        #    NumEl=tf.size(meas)
        if checkPos:
            fwd = ((tf.sign(fwd) + 1) / 2) * fwd
        FwdBg = tf.cast(fwd + Bg, lossDataType)
        totalError = tf.reduce_mean(input_tensor=(FwdBg - meas) - meas * tf.math.log(
            (FwdBg) / (meas + Bg))) / meanmeas  # the modification in the log normalizes the error. For full normalization see PoissonErrorAndDerivNormed
        #       totalError = tf.reduce_mean((fwd-meas) - meas * tf.log(fwd)) / meanmeas  # the modification in the log normalizes the error. For full normalization see PoissonErrorAndDerivNormed
        #        def grad(dy):
        #            return dy*(1.0 - meas/(fwd+Bg))/meanmeas
        #        return totalError,grad
        return totalError


def Loss_Poisson2(fwd, meas, Bg=0.05, checkPos=False, lossDataType=None, checkScaling=False):
    if lossDataType is None:
        lossDataType = defaultLossDataType
    if checkScaling:
        fwd = doCheckScaling(fwd, meas)

    # with tf.compat.v1.name_scope('Loss_Poisson2'):
    #       meas[meas<0]=0
    meanmeas = tf.reduce_mean(meas)
    meassize = np.prod(meas.shape)
    #    NumEl=tf.size(meas)
    if checkPos:
        fwd = ((tf.sign(fwd) + 1) / 2) * fwd  # force positive

    #       totalError = tf.reduce_mean((fwd-meas) - meas * tf.log(fwd)) / meanmeas  # the modification in the log normalizes the error. For full normalization see PoissonErrorAndDerivNormed
    @tf.custom_gradient
    def BarePoisson(myfwd):
        def grad(dy):
            mygrad = dy * (1.0 - meas / (myfwd + Bg)) / meassize  # the size accounts for the mean operation (rather than sum)
            #                image_shaped_input = tf.reshape(mygrad, [-1, mygrad.shape[0], mygrad.shape[1], 1])
            #                tf.summary.image('mygrad', image_shaped_input, 10)
            return mygrad

        toavg = (myfwd + Bg - meas) - meas * tf.math.log((myfwd + Bg) / (meas + Bg))
        toavg = tf.cast(toavg, lossDataType)
        totalError = tf.reduce_mean(input_tensor=toavg)  # the modification in the log normalizes the error. For full normalization see PoissonErrorAndDerivNormed
        return totalError, grad

    return BarePoisson(fwd) / meanmeas


# ---- End of code from the inverse Modelling Toolbox

def retrieveData():
    import json_to_pandas
    dl = json_to_pandas.DataLoader()  # instantiate DataLoader #from_back_end=True
    data_dict = dl.process_data()  # loads and forms the data dictionary
    rki_data = data_dict["RKI_Data"]  # only RKI dataframe
    print('Last Day loaded: ' + str(pd.to_datetime(np.max(rki_data.Meldedatum), unit='ms')))
    return rki_data


def deltas(WhenHowMuch, SimTimes):
    res = np.zeros(SimTimes)
    for w, h in WhenHowMuch:
        res[w] = h;
    return res


def showResiduum(meas, fit):
    res1 = np.mean(meas - fit, (1, 2))
    print('Loss: ' + str(np.mean(abs(res1) ** 2)))
    plt.plot(res1)
    plt.xlabel('days')
    plt.ylabel('mean difference / cases')
    plt.title('residuum')


def toDay(timeInMs):
    return int(timeInMs / (1000 * 60 * 60 * 24))


def getLabels(rki_data, label):
    try:
        labels = rki_data[label].unique()
        labels.sort();
        labels = labels.tolist()
    except KeyError:
        labels = ['BRD']
    return labels

def correctWeekdayEffect(RawCases):
    s = RawCases.shape
    nTimes = s[0]
    weeks = nTimes//7
    rest = np.mod(nTimes,7)
    weekdayLoad = np.sum(np.reshape(RawCases[:weeks*7], [weeks, 7, s[1], s[2],s[3]]), (0,2,3,4))
    weekdayLoad /= np.mean(weekdayLoad)
    for week in range(weeks):
        RawCases[week*7:(week+1)*7] /= weekdayLoad[:,np.newaxis,np.newaxis,np.newaxis] # attempt to correct for the uneven reporting
    RawCases[weeks*7:] /= weekdayLoad[:rest,np.newaxis,np.newaxis,np.newaxis] # attempt to correct for the uneven reporting
    return RawCases

def binThuringia(data, df):
    #import locale
    #locale.setlocale(locale.LC_ALL, 'de_DE')
    whichDate = 'Erkrankungsbeginn'
    # data = data.sort_values(whichDate)
    if '"AbsonderungEnde"' in data.keys():
        AbsonderungEnde = '"AbsonderungEnde"'
    else:
        AbsonderungEnde = '"AbsonderungBis"'
    data['AbsonderungEnde'] = pd.to_datetime(data[AbsonderungEnde].str.replace('"', '').str[:10], dayfirst=True)
    day1 = np.min(data[whichDate])
    dayLast1 = np.max(data[whichDate] - day1)
    dayLast2 = np.max(data['VerstorbenDatum'] - day1)
    dayLast3 = np.max(data['AbsonderungEnde'] - day1)
    dayLast = np.max([dayLast1, dayLast2, dayLast3])
    numDays = dayLast.days + 1
    minAge = np.min(data[data['AlterBerechnet'] > 0]['AlterBerechnet'])
    maxAge = np.max(data[data['AlterBerechnet'] > 0]['AlterBerechnet'])
    numAge = maxAge + 1
    labelsLK, levelsLK = data['MeldeLandkreis'].factorize()
    data['LandkreisID'] = labelsLK
    minLK = np.min(data['LandkreisID'])
    maxLK = np.max(data['LandkreisID'])
    numLK = maxLK + 1
    labels, levelsGe = data['Geschlecht'].factorize()
    data['GeschlechtID'] = labels
    minGe = np.min(data['GeschlechtID'])
    maxGe = np.max(data['GeschlechtID'])
    numGender = maxGe + 1
    Cases = np.zeros([numDays, numLK, numAge, numGender])
    Hospitalized = np.zeros([numDays, numLK, numAge, numGender])
    Cured = np.zeros([numDays, numLK, numAge, numGender])
    Dead = np.zeros([numDays, numLK, numAge, numGender])
    # data = data.set_index('InterneRef') # to make it unique
    for index, row in data.iterrows():
        myLK = int(row['LandkreisID'])
        myday = (row[whichDate] - day1).days
        if myday is np.nan:
            myday = (row['Meldedatum']-day1).days
        myAge = row['AlterBerechnet']
        myGender = row['GeschlechtID']
        if myAge < 1:
            print('unknown age.' + str(myAge)+'... skipping ...')
            continue
        Cases[myday, myLK, myAge, myGender] += 1.0
        myCuredDay = (row['AbsonderungEnde'] - day1).days
        if myCuredDay is not np.nan:
            Cured[myCuredDay, myLK, myAge, myGender] += 1
        if row['HospitalisierungStatus'] == "Ja":
            Hospitalized[myday, myLK, myAge, myGender] += 1
        myDeadDay = (row['VerstorbenDatum'] - day1).days
        if myDeadDay is not np.nan:
            Dead[myDeadDay, myLK, myAge, myGender] += 1
    Dates = pd.date_range(start = day1, periods=numDays).map(lambda x: x.strftime('%d.%m.%Y')).to_list()

    df = df.set_index('Kreisfreie Stadt\nKreis / Landkreis')
    Area = np.zeros(numLK)
    PopW = np.zeros(numLK)
    PopM = np.zeros(numLK)
    for lk, level in zip(np.arange(numLK), levelsLK):
        if level[:3] == 'LK ':
            level = level[3:]
        elif level[:3] == 'SK ':
            level = level[3:]+', Stadt'
        mySuppl = df.loc[level]
        Area[lk] = mySuppl['Flaeche in km2']
        PopW[lk] = mySuppl['Bev. W']
        PopM[lk] = mySuppl['Bev. M']
    if True:
        # now adapt the age groups to the RKI-data:
        Ages = ('0-4','5-14', '15-34', '35-59', '60-79', '> 80')
        AgeGroupStart = (5,14,34,59,79, None)
        C=[];D=[];Cu=[];H=[]
        AgeStart=0
        for AgeEnd in AgeGroupStart:
            C.append(np.sum(Cases[:,:,AgeStart:AgeEnd,:],-2))
            D.append(np.sum(Dead[:,:,AgeStart:AgeEnd,:],-2))
            Cu.append(np.sum(Cured[:,:,AgeStart:AgeEnd,:],-2))
            H.append(np.sum(Hospitalized[:,:,AgeStart:AgeEnd,:],-2))
            AgeStart = AgeEnd
        Cases = np.stack(C,-2);Dead = np.stack(D,-2);Cured = np.stack(Cu,-2); Hospitalized = np.stack(H,-2)
    else:
        Ages = np.arange(numAge)
    Gender = levelsGe
    CumulCases = np.cumsum(Cases,0)
    CumulDead = np.cumsum(Dead,0)
    CumulHospitalized = np.cumsum(Hospitalized,0)

    measured = {'Cases': Cases, 'Hospitalized': Hospitalized, 'Dead': Dead, 'Cured': Cured,
                'LKs': levelsLK.to_list(), 'IDs': labelsLK, 'Dates': Dates, 'Gender': Gender, 'Ages': Ages,
                'PopM': PopM, 'PopW':PopW, 'Area': Area, 'CumulCases': CumulCases,
                'CumulDead': CumulDead,'CumulHospitalized': CumulHospitalized}
    return measured

def cumulate(rki_data, df):
    # rki_data.keys()  # IdBundesland', 'Bundesland', 'Landkreis', 'Altersgruppe', 'Geschlecht',
    #        'AnzahlFall', 'AnzahlTodesfall', 'ObjectId', 'Meldedatum', 'IdLandkreis'
    # TotalCases = 0;
    # whichDate = 'Refdatum'  # 'Meldedatum'
    whichDate = 'Meldedatum' # It may be useful to stick to the "Meldedatum", since Refdatum is a mix anyway.
    # Furthermore: Redatum has a missing data problem near the end of the reporting period, so it always goes down!
    rki_data = rki_data.sort_values(whichDate)
    day1 = toDay(np.min(rki_data[whichDate]))
    dayLast = toDay(np.max(rki_data[whichDate]))
    toDrop = []
    toDropID = []
    ValidIDs = df['Key'].to_numpy()
    print("rki_data (before drop): " + str(np.sum(rki_data.to_numpy()[:,5])))
    # sumDropped=0
    rki_data = rki_data.set_index('ObjectId') # to make it unique
    for index, row in rki_data.iterrows():
        myId = int(row['IdLandkreis'])
        if myId not in ValidIDs:
            myLK = row['Landkreis']
            if myId not in toDropID:
                print("WARNING: RKI-data district " + str(myId) + ", " + myLK + " is not in census. Dropping this data.")
                toDropID.append(myId)
            toDrop.append(index)
            # print("Dropping: "+str(index)+", "+str(myLK))
            # sumDropped += int(row['AnzahlFall'])
            # print("Dropped: "+str(sumDropped))
    rki_data = rki_data.drop(toDrop)
    print("rki_data (directly after drop): " + str(np.sum(rki_data.to_numpy()[:,5])))

    IDs = getLabels(rki_data, 'IdLandkreis')
    LKs = getLabels(rki_data, 'Landkreis')
    Ages = getLabels(rki_data, 'Altersgruppe')
    Gender = getLabels(rki_data, 'Geschlecht')
    CumulSumCases = np.zeros([len(LKs), len(Ages), len(Gender)])
    AllCumulCases = np.zeros([dayLast - day1 + 1, len(LKs), len(Ages), len(Gender)])
    CumulSumDead = np.zeros([len(LKs), len(Ages), len(Gender)])
    AllCumulDead = np.zeros([dayLast - day1 + 1, len(LKs), len(Ages), len(Gender)])
    AllCases = np.zeros([dayLast - day1 + 1, len(LKs), len(Ages), len(Gender)])
    AllDead = np.zeros([dayLast - day1 + 1, len(LKs), len(Ages), len(Gender)])
    AllCured = np.zeros([dayLast - day1 + 1, len(LKs), len(Ages), len(Gender)])
    CumulSumCured = np.zeros([len(LKs), len(Ages), len(Gender)])
    AllCumulCured = np.zeros([dayLast - day1 + 1, len(LKs), len(Ages), len(Gender)])
    Area = np.zeros(len(LKs))
    PopW = np.zeros(len(LKs))
    PopM = np.zeros(len(LKs))
    allDates = (dayLast - day1 + 1)*['']

    df = df.set_index('Key')
    # IDs = [int(ID) for ID in IDs]
    # diff = df.index.difference(IDs)
    # if len(diff) != 0:
    #     for Id in diff:
    #         Name = df.loc[Id]['Kreisfreie Stadt\nKreis / Landkreis']
    #         print("WARNING: "+str(Id)+", "+Name+" is not mentioned in the RKI List. Removing Entry.")
    #         df = df.drop(Id)
    #     IDs = [int(ID) for ID in IDs]
    # sorted = df.loc[IDs]

    # CumulMale = np.zeros(dayLast-day1); CumulFemale = np.zeros(dayLast-day1)
    # TMale = 0; TFemale = 0; # TAge = zeros()
    prevday = -1
    sumTotal = 0
    for index, row in rki_data.iterrows():
        myLKId = row['IdLandkreis']
        myLK = LKs.index(row['Landkreis'])
        if myLKId not in IDs:
            ValueError("Something went wrong! These datasets should have been dropped already.")
        mySuppl = df.loc[int(myLKId)]
        Area[myLK] = mySuppl['Flaeche in km2']
        PopW[myLK] = mySuppl['Bev. W']
        PopM[myLK] = mySuppl['Bev. M']
        # datetime = pd.to_datetime(row['Meldedatum'], unit='ms').to_pydatetime()
        # day = toDay(row['Meldedatum']) - day1  # convert to days with an offset
        day = toDay(row[whichDate]) - day1  # convert to days with an offset
        # dayD = datetime.strptime(row['Datenstand'][:10], '%d.%m.%Y') - datetime(1970,1,1)
        # rday = dayD.days - day1  # convert to days with an offset
        # print(day)
        allDates[day] = pd.to_datetime(row[whichDate], unit='ms').strftime("%d.%m.%Y") #dayfirst=True, yearfirst=False
        myAge = Ages.index(row['Altersgruppe'])
        myG = Gender.index(row['Geschlecht'])
        NeuerFall = row['NeuerFall']  # see the fetch_data.py file for the details of what NeuerFall means.
        if NeuerFall == -1:
            AnzahlFall = 0
        else:
            AnzahlFall = row['AnzahlFall']
        sumTotal += AnzahlFall
        NeuerTodesFall = row['NeuerTodesfall']  # see the fetch_data.py file for the details of what NeuerFall means.
        if NeuerTodesFall == 0 or NeuerTodesFall == 1: # only new cases are counted. NeuerTodesFall == 0 or
            AnzahlTodesfall = row['AnzahlTodesfall']
        else:
            AnzahlTodesfall = 0
        NeuGenesen = row['NeuGenesen']  # see the fetch_data.py file for the details of what NeuerFall means.
        if NeuGenesen == 0 or NeuGenesen == 1: # only newly cured
             AnzahlGenesen = row['AnzahlGenesen']
        else:
            AnzahlGenesen = 0
        AllCases[day, myLK, myAge, myG] += AnzahlFall
        AllDead[day, myLK, myAge, myG] += AnzahlTodesfall
        AllCured[day, myLK, myAge, myG] += AnzahlGenesen

        CumulSumCases[myLK, myAge, myG] += AnzahlFall
        AllCumulCases[prevday + 1:day + 1, :, :, :] = CumulSumCases
        CumulSumDead[myLK, myAge, myG] += AnzahlTodesfall
        AllCumulDead[prevday + 1:day + 1, :, :, :] = CumulSumDead
        CumulSumCured[myLK, myAge, myG] += AnzahlGenesen
        AllCumulCured[prevday + 1:day + 1, :, :, :] = CumulSumCured
        if day < prevday:
            print(row['Key'])
            raise ValueError("Something is wrong: dates are not correctly ordered")
        prevday = day-1

    print("Total Cases: "+ str(sumTotal))
    print("rki_data (at the end): " + str(np.sum(rki_data.to_numpy()[:,5])))
    measured = {'CumulCases': AllCumulCases, 'CumulDead': AllCumulDead, 'Cases': AllCases, 'Dead': AllDead, 'Cured': AllCured,
                'IDs': IDs, 'LKs':LKs, 'PopM': PopM, 'PopW': PopW, 'Area': Area, 'Ages':Ages, 'Gender': Gender, 'Dates': allDates}
    # AllCumulCase, AllCumulDead, AllCumulCured, (IDs, LKs, PopM, PopW, Area, Ages, Gender, allDates)
    return measured


def plotAgeGroups(res1, res2):
    plt.figure()
    plt.title('Age Groups')
    plt.plot(res1)
    plt.gca().set_prop_cycle(None)
    plt.plot(res2, '--')
    plt.xlabel('days')
    plt.ylabel('population')


class axisType:
    const = 'const'
    gaussian = 'gaussian'
    sigmoid = 'sigmoid'
    individual = 'individual'
    uniform = 'uniform'


def prependOnes(s1, s2):
    l1 = len(s1);
    l2 = len(s2)
    maxDim = max(l1, l2)
    return np.array((maxDim - l1) * [1] + list(s1)), np.array((maxDim - l2) * [1] + list(s2))


def equalShape(s1, s2):
    if isinstance(s1, tf.TensorShape):
        s1 = s1.as_list()
    if isinstance(s2, tf.TensorShape):
        s2 = s2.as_list()
    s1, s2 = prependOnes(s1, s2)
    return np.linalg.norm(s1 - s2) == 0


class Axis:
    def ramp(self):
        x = self.shape
        if isinstance(x, np.ndarray) or isNumber(x) or isTuple(x) or isList(x):
            aramp = tf.constant(np.arange(np.max(x)), dtype=CalcFloatStr)
            if isNumber(x):
                x = [x]
            x = tf.reshape(aramp, x)  # if you get an error here, the size is not 1D!
        else:
            x = totensor(x)
        return x

    def __init__(self, name, numAxis, maxAxes, entries=1, queue=False, labels=None):
        self.name = name
        self.queue = queue
        self.shape = np.ones(maxAxes, dtype=int)
        self.shape[-numAxis] = entries
        self.curAxis = numAxis
        self.Labels = labels

        # self.initFkt = self.initZeros()

    def __str__(self):
        return self.name + ", number:" + str(self.curAxis) + ", is queue:" + str(self.queue)

    def __repr__(self):
        return self.__str__()

    def initZeros(self):
        return tf.constant(0.0, dtype=CalcFloatStr, shape=self.shape)

    def init(self, vals):
        if isNumber(vals):
            return tf.constant(vals, dtype=CalcFloatStr, shape=self.shape)
        else:
            if isinstance(vals, list) or isinstance(vals, np.ndarray):
                if len(vals) != np.prod(self.shape):
                    raise ValueError('Number of initialization values ' + str(len(vals)) + ' of variable ' + self.name + ' does not match its shape ' + str(self.shape))
                vals = np.reshape(np.array(vals, dtype=CalcFloatStr), self.shape)
            # if callable(vals):
            #    vshape = vals().shape
            # else:
            #    vshape = vals.shape
            # if not equalShape(vshape, self.shape):
            #    raise ValueError('Initialization shape ' + str(vshape) + ' of variable ' + self.name + ' does not match its shape ' + str(self.shape))
            return totensor(vals)

    def initIndividual(self, vals):
        return tf.variable(vals, dtype=CalcFloatStr)

    def initGaussian(self, mu=0.0, sig=1.0):
        x = self.ramp()
        mu = totensor(mu)
        sig = totensor(sig)
        initVals = tf.exp(-(x - mu) ** 2. / (2 * (sig ** 2.)))
        initVals = initVals / tf.reduce_sum(input_tensor=initVals)  # normalize (numerical !, since the domain is not infinite)
        return initVals

    def initDelta(self, pos=0):
        x = self.ramp()
        initVals = tf.cast(x == pos, CalcFloatStr)  # 1.0 *
        return initVals

    def initSigmoid(self, mu=0.0, sig=1.0, offset=0.0):
        x = self.ramp()
        mu = totensor(mu);
        sig = totensor(sig)
        initVals = 1. / (1. + tf.exp(-(x - mu) / sig)) + offset
        initVals = initVals / tf.reduce_sum(input_tensor=initVals)  # normalize (numerical !, since the domain is not infinite)
        return initVals


def NDim(var):
    if istensor(var):
        return var.shape.ndims
    else:
        return var.ndim


def subSlice(var, dim, sliceStart, sliceEnd):  # extracts a subslice along a particular dimension
    numdims = NDim(var)
    idx = [slice(sliceStart, sliceEnd) if (d == dim or numdims + dim == d) else slice(0, None) for d in range(numdims)]
    return var[idx]


def firstSlice(var, dim):  # extracts the first subslice along a particular dimension
    return subSlice(var, dim, 0, 1)


def lastSlice(var, dim):  # extracts the last subslice along a particular dimension
    return subSlice(var, dim, -1, None)


def reduceSumTo(State, dst):
    # redsz = min(sz1, sz2)
    if isinstance(dst, np.ndarray):
        dstSize = np.array(dst.shape)
    else:
        dstSize = np.array(dst.shape.as_list(), dtype=int)
    if len(dst.shape) == 0:  # i.e. a scalar
        dstSize = np.ones(State.ndim, dtype=int)
    rs = np.array(State.shape.as_list(), dtype=int)
    toReduce = np.nonzero((rs > dstSize) & (dstSize == 1))
    toReduce = list(toReduce[0])
    if toReduce is not None:
        State = tf.reduce_sum(input_tensor=State, axis=toReduce, keepdims=True)
    return State


# class State:
#     def __init__(self, name='aState'):
#         self.name = name
#         self.Axes = {}

class Model:
    def __init__(self, name='stateModel', maxAxes=4, rand_seed=1234567):
        self.__version__ = 1.01
        self.name = name
        self.maxAxes = maxAxes
        self.curAxis = 1
        self.QueueStates = {}  # stores the queue axis in every entry
        self.Axes = {}
        self.RegisteredAxes = []  # just to have a convenient way of indexing them
        self.State = {}  # dictionary of state variables
        self.Var = {}  # may be variables or lambdas
        self.rawVar = {}  # saves the raw variables
        self.toRawVar = {}  # stores the inverse functions to initialize the rawVar
        self.toVar = {}  # stores the function to get from the rawVar to the Var
        self.Original = {}  # here the values previous to a distortion are stored (for later comparison)
        self.Distorted = {}  # here the values previous to a distortion are stored (for later comparison)
        self.Simulations = {}
        self.Measurements = {}
        self.FitResultVals = {}  # resulting fit results (e.g. forward model or other curves)
        self.FitResultVars = {}  # resulting fit variables
        self.Rates = []  # stores the rate equations
        self.Loss = []
        self.ResultCalculator = {}  # remembers the variable names that define the results
        self.ResultVals = {}
        self.Progression = {}  # dictionary storing the state and resultVal(s) progression (List per Key)
        np.random.seed(rand_seed)

    def addAxis(self, name, entries, queue=False, labels=None):
        axis = Axis(name, self.curAxis, self.maxAxes, entries, queue, labels)
        self.curAxis += 1
        self.Axes[name] = axis
        self.RegisteredAxes.append(axis)

    def initGaussianT0(self, t0, t, sig=2.0):
        initVals = tf.exp(-(t - t0) ** 2. / (2 * (sig ** 2.)))
        return initVals

    def initDeltaT0(self, t0, t, sig=2.0):
        initVals = ((t - t0) == 0.0)*1.0
        return initVals

    def initSigmoidDropT0(self, t0, t, sig, dropTo=0.0):
        initVals = (1.-dropTo) / (1. + tf.exp((t - t0) / sig)) + dropTo
        return initVals

    def newState(self, name, axesInit=None, makeInitVar=False):
        # state = State(name)
        # self.States[name]=state

        if name in self.ResultCalculator:
            raise ValueError('Key ' + name + 'already exists in results.')
        elif name in self.State:
            raise ValueError('Key ' + name + 'already exists as a state.')
        prodAx = None
        if not isinstance(axesInit, dict):
            if (not isNumber(axesInit)) and (np.prod(removeCallable(axesInit).shape) != 1):
                raise ValueError("State " + name + " has a non-scalar initialization but no related axis. Please make it a dictionary with keys being axes names.")
            else:
                # no changes (like reshape to the original tensors are allowed since this "breaks" the chain of connections
                axesInit = {'StartVal': totensor(axesInit)}  # so that it can be appended to the time trace
        if axesInit is not None:
            res = []
            hasQueue = False
            for AxName, initVal in axesInit.items():
                if AxName in self.Axes:
                    myAxis = self.Axes[AxName]
                    if (initVal is None):
                        continue
                        # initVal = myAxis.init(1.0/np.prod(myAxis.shape, dtype=CalcFloatStr))
                    if (not isinstance(initVal, Axis) and not callable(initVal)) or isNumber(initVal):
                        initVal = myAxis.init(initVal)
                    if myAxis.queue:
                        if hasQueue:
                            raise ValueError("Each State can only have one queue axis. This state " + name + " wants to have more than one.")
                        hasQueue = True
                        self.QueueStates[name] = myAxis
                else:
                    initVal = totensor(initVal)
                if res == []:
                    res = initVal
                elif callable(res):
                    if callable(initVal):
                        res = res() * initVal()
                    else:
                        res = res() * initVal
                else:
                    if callable(initVal):
                        res = res * initVal()
                    else:
                        res = res * initVal
        if makeInitVar:  # make the initialization value a variable
            prodAx = self.newVariables({name: prodAx})  # initially infected
        elif not callable(res):
            prodAx = lambda: res
        else:
            prodAx = res
        self.State[name] = prodAx

    def newVariables(self, VarList=None, forcePos=True, normalize='max', b2=1.0, overwrite=True):
        if VarList is not None:
            for name, initVal in VarList.items():
                if name in self.Var:
                    if not overwrite:
                        raise ValueError("Variable " + name + " was previously defined.")
                    else:
                        self.assignNewVar(name, initVal)
                        print('assigned new value to variable: '+name)
                    continue
                if name in self.State:
                    raise ValueError("Variable " + name + " is already defined as a State.")
                if name in self.ResultVals:
                    raise ValueError("Variable " + name + " is already defined as a Result.")

                toVarFkt = lambda avar: avar
                toRawFkt = lambda avar: avar
                if normalize is not None:
                    toRawFkt2 = lambda avar: invNormalize(toRawFkt(avar), normalize, initVal);
                    toVarFkt2 = lambda avar: toVarFkt(doNormalize(avar, normalize, initVal))
                else:
                    toRawFkt2 = toRawFkt
                    toVarFkt2 = toVarFkt
                if forcePos:
                    toRawFkt3 = lambda avar: invMonotonicPos(toRawFkt2(avar), b2);
                    toVarFkt3 = lambda avar: toVarFkt2(monotonicPos(avar, b2))
                else:
                    toRawFkt3 = toRawFkt2
                    toVarFkt3 = toVarFkt2

                rawvar = tf.Variable(toRawFkt3(initVal), name=name, dtype=CalcFloatStr)
                self.toRawVar[name] = toRawFkt3
                self.rawVar[name] = rawvar  # this is needed for optimization
                self.toVar[name] = toVarFkt3
                self.Var[name] = lambda: toVarFkt3(rawvar)
                self.Original[name] = rawvar.numpy()  # store the original

        return self.Var[name]  # return the last variable for convenience

    def restoreOriginal(self):
        for varN, rawval in self.Original.items():
            self.rawVar[varN].assign(rawval)

    def assignNewVar(self, varname, newval=None, relval=None):
        if newval is not None:
            self.rawVar[varname].assign(self.toRawVar[varname](newval))
        else:
            self.rawVar[varname].assign(self.toRawVar[varname](self.Var[varname]() * relval))

    def addRate(self, fromState, toState, rate, queueSrc=None, queueDst=None, name=None, hasTime=False, hoSumDims=None, resultTransfer=None):  # S ==> I[0]
        if queueSrc is not None:
            ax = self.QueueStates[fromState]
            if queueSrc != ax.name and queueSrc != "total":
                raise ValueError('The source state ' + fromState + ' does not have an axis named ' + queueSrc + ', but it was given as queueSrc.')
        if queueDst is not None:
            ax = self.QueueStates[toState]
            if queueDst != ax.name:
                raise ValueError('The destination state ' + toState + ' does not have an axis named ' + queueDst + ', but it was given as queueDst.')
        if hoSumDims is not None:
            hoSumDims = [- self.Axes[d].curAxis for d in hoSumDims]

        self.Rates.append([fromState, toState, rate, queueSrc, queueDst, name, hasTime, hoSumDims, resultTransfer])

    def findString(self, name, State=None):
        if State is None:
            State = self.State
        if name in self.Var:
            return self.Var[name]
        elif name in self.Axes:
            return self.Axes[name]
        elif name in State:
            return State[name]
        elif name in self.ResultVals:
            return self.ResultVals[name]
        else:
            ValueError('findString: Value ' + name + ' not found in Vars, States or Results')

    def reduceToResultByName(self, transferred, resultTransfer):
        Results = {}
        if isinstance(resultTransfer, list) or isinstance(resultTransfer, tuple):
            resultTransferName = resultTransfer[0]
            resultT = tf.reduce_sum(transferred, self.findAxesDims(resultTransfer[1:]), keepdims=True)
        else:
            resultTransferName = resultTransfer
            resultT = transferred

        if resultTransferName in Results:
            if resultT.shape == Results[resultTransferName].shape:
                Results[resultTransferName] = Results[resultTransferName] + resultT
            else:
                raise ValueError('Shape not the same in resultTransfer ' + resultTransferName + ' from:' + fromName + ' to ' + toName)
        else:
            Results[resultTransferName] = resultT
        return Results

    def applyRates(self, State, time):
        toQueue = {}  # stores the items to enter into the destination object
        Results = {}
        # insert here the result variables
        OrigStates = State.copy()  # copies the dictionary but NOT the variables in it
        for fromName, toName, rate, queueSrc, queueDst, name, hasTime, hoSumDims, resultTransfer in self.Rates:
            if isinstance(rate, str):
                rate = self.findString(rate)
            higherOrder = None
            if isinstance(fromName, list) or isinstance(fromName, tuple):  # higher order rate
                higherOrder = fromName[1:]
                fromName = fromName[0]
            fromState = OrigStates[fromName]
            if queueSrc is not None:
                if queueSrc in self.Axes:
                    axnum = self.Axes[queueSrc].curAxis
                    fromState = lastSlice(fromState, -axnum)
                elif queueSrc == "total":
                    pass
                else:
                    raise ValueError("Unknown queue source: " + str(queueSrc) + ". Please select an axis or \"total\".")
            if hasTime:
                if callable(rate):
                    rate = rate(time)  # calculate the transfer for this rate equation
                else:
                    tf.print("WARNING: hasTime is True, but the rate is not callable!")
            else:
                if callable(rate):
                    rate = rate()  # calculate the transfer for this rate equation
            transferred = fromState * rate  # calculate the transfer for this rate equation
            if higherOrder is not None:
                for hState in higherOrder:
                    if hoSumDims is None:
                        hoSum = OrigStates[hState]
                    else:
                        hoSum = tf.reduce_sum(OrigStates[hState], hoSumDims, keepdims=True)
                    transferred = transferred * hoSum  # apply higher order rates
            if resultTransfer is not None:
                if isinstance(resultTransfer, list) or isinstance(resultTransfer, tuple):
                    if isinstance(resultTransfer[0], list) or isinstance(resultTransfer[0], tuple):
                        for rT in resultTransfer:
                            Res = self.reduceToResultByName(transferred, rT)
                            Results = addDicts(Results, Res)
                            # State = addDicts(State, Res)
                    else:
                        Res = self.reduceToResultByName(transferred, resultTransfer)
                        Results = addDicts(Results, Res)
                        # State = addDicts(State, Res)
                else:
                    Results = addDicts(Results, {resultTransfer: transferred})
                    # State = addDicts(State, {resultTransfer: transferred})
            try:
                toState = OrigStates[toName]
            except KeyError:
                raise ValueError('Error in Rate equation: state "' + str(toName) + '" was not declared. Please use Model.newState() first.')

            if queueDst is not None:  # handle the queuing
                axnum = self.Axes[queueDst].curAxis
                if toName in toQueue:
                    toS, lastAx = toQueue[toName]
                else:
                    toS = firstSlice(toState, -axnum) * 0.0
                if queueSrc == 'total':
                    scalarRate = tf.reduce_sum(transferred, keepdims=True)
                    toS = toS + reduceSumTo(scalarRate, toS)
                else:
                    toS = toS + reduceSumTo(transferred, toS)
                toQueue[toName] = (toS, axnum)
            else:  # just apply to the destination state
                myTransfer = reduceSumTo(transferred, OrigStates[toName])
                myTransfer = self.ReduceByShape(OrigStates[toName], myTransfer)
                State[toName] = State[toName] + myTransfer

            if queueSrc is None or queueSrc == "total":
                myTransfer = reduceSumTo(transferred, State[fromName])
                transferred = self.ReduceByShape(State[fromName], myTransfer)
                State[fromName] = State[fromName] - transferred  # the original needs to be individually subtracted!
            else:
                pass  # this dequeing is automatically removed
        self.advanceQueues(State, toQueue)
        return State, Results

    def ReduceByShape(self, State, Transfer):
        factor = np.prod(np.array(Transfer.shape) / np.array(State.shape))
        if factor != 1.0:
            Transfer = Transfer * factor
        return Transfer

    def advanceQueues(self, State, toQueue):
        for queueN in self.QueueStates:
            dstState = State[queueN]
            if queueN in toQueue:  # is this queued state a target of a rate equation?
                (dst, axnum) = toQueue[queueN]  # unpack the information
                myAx = self.QueueStates[queueN]
                if axnum != myAx.curAxis:
                    raise ValueError("The axis " + myAx.name + " of the destination state " + queueN + " of a rate equation does not agree to the axis definition direction.")
            else:  # advance the state nonetheless, but fill zeros into the entry point
                myAx = self.QueueStates[queueN]
                axnum = myAx.curAxis
                dstShape = dstState.shape.as_list()
                dstShape[-axnum] = 1
                dst = tf.zeros(dstShape)
            # the line below advances the queue
            State[queueN] = tf.concat((dst, subSlice(dstState, -axnum, None, -1)), axis=-axnum)

    def recordResults(self, State, Results):
        # record all States
        for vName, val in State.items():
            # if vName in self.State:
            if vName not in self.Progression:
                self.Progression[vName] = [val]
            else:
                self.Progression[vName].append(val)
            # else: # this is a Result item, which may or may not be used in the calculations below
            #     pass
                # raise ValueError('detected a State, which is not in States.')

        for resName, res in Results.items():
            if resName not in self.ResultVals:
                self.ResultVals[resName] = [res]
            else:
                self.ResultVals[resName].append(res)

        # now record all calculated result values
        for resName, calc in self.ResultCalculator.items():
            res = calc(State)
            # if len(res.shape) == self.maxAxes:
                # sqax = list(range(self.maxAxes - self.curAxis + 1))
                # res = tf.squeeze(res, sqax)
            if resName not in self.ResultVals:
                self.ResultVals[resName] = [res]
            else:
                self.ResultVals[resName].append(res)

    def cleanupResults(self):
        for sName, predicted in self.Progression.items():
            predicted = tf.stack(predicted)
            self.Progression[sName] = predicted
        for predictionName, predicted in self.ResultVals.items():
            predicted = tf.stack(predicted)
            self.ResultVals[predictionName] = predicted

    def checkDims(self, State):
        for varN, var in State.items():
            missingdims = self.maxAxes - len(var.shape)
            if missingdims > 0:
                newShape = [1] * missingdims + var.shape.as_list()
                State[varN] = tf.reshape(var, newShape)
        return State

    def evalLambdas(self, State):
        for varN, var in State.items():
            if callable(var):
                State[varN] = var()
        return State

    def traceModel(self, Tmax, verbose=True):
        print("tracing traceModel")
        # tf.print("running traceModel")
        State = self.State.copy()  # to ensure that self is not overwritten
        State = self.evalLambdas(State)
        State = self.checkDims(State)
        self.ResultVals = {}
        self.Progression = {}
        for t in range(Tmax):
            if verbose:
                print('tracing time step ' + str(t), end='\r')
                tf.print('tracing time step ' + str(t), end='\r')
            newState, Results = self.applyRates(State, t)
            self.recordResults(State, Results)
            State = newState
        print()
        self.cleanupResults()
        print(" .. done\n")
        return State

    def addResult(self, name, anEquation):
        if name in self.ResultCalculator:
            raise ValueError('Key ' + name + 'already exists in results.')
        elif name in self.State:
            raise ValueError('Key ' + name + 'already exists as a state.')
        else:
            self.ResultCalculator[name] = anEquation

    # @tf.function
    # def quadratic_loss_and_gradient(self, x): # x is a list of fit variables
    #     return tfp.math.value_and_gradient(
    #         lambda x: tf.reduce_sum(tf.math.squared_difference(x, self.predicted)), x)

    @tf.function
    def doBuildModel(self, dictToFit, Tmax, FitStart=0, FitEnd=1e10, oparam={"noiseModel": "Gaussian"}):
        print("tracing doBuildModel")
        # tf.print("running doBuildModel")
        finalState = self.traceModel(Tmax)
        Loss = None
        for predictionName, measured in dictToFit.items():
            predicted = self.ResultVals[predictionName]
            try:
                predicted = reduceSumTo(predicted, measured)
            except ValueError:
                print('Predicted: ' + predictionName)
                print('Predicted shape: ' + str(np.array(predicted.shape)))
                print('Measured shape: ' + str(np.array(measured.shape)))
                raise ValueError('Predicted and measured data have different shape. Try introducing np.newaxis into measured data.')
            self.ResultVals[predictionName] = predicted  # .numpy()
            myFitEnd = min(measured.shape[0], predicted.shape[0], FitEnd)
            if "noiseModel" in oparam:
                if oparam["noiseModel"] == "Gaussian":
                    thisLoss = Loss_FixedGaussian(predicted[FitStart:myFitEnd], measured[FitStart:myFitEnd])
                elif oparam["noiseModel"] == "ScaledGaussian":
                    thisLoss = Loss_ScaledGaussianReadNoise(predicted[FitStart:myFitEnd], measured[FitStart:myFitEnd])
                elif oparam["noiseModel"] == "Poisson":
                    thisLoss = Loss_Poisson2(predicted[FitStart:myFitEnd], measured[FitStart:myFitEnd])
                else:
                    ValueError("Unknown noise model: " + oparam["noiseModel"])
            else:
                thisLoss = Loss_FixedGaussian(predicted[FitStart:myFitEnd], measured[FitStart:myFitEnd])
            if Loss is None:
                Loss = thisLoss
            else:
                Loss = Loss + thisLoss
        return Loss, self.ResultVals, self.Progression

    def buildModel(self, dictToFit, Tmax, FitStart=0, FitEnd=1e10):
        Loss = lambda: self.doBuildModel(dictToFit, Tmax, FitStart, FitEnd)
        return Loss

    def simulate(self, resname, varDict={}, Tmax=100, applyPoisson=False, applyGaussian=None):
        finalState = self.traceModel(Tmax)
        measured = {}
        simulated = {}
        for name in varDict:
            varDict[name] = self.findString(name)  # .numpy() # ev()
            simulated[name] = varDict[name].numpy()
            measured[name] = simulated[name]
            if applyPoisson:
                mm = np.min(measured[name])
                if (mm < 0.0):
                    raise ValueError('Poisson noise generator discovered a negative number ' + str(mm) + ' in ' + name)
                measured[name] = self.applyPoissonNoise(measured[name])
            if applyGaussian is not None:
                measured[name] = self.applyGaussianNoise(measured[name], sigma=applyGaussian)
        self.Simulations[resname] = simulated
        self.Measurements[resname] = measured
        if applyPoisson or applyGaussian is not None:
            toReturn = self.Measurements
        else:
            toReturn = self.Simulations
        if len(toReturn.keys()) == 1:
            dict = next(iter(toReturn.values()))
            if len(dict.keys()) == 1:
                return next(iter(dict.values()))

    def applyPoissonNoise(self, data, maxPhotons=None):
        if maxPhotons is not None:
            if maxPhotons > 0:
                return np.random.poisson(maxPhotons * data / np.max(data)).astype(CalcFloatStr)
            else:
                return data
        else:
            return np.random.poisson(data).astype(CalcFloatStr)

    def applyGaussianNoise(self, data, sigma=1.0):
        return np.random.normal(data, scale=sigma).astype(CalcFloatStr)

    def toFit(self, listOfVars):
        self.FitVars = listOfVars

    # def loss_Fn(self):
    #     return self.Loss
    def relDistort(self, var_list):
        for name, relDist in var_list.items():
            var = self.Var[name]
            if callable(var):
                self.rawVar[name].assign(self.toRawVar[name](var() * tf.constant(relDist)))
                self.Distorted[name] = var().numpy()
            else:
                self.Var[name].assign(self.Var[name] * relDist)
                self.Distorted[name] = var.numpy()

    def fit(self, data_dict, Tmax, NIter=50, otype='L-BFGS', oparam={"learning_rate": None}, verbose=False, lossScale=None):
        # if "normFac" not in oparam:
        #     oparam["normFac"] = "max"
        if "learning_rate" not in oparam:
            oparam["learning_rate"] = None

        self.Measurements['measured'] = {}
        for predictionName, measured in data_dict.items():
            data_dict[predictionName] = tf.constant(measured, CalcFloatStr)
            self.Measurements['measured'][predictionName] = data_dict[predictionName]  # save as measurement for plot

        loss_fn = lambda: self.doBuildModel(data_dict, Tmax, oparam=oparam)

        FitVars = [self.rawVar[varN] for varN in self.FitVars]
        # if lossScale == "max":
        #     lossScale = np.max(data_dict)
        if lossScale is not None:
            loss_fnOnly = lambda: loss_fn()[0] / lossScale
        else:
            loss_fnOnly = lambda: loss_fn()[0]
            lossScale = 1.0
        result_dict = lambda: loss_fn()[1]
        progression_dict = lambda: loss_fn()[2]
        self.Loss = loss_fnOnly

        opt = optimizer(loss_fnOnly, otype=otype, oparam=oparam, NIter=NIter, var_list=FitVars, verbose=verbose)
        opt.optName = otype  # just to store this
        if NIter > 0:
            res = Optimize(opt, loss=loss_fnOnly, lossScale=lossScale)  # self.ResultVals.items()
        else:
            res = loss_fnOnly()
            print("Loss is: " + str(res))
        self.ResultVals = result_dict  # stores how to calculate results
        ResultVals = result_dict()  # calculates the results
        self.Progression = progression_dict
        # Progression = progression_dict()
        self.FitResultVars = {'Loss': res}
        for varN in self.FitVars:
            var = self.Var[varN]
            if callable(var):
                var = var()
            self.FitResultVars[varN] = var.numpy()  # res[n]
        # for varN in Progression:
        #     self.Progression[varN] = Progression[varN]  # res[n]
        for varN in ResultVals:
            self.FitResultVals[varN] = ResultVals[varN]  # .numpy() # res[n]
        return self.FitResultVars, self.FitResultVals

    def findAxis(self, d):
        """
        d can be an axis label or an axis number
        """
        if isinstance(d, str):
            return self.Axes[d]
        else:
            # if d == 0:
            #     raise ValueError("The axes numbers start with one or be negative!")
            if d < 0:
                for axN, ax in self.Axes.items():
                    if ax.curAxis == -d:
                        return ax
            else:
                for axN, ax in self.Axes.items():
                    if ax.curAxis == len(self.Axes) - d:
                        return ax
                # d = -d
            raise ValueError("Axis not found.")

    def findAxesDims(self, listOfAxesNames):
        """
            finds a list of dimension positions from a list of Axis names
        """
        listOfAxes = list(listOfAxesNames)
        return [- self.findAxis(d).curAxis for d in listOfAxesNames]


    def selectDims(self, toPlot, dims=None, includeZero=False):
        """
            selects the dimensions to plot and returns a list of labels
            The result is summed over all the other dimensions
        """
        labels = []
        if dims is None:
            toPlot = np.squeeze(toPlot)
            if toPlot.ndim > 1:
                toPlot = np.sum(toPlot, tuple(range(1, toPlot.ndim)))
        else:
            if not isinstance(dims, list) and not isinstance(dims, tuple):
                dims = list([dims])
            if includeZero:
                rd = list(range(1, toPlot.ndim))  # already exclude the zero axis from being deleted here.
            else:
                rd = list(range(toPlot.ndim))  # choose all dimensions
            for d in dims:
                if d == "time":
                    d = 0
                    labels.append("time")
                else:
                    ax = self.findAxis(d)
                    d = - ax.curAxis
                    labels.append(ax.Labels)
                if d < 0:
                    d = toPlot.ndim + d
                rd.remove(d)
            toPlot = np.sum(toPlot, tuple(rd))
        return toPlot, labels

    def showDates(self, Dates, offsetDay = 0):          # being sunday
        plt.xticks(range(offsetDay, len(Dates), 7), [date for date in Dates[offsetDay:-1:7]], rotation=70)
        # plt.xlim(45, len(Dates))
        plt.tight_layout()

    def showResults(self, title='Results', xlabel='time step', xlim=None, ylim=None, ylabel='probability', dims=None, legendPlacement='upper left', Dates=None, offsetDay=0, logY=True, styles = ['.', '-', ':', '--','-.','*']):
        if logY:
            plot = plt.semilogy
        else:
            plot = plt.plot
        # Plot results
        plt.figure(title)
        plt.title(title)
        legend = []
        n = 0
        # for resN, dict in self.Simulations.items():
        #     style = styles[n]
        #     n+=1
        #     for dictN, toPlot in dict.items():
        #         plt.plot(toPlot, style)
        #         legend.append(resN + "_" + dictN)
        # n=0
        for resN, dict in self.Measurements.items():
            for dictN, toPlot in dict.items():
                toPlot, labels = self.selectDims(toPlot, dims=dims, includeZero=True)
                toPlot = np.squeeze(toPlot)
                plot(toPlot, styles[n])
                if toPlot.ndim > 1:
                    for d in range(toPlot.shape[1]):
                        legend.append(resN + "_" + dictN + "_" + labels[0][d])
                else:
                    legend.append(resN + "_" + dictN)
            plt.gca().set_prop_cycle(None)
            n += 1
        for dictN, toPlot in self.FitResultVals.items():
            toPlot, labels = self.selectDims(toPlot, dims=dims, includeZero=True)
            toPlot = np.squeeze(toPlot)
            plot(toPlot, styles[n])
            if toPlot.ndim > 1:
                for d in range(toPlot.shape[1]):
                    legend.append("Fit_" + dictN + "_" + labels[0][d])
            else:
                legend.append("Fit_" + "_" + dictN)
        plt.legend(legend, loc=legendPlacement)
        if Dates is not None:
            self.showDates(Dates,offsetDay)
        elif xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if xlim is not None:
            plt.xlim(xlim[0],xlim[1])
        if ylim is not None:
            plt.xlim(ylim[0],ylim[1])

    def sumOfStates(self, Progression, sumcoords=None):
        sumStates = 0
        if sumcoords is None:
            sumcoords = tuple(np.arange(self.maxAxes + 1)[1:])
        for name, state in Progression.items():
            sumStates = sumStates + np.sum(state.numpy(), axis=sumcoords)
        return sumStates

    def showStates(self, title='States', exclude={}, xlabel='time step', ylabel='probability', dims=None, dims2d=[0, 1], MinusOne=[], legendPlacement='upper left', Dates = None, offsetDay=0, logY=False):
        if logY:
            plot = plt.semilogy
        else:
            plot = plt.plot

        # Plot the state population
        plt.figure(10)
        plt.title(title)
        legend = []
        if callable(self.Progression):
            Progression = self.Progression()
        else:
            Progression = self.Progression

        sumStates = np.squeeze(self.sumOfStates(Progression, (1,2,3)))
        initState = sumStates[0]
        meanStates = np.mean(sumStates)
        maxDiff = np.max(abs(sumStates - initState))
        print("Sum of states deviates by: " + str(maxDiff) + ", from the starting state:" + str(initState)+". relative: " + str(maxDiff / initState))
        N = 1
        for varN in Progression:
            if varN not in exclude:
                sh = np.array(Progression[varN].shape, dtype=int)
                pdims = np.nonzero(sh > 1)
                toPlot = Progression[varN]  # np.squeeze(
                myLegend = varN
                if np.squeeze(toPlot).ndim > 1:
                    if dims2d is not None:
                        plt.figure(10 + N)
                        plt.ylabel(xlabel)
                        N += 1
                        plt.title("State " + varN)
                        toPlot2, labels = self.selectDims(toPlot, dims=dims2d)
                        toPlot2 = np.squeeze(toPlot2)
                        plt.imshow(toPlot2, aspect="auto")
                        #plt.xlabel(self.RegisteredAxes[self.maxAxes - pdims[0][1]].name)
                        plt.xlabel(dims2d[1])
                        plt.xticks(range(toPlot2.shape[1]), labels[1], rotation=70)
                        plt.colorbar()
                    toPlot, labels = self.selectDims(toPlot, dims=dims)
                    if varN in MinusOne:
                        toPlot = toPlot - 1.0
                        myLegend = myLegend + "-1"
                    myLegend = myLegend + " (summed)"
                plt.figure(10)
                plot(np.squeeze(toPlot))
                legend.append(myLegend)
        plt.legend(legend, loc=legendPlacement)
        if Dates is not None:
            self.showDates(Dates,offsetDay)
        elif xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)

    def compareFit(self, maxPrintSize=10, dims=None, legendPlacement='upper left', Dates = None, offsetDay=0):
        for varN, orig in self.Original.items():
            fit = totensor(removeCallable(self.Var[varN])).numpy()
            if varN not in self.Distorted:
                continue
            dist = self.Distorted[varN]
            orig = self.toVar[varN](orig).numpy() # convert from rawVar to Var
            if isNumber(fit) or np.prod(fit.shape) < maxPrintSize:
                print("Comparison " + varN + ", Distorted:" + str(dist) + ", Original: " + str(orig) + ", fit: " + str(fit) + ", rel. error:" + str(np.max((fit - orig) / orig)))
            else:
                plt.figure("Comparison " + varN)
                dist, labelsD = self.selectDims(dist, dims=dims)
                plt.plot(dist)
                orig, labelsO = self.selectDims(orig, dims=dims)
                plt.plot(orig)
                fit, labelsF = self.selectDims(fit, dims=dims)
                plt.plot(fit)
                plt.legend(["distorted", "original", "fit"], loc=legendPlacement)
                if Dates is not None:
                    self.showDates(Dates,offsetDay)


# --------- Stuff concerning loading data

def getMeasured(params={}):
    param = {'oldFormat': True, 'IndividualLK': True, 'TwoLK': False, 'IndividualAge': True, 'UseGender': False}
    param = {**param, **params}  # overwrite the defaults without destroying them
    results = {}

    if param['oldFormat']:
        dat = retrieveData()  # loads the data from the server
    else:
        pass
        #    dat = data_update_handlers.fetch_data.DataFetcher.fetch_german_data()

    PopTotalLK = dat.groupby(by='IdLandkreis').first()["Bev Insgesamt"]  # .to_dict()  # population of each district
    TPop = np.sum(PopTotalLK)
    # dat = dat[dat.IdLandkreis == 9181];
    if not param['IndividualLK']:
        # dat['AnzahlFall'] = dat.groupby(by='IdLandkreis').sum()['AnzahlFall']
        dat['IdLandkreis'] = 1
        dat['Landkreis'] = 'BRD'
        dat['Bev Insgesamt'] = TPop
        if param['TwoLK']:
            dat2 = retrieveData()
            dat2['IdLandkreis'] = 2
            dat2['Landkreis'] = 'DDR'
            dat['Bev Insgesamt'] = TPop
            dat2['Bev Insgesamt'] = TPop
            dat = pd.concat([dat, dat2], axis=0)
    try:
        NumLK = dat['IdLandkreis'].unique().shape[0]  # number of districts to simulate for (dimension: -3)
        PopTotalLK = dat.groupby(by='IdLandkreis').first()["Bev Insgesamt"]  # .to_dict()  # population of each district
    except:
        NumLK = 1
        PopTotalLK = np.array([82790000])

    if not param['IndividualAge']:
        dat['Altersgruppe'] = 'AllAge'

    results['TPop'] = np.sum(PopTotalLK)  # total population

    # dat['Altersgruppe'].unique().shape[0]
    Pop = 1e6 * np.array([(3.88 + 0.78), 6.62, 2.31 + 2.59 + 3.72 + 15.84, 23.9, 15.49, 7.88, 1.0], CalcFloatStr)
    AgeDist = (Pop / np.sum(Pop))
    results['AgeDist'] = AgeDist

    # Pop = [] # should not be used below
    # LKPopulation *= 82790000 / np.sum(LKPopulation)

    if True:
        LKReported, AllCumulDead, Indices = cumulate(dat)
        if False:
            LKReported = LKReported[20:]
            AllCumulDead = AllCumulDead[20:]
        if not param['UseGender']:  # no sex information
            LKReported = np.sum(LKReported, (-1))  # sum over the sex information for now
            AllCumulDead = np.sum(AllCumulDead, (-1))  # sum over the sex information for now
            results['AllCumulDead'] = AllCumulDead
        AllGermanReported = np.sum(LKReported, (1, 2))  # No age groups, only time
    else:
        dat["Meldedatum"] = pd.to_datetime(dat["Meldedatum"], unit="ms")
        qq = dat.groupby(["Meldedatum"]).aggregate(func="sum")[["AnzahlFall"]].reset_index()
        dat["CumSum"] = np.cumsum(qq['AnzahlFall'])
        results['Daten'] = qq["Meldedatum"]
        AllGermanReported = np.cumsum(qq['AnzahlFall'])

    results['AllGermanReported'] = AllGermanReported
    results['ReportedTimes'] = LKReported.shape[0]
    results['NumAge'] = LKReported.shape[-1]  # represents the age groups according to the RKI

    if not param['IndividualAge']:
        LKPopulation = (PopTotalLK[:, np.newaxis]).astype(CalcFloatStr)
    else:
        LKPopulation = (AgeDist * PopTotalLK[:, np.newaxis]).astype(CalcFloatStr)

    results['LKPopulation'] = LKPopulation
    results['measured'] = LKReported
    results['measured_dead'] = AllCumulDead

    return results
