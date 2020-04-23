import fetch_data
import pandas as pd
import numpy as np
import StateModeling as stm

def loadData(filename = None, useThuringia = True, pullData=False):

    if useThuringia:
        if filename is None:
            filename = r"COVID-19 Linelist 2020_04_22.xlsx"
        basePath = r"C:\Users\pi96doc\Documents\Anträge\Aktuell\COVID_Dickmann_2020\PetraDickmann"
        # Thuringia = pd.read_excel(r"C:\Users\pi96doc\Documents\Anträge\Aktuell\COVID_Dickmann_2020\COVID-19 Linelist 2020_04_06.xlsx")
        Thuringia = pd.read_excel(basePath + '\\'+ filename)
        basePath = r"C:\Users\pi96doc\Documents\Programming\PythonScripts\StateModeling"
        df = pd.read_excel(basePath + r"\Examples\bev_lk.xlsx")  # support information about the population
        AllMeasured = stm.binThuringia(Thuringia, df)
        AllMeasured['Region'] = "Thuringia"
    else:
        basePath = r"C:\Users\pi96doc\Documents\Programming\PythonScripts\StateModeling"
        if pullData:
            data = fetch_data.DataFetcher().fetch_german_data()
            # with open(r"C:\Users\pi96doc\Documents\Anträge\Aktuell\COVID_Dickmann_2020\Global_Mobility_Report.csv", 'r', encoding="utf8") as f:
            #     mobility = list(csv.reader(f, delimiter=","))
            # mobility = np.array(mobility[1:], dtype=np.float)

            df = pd.read_excel(basePath + r"\Examples\bev_lk.xlsx")  # support information about the population
            AllMeasured = stm.cumulate(data, df)
            np.save(basePath + r'\Data\AllMeasured', AllMeasured)

            # can be checked with
            # https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Situationsberichte/2020-04-16-de.pdf?__blob=publicationFile
        else:
            AllMeasured = np.load(basePath + r'\Data\AllMeasured.npy', allow_pickle=True).item()
        AllMeasured['Region'] = "Germany"
    AgePop = np.array([(3.88 + 0.78), 6.62, 2.31 + 2.59 + 3.72 + 15.84, 23.9, 15.49, 7.88], stm.CalcFloatStr)
    AgePop /= np.sum(AgePop)
    PopM = AgePop[np.newaxis,:] * AllMeasured['PopM'][:,np.newaxis]
    PopW = AgePop[np.newaxis,:] * AllMeasured['PopW'][:,np.newaxis]
    AllMeasured['Population'] = np.stack((PopM, PopW),-1)

    # mobility only to 11.04.2020:
    mobility = pd.read_csv(r"C:\Users\pi96doc\Documents\Anträge\Aktuell\COVID_Dickmann_2020\Global_Mobility_Report.csv", low_memory=False)
    mobdat = mobility[mobility['sub_region_1'] == "Thuringia"]
    AllMeasured['mobility'] = mobdat

    return AllMeasured

def preprocessData(AllMeasured, CorrectWeekdays=False, ReduceDistricts=('LK Greiz', 'SK Gera', 'SK Jena'), ReduceAges=None, ReduceGender = slice(0, 2), SumDistricts=False, SumAges=True, SumGender=True):
    # LKs.index('SK Jena'), SK Gera, LK Nordhausen, SK Erfurt, Sk Suhl, LK Weimarer Land, SK Weimar
    # LK Greiz, LK Schmalkalden-Meiningen, LK Eichsfeld, LK Sömmerda, LK Hildburghausen,
    # LK Saale-Orla-Kreis, LK Kyffhäuserkreis, LK Saalfeld-Rudolstadt, LK Ilm-Kreis,
    # LK Unstrut-Hainich-Kreis, LK Gotha, LK Sonneberg, SK Eisenach, LK Altenburger Land, LK Wartburgkreis
    if CorrectWeekdays:
        AllMeasured['Cases'] = stm.correctWeekdayEffect(AllMeasured['Cases'])
        AllMeasured['Dead'] = stm.correctWeekdayEffect(AllMeasured['Dead'])
        AllMeasured['Hospitalized'] = stm.correctWeekdayEffect(AllMeasured['Hospitalized'])

    if ReduceDistricts == 'Thuringia':
        ReduceDistricts = (352, 342, 167, 332, 399, 278, 403, 82, 230, 55, 251, 102, 221, 122, 223, 110, 263, 80, 240, 330, 3, 276)
    elif ReduceDistricts == 'Model Regions':
        ReduceDistricts = ['SK Gera', 'SK Jena', 'LK Greiz', 'SK Suhl', 'LK Nordhausen']
    elif isinstance(ReduceDistricts, str):
        ValueError('Unknown String for reduce districts: ' + ReduceDistricts + '. Use a list or tuple if this is a single district.')

    if isinstance(ReduceDistricts,list) or isinstance(ReduceDistricts,tuple) and isinstance(ReduceDistricts[0], str):
        allDist = []
        for name in ReduceDistricts:
            if name in AllMeasured['LKs']:
                allDist.append(AllMeasured['LKs'].index(name))
            else:
                raise ValueError('District name '+name+' is not present in data')
        ReduceDistricts = allDist

    if ReduceDistricts is None:
        ReduceDistricts = slice(None,None,None) # means take all
    else:
        AllMeasured['IDs'] = [AllMeasured['IDs'][index] for index in ReduceDistricts]  # IDs[0:-1:DistrictStride]
        AllMeasured['LKs'] = [AllMeasured['LKs'][index] for index in ReduceDistricts]

    if ReduceAges is None:
        ReduceAges = slice(None,None,None) # means take all
    if ReduceGender is None:
        ReduceGender = slice(None,None,None) # means take all


    sumDims = ()
    if SumGender:
        AllMeasured['Gender'] = ['All Genders']
        sumDims  = sumDims+(-1,)
    if SumAges:
        AllMeasured['Ages'] = ['summed Ages']
        sumDims  = sumDims+(-2,)
    if SumDistricts:
        AllMeasured['IDs'] = np.array(0)
        AllMeasured['LKs'] = [AllMeasured['Region']]
        sumDims  = sumDims+(-3,)

    AllMeasured['CumulCases'] = np.sum(AllMeasured['CumulCases'][:, ReduceDistricts, ReduceAges, ReduceGender], sumDims, keepdims=True)
    AllMeasured['CumulDead'] = np.sum(AllMeasured['CumulDead'][:, ReduceDistricts, ReduceAges, ReduceGender], sumDims, keepdims=True)
    AllMeasured['Cases'] = np.sum(AllMeasured['Cases'][:, ReduceDistricts, ReduceAges, ReduceGender], sumDims, keepdims=True)
    AllMeasured['Dead'] = np.sum(AllMeasured['Dead'][:, ReduceDistricts, ReduceAges, ReduceGender], sumDims, keepdims=True)
    AllMeasured['Hospitalized'] = np.sum(AllMeasured['Hospitalized'][:, ReduceDistricts, ReduceAges, ReduceGender], sumDims, keepdims=True)
    AllMeasured['Cured'] = np.sum(AllMeasured['Cured'][:, ReduceDistricts, ReduceAges, ReduceGender], sumDims, keepdims=True)
    AllMeasured['Population']  = np.sum(AllMeasured['Population'][ReduceDistricts, ReduceAges, ReduceGender], sumDims, keepdims=True)
    return AllMeasured
