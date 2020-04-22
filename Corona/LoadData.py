import fetch_data
import pandas as pd
import numpy as np
import StateModeling as stm

def loadData(useThuringia = True, pullData=False):

    basePath = r"C:\Users\pi96doc\Documents\Programming\PythonScripts\StateModeling"
    if useThuringia:
        Region = "Thuringia"
        # Thuringia = pd.read_excel(r"C:\Users\pi96doc\Documents\Anträge\Aktuell\COVID_Dickmann_2020\COVID-19 Linelist 2020_04_06.xlsx")
        Thuringia = pd.read_excel(r"C:\Users\pi96doc\Documents\Anträge\Aktuell\COVID_Dickmann_2020\PetraDickmann\COVID-19 Linelist 2020_04_21.xlsx")
        df = pd.read_excel(basePath + r"\Examples\bev_lk.xlsx")  # support information about the population
        AllMeasured = stm.binThuringia(Thuringia, df)
        Hospitalized = AllMeasured['Hospitalized']
    else:
        Region = "Germany"
        if pullData:
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

    # mobility only to 11.04.2020:
    mobility = pd.read_csv(r"C:\Users\pi96doc\Documents\Anträge\Aktuell\COVID_Dickmann_2020\Global_Mobility_Report.csv", low_memory=False)
    mobdat = mobility[mobility['sub_region_1'] == "Thuringia"]
    AllMeasured['mobility'] = mobdat

    AllMeasured['Pop'] = 1e6 * np.array([(3.88 + 0.78), 6.62, 2.31 + 2.59 + 3.72 + 15.84, 23.9, 15.49, 7.88], stm.CalcFloatStr)

    return AllMeasured

def PreprocessData(AllMeasured, CorrectWeekdays=False, ReduceDistricts = (352,342,167,332,399, 278, 403, 82, 230, 55, 251, 102, 221, 122, 223, 110, 263, 80, 240, 330, 3, 276), ReduceAges=None, SumDistricts=False, SumAges=True, SumGender=True):
    # LKs.index('SK Jena'), SK Gera, LK Nordhausen, SK Erfurt, Sk Suhl, LK Weimarer Land, SK Weimar
    # LK Greiz, LK Schmalkalden-Meiningen, LK Eichsfeld, LK Sömmerda, LK Hildburghausen,
    # LK Saale-Orla-Kreis, LK Kyffhäuserkreis, LK Saalfeld-Rudolstadt, LK Ilm-Kreis,
    # LK Unstrut-Hainich-Kreis, LK Gotha, LK Sonneberg, SK Eisenach, LK Altenburger Land, LK Wartburgkreis
    if CorrectWeekdays:
        AllMeasured['Cases'] = stm.correctWeekdayEffect(AllMeasured['Cases'])
        AllMeasured['Dead'] = stm.correctWeekdayEffect(AllMeasured['Dead'])
        AllMeasured['Hospitalized'] = stm.correctWeekdayEffect(AllMeasured['Hospitalized'])

    SelectedGender = slice(0, 2)  # remove the "unknown" part

    if ReduceDistricts is not None:
        # DistrictStride = 50
        # SelectedIDs = slice(0,MeasDetected.shape[1],DistrictStride)
        # IDLabels = LKs[SelectedIDs]
        IDLabels = [LKs[index] for index in ReduceDistricts]
        AllMeasured['CumulCases'] = AllMeasured['CumulCases'][:, SelectedIDs, SelectedAges, SelectedGender]
        AllMeasured['CumulDead'] = AllMeasured['CumulDead'][:, SelectedIDs, SelectedAges, SelectedGender]
        AllMeasured['Cases'] = AllMeasured['Cases'][:, SelectedIDs, SelectedAges, SelectedGender]
        AllMeasured['Dead'] = AllMeasured['Dead'][:, SelectedIDs, SelectedAges, SelectedGender]
        AllMeasured['Hospitalized'] = AllMeasured['Hospitalized'][:, SelectedIDs, SelectedAges, SelectedGender]
        AllMeasured['Cured'] = AllMeasured['Cured'][:, SelectedIDs, SelectedAges, SelectedGender]
        AllMeasured['Population']  = AllMeasured['Population'][:, SelectedIDs, SelectedAges, SelectedGender]
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
