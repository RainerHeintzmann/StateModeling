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
        AllMeasured = binThuringia(Thuringia, df)
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

def stripQuotesFromAxes(data):
    def removeQuote(x):
        if isinstance(x, str):
            return x.replace('"', '')
        else:
            return x
    data = data.applymap(removeQuote)
    new_keys = map(lambda ax: ax.strip('"'),data.keys())
    for ax, newax in zip(data.keys(), new_keys):
        data.rename(columns={ax: newax}, inplace=True)
    return data

def binThuringia(data, df):
    #import locale
    #locale.setlocale(locale.LC_ALL, 'de_DE')
    whichDate = 'Erkrankungsbeginn'
    # data = data.sort_values(whichDate)
    data = stripQuotesFromAxes(data)
    if 'AbsonderungEnde' in data.keys():
        AbsonderungEnde = 'AbsonderungEnde'
    else:
        AbsonderungEnde = 'AbsonderungBis'
    # replace empty Erkrankungsbeginn with Meldedatum
    data['Erkrankungsbeginn'] = data[['Erkrankungsbeginn', 'Meldedatum']].apply(lambda x: x[1] if x[0]=="" else x[0], axis=1)
    data['AbsonderungEnde'] = pd.to_datetime(data[AbsonderungEnde].str.replace('"', '').str[:10], dayfirst=True)
    data['Meldedatum'] = pd.to_datetime(data['Meldedatum'])
    data['Erkrankungsbeginn'] = pd.to_datetime(data['Erkrankungsbeginn'])
    data['AbsonderungVon'] = pd.to_datetime(data['AbsonderungVon'])
    data['VerstorbenDatum'] = pd.to_datetime(data['VerstorbenDatum'])
    data['AlterBerechnet'] = pd.to_numeric(data['AlterBerechnet'])
    data['InterneRef'] = pd.to_numeric(data['InterneRef'])
    day1 = np.min(data[whichDate])
    dayLast0 = np.max(data['Meldedatum'] - day1)
    dayLast1 = np.max(data['Erkrankungsbeginn'] - day1)
    dayLast2 = np.max(data['VerstorbenDatum'] - day1)
    dayLast3 = np.max(data['AbsonderungEnde'] - day1)
    dayLast = np.max([dayLast0, dayLast1, dayLast2])  # , dayLast3 : AbsonderungEnde is not used to fill in data, as it is often beyond the current date
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
        if myAge < 0:
            print('unknown age.' + str(myAge)+'... skipping ...')
            continue
        Cases[myday, myLK, myAge, myGender] += 1.0
        myCuredDay = (row['AbsonderungEnde'] - day1).days
        if myCuredDay is not np.nan and myCuredDay < Cured.shape[0]:
            Cured[myCuredDay, myLK, myAge, myGender] += 1
        if row['HospitalisierungStatus'] == "Ja":
            Hospitalized[myday, myLK, myAge, myGender] += 1
        myDeadDay = (row['VerstorbenDatum'] - day1).days
        if myDeadDay is not np.nan:
            Dead[myDeadDay, myLK, myAge, myGender] += 1
    Dates = pd.date_range(start = day1, periods=numDays).map(lambda x: x.strftime('%d.%m.%Y'))

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


def preprocessData(AllMeasured, CorrectWeekdays=False, ReduceDistricts=('LK Greiz', 'SK Gera', 'SK Jena'), ReduceAges=None, ReduceGender = slice(0, 2), SumDistricts=False, SumAges=True, SumGender=True):
    # LKs.index('SK Jena'), SK Gera, LK Nordhausen, SK Erfurt, Sk Suhl, LK Weimarer Land, SK Weimar
    # LK Greiz, LK Schmalkalden-Meiningen, LK Eichsfeld, LK Sömmerda, LK Hildburghausen,
    # LK Saale-Orla-Kreis, LK Kyffhäuserkreis, LK Saalfeld-Rudolstadt, LK Ilm-Kreis,
    # LK Unstrut-Hainich-Kreis, LK Gotha, LK Sonneberg, SK Eisenach, LK Altenburger Land, LK Wartburgkreis
    if CorrectWeekdays:
        AllMeasured['Cases'] = correctWeekdayEffect(AllMeasured['Cases'])
        AllMeasured['Dead'] = correctWeekdayEffect(AllMeasured['Dead'])
        AllMeasured['Hospitalized'] = correctWeekdayEffect(AllMeasured['Hospitalized'])

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

