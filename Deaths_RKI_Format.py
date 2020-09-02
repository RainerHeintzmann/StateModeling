import pandas as pd
import numpy as np
import multiprocessing
import os
from datetime import datetime, timedelta

#global RKI_Data
#global old_format

def getBundesland(landkreis, RKI_Data):
    #global RKI_Data
    return RKI_Data[RKI_Data['Landkreis'] == landkreis]['Bundesland'].iloc[0]

def convertRow(args, old_format, RKI_Data):
    #print(args)
    Datum, Landkreis, Altersgruppe, Geschlecht, Tote = args.to_list()
    #print('args')
    #print(args.to_list())
    refdatum = (datetime.strptime(Datum, '%Y/%m/%d') - datetime(1970, 1, 1)) // timedelta(milliseconds=1)
    index = old_format[old_format['Datum'] == Datum][old_format['Landkreis'] == Landkreis][old_format['Altersgruppe'] == Altersgruppe][old_format['Geschlecht'] == Geschlecht].index.to_list()[0]
    dead_yesterday = (old_format.head(int(index))[old_format['Landkreis'] == Landkreis][old_format['Altersgruppe'] == Altersgruppe][old_format['Geschlecht'] == Geschlecht]).tail(1)['Tote']
    if dead_yesterday.empty:
        diff = Tote
    else:
        diff = Tote - int(dead_yesterday)
    if diff == 0:
        return None
    else:
        return {'AnzahlFall':0, 'AnzahlTodesfall':diff, 'NeuerTodesfall':1,
                'Bundesland':getBundesland(Landkreis, RKI_Data), 'Landkreis':Landkreis, 'Refdatum':refdatum, 'Meldedatum':refdatum, 'Altersgruppe':Altersgruppe, 'Geschlecht':Geschlecht}

def _apply_df(args):
    df, func, kwargs = args
    formatted = df.apply(func, **kwargs)
    ret = pd.DataFrame(columns=['AnzahlFall', 'AnzahlTodesfall', 'NeuerTodesfall', 'Bundesland', 'Landkreis', 'Refdatum', 'Meldedatum', 'Altersgruppe', 'Geschlecht'])
    for row in formatted.iloc:
        try:
            ret = ret.append(row, ignore_index=True)
        except:
            raise TypeError(row)
    return ret

def apply_by_multiprocessing(df, RKI_Data, func, **kwargs):
    kwargs.update({'old_format':df, 'RKI_Data':RKI_Data})
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(result, ignore_index=True)

def reformatDeaths(DataDir=None, NumThreads=12):
    global RKI_Data
    if DataDir is None:
        DataDir = '..' + os.sep + 'RKI-Daten' # files in the data directory

    RKI_Data = pd.read_csv(DataDir + os.sep + 'RKI_COVID19_2020-06-21.csv')
    old_format = pd.read_csv(DataDir + os.sep + 'Deaths.csv')

    #old_format = old_format.head(20)

    rki_format = pd.DataFrame(columns=['AnzahlFall', 'AnzahlTodesfall', 'NeuerTodesfall', 'Bundesland', 'Landkreis', 'Refdatum', 'Meldedatum', 'Altersgruppe', 'Geschlecht'])

    #rki_format = rki_format.append(old_format.apply(convertRow, result_type='expand', axis=1), ignore_index=True)

    #print(apply_by_multiprocessing(old_format, convertRow, axis=1, workers=12))

    rki_format = rki_format.append(apply_by_multiprocessing(old_format, RKI_Data, convertRow, axis=1, workers=NumThreads), ignore_index=True)

    print(rki_format)
    print(rki_format['AnzahlTodesfall'].sum())

    rki_format.to_csv(DataDir + os.sep + 'Deaths_RKI_Format_new.csv', index=False)

if __name__ == '__main__':
    # reformatDeaths(DataDir=r'C:\Users\pi96doc\Documents\Programming\PythonScripts\FromWeb\CoronaData\CSV-Dateien-mit-Covid-19-Infektionen-', NumThreads=8)
    reformatDeaths(r'C:\Users\pi96doc\Documents\Programming\PythonScripts\FromWeb\CoronaData\CSV-Dateien-mit-Covid-19-Infektionen-\\')
