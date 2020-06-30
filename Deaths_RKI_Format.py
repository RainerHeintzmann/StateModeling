import pandas as pd
import numpy as np
import multiprocessing
import os
from datetime import datetime, timedelta

global RKI_Data
RKI_Data = pd.read_csv('..' + os.sep + 'RKI-Daten' + os.sep + 'RKI_COVID19_2020-06-21.csv')

def getBundesland(landkreis):
    return RKI_Data[RKI_Data['Landkreis'] == landkreis]['Bundesland'].iloc[0]

def convertRow(args):
    #print(args)
    Datum, Landkreis, Altersgruppe, Geschlecht, Tote = args.to_list()
    #print('args')
    #print(args.to_list())
    refdatum = (datetime.strptime(Datum, '%Y/%m/%d') - datetime(1970, 1, 1)) // timedelta(milliseconds=1)
    yesterday = (datetime.strptime(Datum, '%Y/%m/%d') - timedelta(days=1)).strftime('%Y/%m/%d')
    dead_yesterday = old_format[old_format['Datum'] == yesterday][old_format['Landkreis'] == Landkreis][old_format['Altersgruppe'] == Altersgruppe][old_format['Geschlecht'] == Geschlecht]['Tote']
    if dead_yesterday.empty:
        diff = 1
    else:
        diff = Tote - int(dead_yesterday)
    #return [0, diff, 1, getBundesland(Landkreis), Landkreis, refdatum, refdatum, Altersgruppe, Geschlecht]
    return {'AnzahlFall':0, 'AnzahlTodesfall':diff, 'NeuerTodesfall':1 if diff > 0 else -9, 'Bundesland':getBundesland(Landkreis), 'Landkreis':Landkreis, 'Refdatum':refdatum, 'Meldedatum':refdatum, 'Altersgruppe':Altersgruppe, 'Geschlecht':Geschlecht}

"""
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, 11)
    print(df_split)
    pool = Pool(11)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
"""

def _apply_df(args):
    df, func, kwargs = args
    formatted = df.apply(func, **kwargs)
    ret = pd.DataFrame(columns=['AnzahlFall', 'AnzahlTodesfall', 'NeuerTodesfall', 'Bundesland', 'Landkreis', 'Refdatum', 'Meldedatum', 'Altersgruppe', 'Geschlecht'])
    for row in formatted.iloc:
        ret = ret.append(row, ignore_index=True)
    return ret

def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    #print(type(result[0]))
    #print(result)
    return pd.concat(result, ignore_index=True)

old_format = pd.read_csv('..' + os.sep + 'RKI-Daten' + os.sep + 'Deaths.csv')

#old_format = old_format.head(20)

rki_format = pd.DataFrame(columns=['AnzahlFall', 'AnzahlTodesfall', 'NeuerTodesfall', 'Bundesland', 'Landkreis', 'Refdatum', 'Meldedatum', 'Altersgruppe', 'Geschlecht'])

#rki_format = rki_format.append(old_format.apply(convertRow, result_type='expand', axis=1), ignore_index=True)

#print(apply_by_multiprocessing(old_format, convertRow, axis=1, workers=12))

rki_format = rki_format.append(apply_by_multiprocessing(old_format, convertRow, axis=1, workers=12), ignore_index=True)

print(rki_format)

rki_format.to_csv('..' + os.sep + 'RKI-Daten' + os.sep + 'Deaths_RKI_Format_new.csv', index=False)
