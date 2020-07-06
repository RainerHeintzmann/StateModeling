import pandas as pd
import numpy as np
import multiprocessing
import os
from datetime import datetime, timedelta

global RKI_Data
RKI_Data = pd.read_csv('..' + os.sep + 'RKI-Daten' + os.sep + 'RKI_COVID19_2020-06-21.csv')

# This file currently does not deal with the problem of incorrect deaths sometimes not being reverted in the raw data available to us.
# But, said problem is only minor in magnitude (0.2% error).
# We do of course have the intention of fixing it but because of its minor role it's not our first priority.

def getBundesland(landkreis):
    return RKI_Data[RKI_Data['Landkreis'] == landkreis]['Bundesland'].iloc[0]

def isPersistent(date, district, age, gender, dead):
    tomorrow = (datetime.strptime(date, '%Y/%m/%d') + timedelta(days=1)).strftime('%Y/%m/%d')
    dead_tomorrow = ref[ref['Datum'] == tomorrow][ref['Landkreis'] == district][ref['Altersgruppe'] == age][ref['Geschlecht'] == gender]['Tote']
    if dead_tomorrow.empty:
        return 0 - dead
    return int(dead_tomorrow) - dead

def convertRow(args):
    #print(args)
    Datum, Landkreis, Altersgruppe, Geschlecht, Tote = args.to_list()
    #print('args')
    #print(args.to_list())
    refdatum = (datetime.strptime(Datum, '%Y/%m/%d') - datetime(1970, 1, 1)) // timedelta(milliseconds=1)
    """
    yesterday = (datetime.strptime(Datum, '%Y/%m/%d') - timedelta(days=1)).strftime('%Y/%m/%d')
    the_day_before_yesterday = (datetime.strptime(Datum, '%Y/%m/%d') - timedelta(days=2)).strftime('%Y/%m/%d')
    dead_yesterday = ref[ref['Datum'] == yesterday][ref['Landkreis'] == Landkreis][ref['Altersgruppe'] == Altersgruppe][ref['Geschlecht'] == Geschlecht]['Tote']
    if dead_yesterday.empty:
        dead_the_day_before_yesterday = ref[ref['Datum'] == the_day_before_yesterday][ref['Landkreis'] == Landkreis][ref['Altersgruppe'] == Altersgruppe][ref['Geschlecht'] == Geschlecht]['Tote']
        if dead_the_day_before_yesterday.empty:
            the_day_before_yesterday = (datetime.strptime(Datum, '%Y/%m/%d') - timedelta(days=3)).strftime('%Y/%m/%d')
            dead_the_day_before_yesterday = ref[ref['Datum'] == the_day_before_yesterday][ref['Landkreis'] == Landkreis][ref['Altersgruppe'] == Altersgruppe][ref['Geschlecht'] == Geschlecht]['Tote']
            if dead_the_day_before_yesterday.empty:
                the_day_before_yesterday = (datetime.strptime(Datum, '%Y/%m/%d') - timedelta(days=4)).strftime('%Y/%m/%d')
                dead_the_day_before_yesterday = ref[ref['Datum'] == the_day_before_yesterday][ref['Landkreis'] == Landkreis][ref['Altersgruppe'] == Altersgruppe][ref['Geschlecht'] == Geschlecht]['Tote']
                if dead_the_day_before_yesterday.empty:
                    diff = Tote
                else:
                    diff = Tote - int(dead_the_day_before_yesterday)
            else:
                diff = Tote - int(dead_the_day_before_yesterday)
        else:
            diff = Tote - int(dead_the_day_before_yesterday)
    else:
        diff = Tote - int(dead_yesterday)
    """
    index = ref[ref['Datum'] == Datum][ref['Landkreis'] == Landkreis][ref['Altersgruppe'] == Altersgruppe][ref['Geschlecht'] == Geschlecht].index.to_list()[0]
    dead_yesterday = (ref.head(int(index))[ref['Landkreis'] == Landkreis][ref['Altersgruppe'] == Altersgruppe][ref['Geschlecht'] == Geschlecht]).tail(1)['Tote']
    #diff_tomorrow = isPersistent(Datum, Landkreis, Altersgruppe, Geschlecht, Tote)
    if dead_yesterday.empty:
        diff = Tote
    else:
        diff = Tote - int(dead_yesterday)
    if diff == 0:
        return None
    else:
        return {'AnzahlFall':0, 'AnzahlTodesfall':diff, 'NeuerTodesfall':1,
                'Bundesland':getBundesland(Landkreis), 'Landkreis':Landkreis, 'Refdatum':refdatum, 'Meldedatum':refdatum, 'Altersgruppe':Altersgruppe, 'Geschlecht':Geschlecht}
"""
    elif diff_tomorrow < 0:
        ret = pd.DataFrame(data={'AnzahlFall':0, 'AnzahlTodesfall':diff, 'NeuerTodesfall':1, 'Bundesland':getBundesland(Landkreis),
            'Landkreis':Landkreis, 'Refdatum':refdatum, 'Meldedatum':refdatum, 'Altersgruppe':Altersgruppe, 'Geschlecht':Geschlecht}, index=[0])
        refdatum += (1000 * 60 * 60 * 24)
        ret2 = pd.DataFrame(data={'AnzahlFall':0, 'AnzahlTodesfall':diff_tomorrow, 'NeuerTodesfall':-1,
            'Bundesland':getBundesland(Landkreis), 'Landkreis':Landkreis, 'Refdatum':refdatum,
            'Meldedatum':refdatum, 'Altersgruppe':Altersgruppe, 'Geschlecht':Geschlecht}, index=[0])
        return (ret, ret2)
"""
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
        try:
            ret = ret.append(row, ignore_index=True)
        except:
            raise TypeError(row)
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
ref = pd.read_csv('..' + os.sep + 'RKI-Daten' + os.sep + 'Deaths.csv')

#old_format = old_format.head(20)

#old_format =

rki_format = pd.DataFrame(columns=['AnzahlFall', 'AnzahlTodesfall', 'NeuerTodesfall', 'Bundesland', 'Landkreis', 'Refdatum', 'Meldedatum', 'Altersgruppe', 'Geschlecht'])

#rki_format = rki_format.append(old_format.apply(convertRow, result_type='expand', axis=1), ignore_index=True)

#print(apply_by_multiprocessing(old_format, convertRow, axis=1, workers=12))

rki_format = rki_format.append(apply_by_multiprocessing(old_format, convertRow, axis=1, workers=12), ignore_index=True)

print(rki_format)
print(rki_format['AnzahlTodesfall'].sum())

rki_format.to_csv('..' + os.sep + 'RKI-Daten' + os.sep + 'Deaths_RKI_Format_new.csv', index=False)
