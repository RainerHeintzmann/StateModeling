import pandas as pd
import os
from datetime import datetime, timedelta

def interpolation(days, value, previous=0):
    daily = (value-previous)/(days+1)
    ret = []
    for i in range(1, days+1):
        ret.append(int((daily*i)+previous))
    return ret


files = os.listdir('..' + os.sep + 'RKI-Daten')
column_order = ('IdBundesland', 'Bundesland', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'AnzahlFall', 'AnzahlTodesfall', 'Meldedatum', 'IdLandkreis', 'Datenstand', 'NeuerFall', 'NeuerTodesfall')
column_list = pd.read_csv('..' + os.sep + 'RKI-Daten' + os.sep + 'RKI_COVID19_2020-03-27.csv').columns.to_list()
column_list.remove('ObjectId')
try:
    files.remove('.git')
except:
    pass
try:
    files.remove('README.md')
except:
    pass
try:
    files.remove('Format.txt')
except:
    pass
try:
    files.remove('Deaths.csv')
except:
    pass

files = sorted(files)
# DEBUG
print(files)

last_data = pd.read_csv('..' + os.sep + 'RKI-Daten' + os.sep + 'RKI_COVID19_2020-06-14.csv')

landkreise = []

for landkreis in last_data['Landkreis']:
    if not landkreis in landkreise:
        landkreise.append(landkreis)

ageGroups = ['A00-A04', 'A05-A14', 'A15-A34', 'A35-A59', 'A60-A79', 'A80+', 'unbekannt']

genders = ['M', 'W', 'unbekannt']
newDeaths = pd.DataFrame(columns=['Datum', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'Tote']) #format: Datum, Landkreis, Altersgruppe, Geschlecht, Tote
append_today_DataFrame = pd.DataFrame(columns=['Datum', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'Tote']) #format: Datum, Landkreis, Altersgruppe, Geschlecht, Tote
append_yesterday_DataFrame = pd.DataFrame(columns=['Datum', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'Tote']) #format: Datum, Landkreis, Altersgruppe, Geschlecht, Tote

prev_date = datetime.strptime('2020/02/24', '%Y/%m/%d').date()

for file in files:
    print(file)
    data = pd.read_csv('..' + os.sep + 'RKI-Daten' + os.sep + file)
    data = data[data['NeuerTodesfall'] != -9]
    data_date = data['Datenstand'].iloc[0]
    data_date = data_date.replace('-', '/')
    print(data_date)
    data_date_obj = datetime.strptime(data_date, '%Y/%m/%d').date()
    lack_of_data = (data_date_obj - prev_date).days - 1
    prev_date = data_date_obj
    if lack_of_data:
        print(lack_of_data - 1)
        yesterday = datetime.strftime(data_date_obj - timedelta(days=1), '%Y/%m/%d')
    else:
        print(lack_of_data)

    for current_district in landkreise:
        interest_district = data[data['Landkreis'] == current_district]
        if interest_district.empty:
            continue
        for age in ageGroups:
            interest_age = interest_district[interest_district['Altersgruppe'] == age]
            if interest_age.empty:
                continue
            for gender in genders:
                interest_gender = interest_age[interest_age['Geschlecht'] == gender]
                if interest_gender.empty:
                    continue
                interest = interest_gender[interest_gender['NeuerTodesfall'] != -1]
                if interest.empty:
                    continue
                dead = interest['AnzahlTodesfall'].sum()
                #print(dead)
                append_dict = {'Datum':data_date, 'Landkreis':current_district, 'Altersgruppe':age, 'Geschlecht':gender, 'Tote':dead}
                #print(append_dict)
                append_today_DataFrame = append_today_DataFrame.append(append_dict, ignore_index=True)
                #print(newDeaths)
                if lack_of_data:
                    interest = interest_gender[interest_gender['NeuerTodesfall'] != 0]
                    diff = interest['AnzahlTodesfall'].sum()
                    dead_yesterday = dead - diff
                    append_dict = {'Datum':yesterday, 'Landkreis':current_district, 'Altersgruppe':age, 'Geschlecht':gender, 'Tote':dead_yesterday}
                    append_yesterday_DataFrame = append_yesterday_DataFrame.append(append_dict, ignore_index=True)

    if lack_of_data:
        newDeaths = newDeaths.append(append_yesterday_DataFrame, ignore_index=True)
        append_yesterday_DataFrame = pd.DataFrame(columns=['Datum', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'Tote'])
    newDeaths = newDeaths.append(append_today_DataFrame, ignore_index=True)
    append_today_DataFrame = append_yesterday_DataFrame

newDeaths.to_csv('..' + os.sep + 'RKI-Daten' + os.sep + 'Deaths.csv', index=False)
print(newDeaths)
