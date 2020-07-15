import pandas as pd
import os
from datetime import datetime, timedelta

# does a linear interpolation between to given values over a given amount of days
def linear_interpolation(value: int, days=1, previous=0) -> list:
    daily = (value-previous)/(days+1)
    ret = []
    for i in range(1, days+1):
        ret.append(int((daily*i)+previous))
    return ret

# orders a dataframe by firstly date and then district
def reorder_dataframe_by_date_and_district(df, start_date, end_date, districts):
    ret = pd.DataFrame(columns=['Datum', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'Tote'])
    #print(df)
    #print(start_date)
    #print(end_date)
    #print(districts)
    for i in range(0, ((end_date - start_date).days + 1)):
        #print('hi')
        for district in districts:
            the_day = datetime.strftime(start_date + timedelta(days=i), '%Y/%m/%d')
            #print(the_day)
            #print(type(the_day))
            append_df = df[df['Datum'] == the_day][df['Landkreis'] == district]
            if append_df.empty:
                continue
            #print('append_df: ', append_df)
            ret = ret.append(append_df, ignore_index=True)
            #print(ret)
    #print(ret)
    return ret


def PreprocessDeaths(DataDir=None):
    if DataDir is None:
        DataDir = '..' + os.sep + 'RKI-Daten' # files in the data directory
    files = os.listdir(DataDir)  # files in the data directory

    #column_order = ('IdBundesland', 'Bundesland', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'AnzahlFall', 'AnzahlTodesfall', 'Meldedatum', 'IdLandkreis', 'Datenstand', 'NeuerFall', 'NeuerTodesfall')
    #column_list = pd.read_csv('..' + os.sep + 'RKI-Daten' + os.sep + 'RKI_COVID19_2020-03-27.csv').columns.to_list()
    #column_list.remove('ObjectId')
    # removes elements in the files list which should not be loaded
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
    try:
        files.remove('Deaths_RKI_Format.csv')
    except:
        pass
    try:
        files.remove('Deaths_RKI_Format_new.csv')
    except:
        pass
    try:
        files.remove('RKI_COVID19_2020-04-16.csv')
    except:
        pass
    """
    try:
        files.remove('RKI_COVID19_2020-04-11.csv')
    except:
        pass
    try:
        files.remove('RKI_COVID19_2020-04-13.csv')
    except:
        pass
    try:
        files.remove('RKI_COVID19_2020-04-18.csv')
    except:
        pass
    try:
        files.remove('RKI_COVID19_2020-04-27.csv')
    except:
        pass
    try:
        files.remove('RKI_COVID19_2020-05-04.csv')
    except:
        pass
    """

    files = sorted(files) # orders files by data date rather than last modification

    # DEBUG
    print(files)

    # reads district list out of recent data
    last_data = pd.read_csv(DataDir+ os.sep + files[-1])
    landkreise = []
    for landkreis in last_data['Landkreis']:
        if not landkreis in landkreise:
            landkreise.append(landkreis)

    # possible age specifications in the data
    ageGroups = ['A00-A04', 'A05-A14', 'A15-A34', 'A35-A59', 'A60-A79', 'A80+', 'unbekannt']

    # possible gender specifications in the data
    genders = ['M', 'unbekannt', 'W']

    newDeaths = pd.DataFrame(columns=['Datum', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'Tote']) #format: Datum, Landkreis, Altersgruppe, Geschlecht, Tote
    append_today_DataFrame = pd.DataFrame(columns=['Datum', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'Tote'])
    append_yesterday_DataFrame = pd.DataFrame(columns=['Datum', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'Tote'])
    append_inter_DataFrame = pd.DataFrame(columns=['Datum', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'Tote'])

    prev_date = datetime.strptime('2020/03/25', '%Y/%m/%d').date()

    for file in files:
        print(file)
        data = pd.read_csv(DataDir + os.sep + file, encoding = "ISO-8859-1") # encoding to also deal with problematic codes see data of 29.4.2020
        NeuerTodesfallTag = 'NeuerTodesfall'
        if NeuerTodesfallTag not in data.keys():
            NeuerTodesfallTag = 'Neuer Todesfall'
        AnzahlTodesfallTag = 'AnzahlTodesfall'
        if AnzahlTodesfallTag not in data.keys():
            AnzahlTodesfallTag = 'Anzahl Todesfall'
        data = data[data[NeuerTodesfallTag] != -9]
        data_date = file[-14:-4]  # get the date information ("Datenstand") directly from the filename. This is cheating a bit but much easier to program...
        # data_date = data['Datenstand'].iloc[0]
        data_date = data_date.replace('-', '/')
        # data_date = data_date.replace('.', '/')
        # print(data_date)
        format = '%Y/%m/%d'
        # if data_date[0:4] != '2020':
        #     format = '%d/%m/%Y'
        # data_date_obj = datetime.strptime(data_date[0:10], format).date()
        data_date_obj = datetime.strptime(data_date, format).date()
        lack_of_data = (data_date_obj - prev_date).days - 1
        if lack_of_data:
            yesterday = datetime.strftime(data_date_obj - timedelta(days=1), '%Y/%m/%d')
            lack_of_data2 = lack_of_data - 1
            print('Lack of data 2:', lack_of_data2)
        else:
            print('Lack of data 1:', lack_of_data)

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
                    interest = interest_gender[interest_gender[NeuerTodesfallTag] != -1]
                    if interest.empty:
                        continue
                    dead = interest[AnzahlTodesfallTag].sum()
                    #print(dead)
                    append_dict = {'Datum':data_date_obj.strftime('%Y/%m/%d'), 'Landkreis':current_district, 'Altersgruppe':age, 'Geschlecht':gender, 'Tote':dead}
                    #print(append_dict)
                    append_today_DataFrame = append_today_DataFrame.append(append_dict, ignore_index=True)
                    #print(newDeaths)
                    if lack_of_data:
                        interest = interest_gender[interest_gender[NeuerTodesfallTag] != 0]
                        diff = interest[AnzahlTodesfallTag].sum()
                        dead_yesterday = dead - diff
                        append_dict = {'Datum':yesterday, 'Landkreis':current_district, 'Altersgruppe':age, 'Geschlecht':gender, 'Tote':dead_yesterday}
                        append_yesterday_DataFrame = append_yesterday_DataFrame.append(append_dict, ignore_index=True)
                    if lack_of_data2:
                        #print('Lack of data 2', lack_of_data2)
                        gap_date = prev_date + timedelta(days=1)
                        if prev_date == datetime.strptime('2020/02/24', '%Y/%m/%d').date():
                            interpolation = linear_interpolation(dead_yesterday, days=30)
                        else:
                            prev_dead = newDeaths[newDeaths['Datum'] == prev_date.strftime('%Y/%m/%d')][newDeaths['Landkreis'] == current_district][newDeaths['Altersgruppe'] == age][newDeaths['Geschlecht'] == gender]['Tote']
                            if prev_dead.empty:
                                continue
                            prev_dead = int(prev_dead)
                            interpolation = linear_interpolation(dead_yesterday, lack_of_data2, prev_dead)
                        for i in range(0, len(interpolation)):
                            if interpolation[i] == 0:
                                continue
                            append_dict = {'Datum':datetime.strftime(gap_date, '%Y/%m/%d'), 'Landkreis':current_district, 'Altersgruppe':age, 'Geschlecht':gender, 'Tote':interpolation[i]}
                            append_inter_DataFrame = append_inter_DataFrame.append(append_dict, ignore_index=True)
                            gap_date = gap_date + timedelta(days=1)

        if lack_of_data2:
            print(append_inter_DataFrame)
            append_inter_DataFrame = reorder_dataframe_by_date_and_district(append_inter_DataFrame, prev_date + timedelta(days=1), datetime.strptime(yesterday, '%Y/%m/%d').date() - timedelta(days=1), landkreise)
            print(append_inter_DataFrame)
            newDeaths = newDeaths.append(append_inter_DataFrame, ignore_index=True)
            print(newDeaths)
            append_inter_DataFrame = pd.DataFrame(columns=['Datum', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'Tote'])
        if lack_of_data:
            newDeaths = newDeaths.append(append_yesterday_DataFrame, ignore_index=True)
            append_yesterday_DataFrame = pd.DataFrame(columns=['Datum', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'Tote'])
        lack_of_data2 = 0
        newDeaths = newDeaths.append(append_today_DataFrame, ignore_index=True)
        append_today_DataFrame = append_yesterday_DataFrame
        #print(data_date_obj)
        #print(type(data_date_obj))
        #print(prev_date)
        #print(type(prev_date))
        prev_date = data_date_obj

    #newDeaths = newDeaths.sort_values(by=['Datum'])
    newDeaths.to_csv(DataDir + os.sep + 'Deaths.csv', index=False)
    print(newDeaths)

if __name__ == '__main__':
    # PreprocessDeaths(r'C:\Users\pi96doc\Documents\Programming\PythonScripts\FromWeb\CoronaData\CSV-Dateien-mit-Covid-19-Infektionen-')
    PreprocessDeaths(r'..\FromWeb\CoronaData\CSV-Dateien-mit-Covid-19-Infektionen-')
