import pandas as pd
import os

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

# DEBUG
print(files)

newDeaths = pd.DataFrame()
additionalDeaths = pd.DataFrame()

for file in files:
    print(file)
    data = pd.read_csv('..' + os.sep + 'RKI-Daten' + os.sep + file, encoding='utf-8')
    print(data.columns)
    append_value = data[data['NeuerTodesfall'] == 1]
    append_value2 = data[data['AnzahlTodesfall'] > 1]
    newDeaths = newDeaths.append(append_value[column_list])
    additionalDeaths = additionalDeaths.append(append_value2[column_list])

print(newDeaths)
