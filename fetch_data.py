# This code is from the HackCorona project and was written largely by Manuel Lang
import pandas as pd
from datetime import date, timedelta, datetime
from dotenv import load_dotenv
import json
import urllib
import urllib.request
from configparser import ConfigParser
import requests
import os
import numpy as np

# von : https://www.arcgis.com/home/item.html?id=dd4580c810204019a7b8eb3e0b329dd6
# Beschreibung der Daten des RKI Covid-19-Dashboards (https://corona.rki.de)
#
# Dem Dashboard liegen aggregierte Daten der gemäß IfSG von den Gesundheitsämtern an das RKI übermittelten Covid-19-Fälle zu Grunde
# Mit den Daten wird der tagesaktuelle Stand (00:00 Uhr) abgebildet und es werden die Veränderungen bei den Fällen und Todesfällen zum Vortag dargstellt
# In der Datenquelle sind folgende Parameter enthalten:
# IdBundesland: Id des Bundeslands des Falles mit 1=Schleswig-Holstein bis 16=Thüringen
# Bundesland: Name des Bundeslanes
# Landkreis ID: Id des Landkreises des Falles in der üblichen Kodierung 1001 bis 16077=LK Altenburger Land
# Landkreis: Name des Landkreises
# Altersgruppe: Altersgruppe des Falles aus den 6 Gruppe 0-4, 5-14, 15-34, 35-59, 60-79, 80+ sowie unbekannt
# Geschlecht: Geschlecht des Falles M0männlich, W=weiblich und unbekannt
# AnzahlFall: Anzahl der Fälle in der entsprechenden Gruppe
# AnzahlTodesfall: Anzahl der Todesfälle in der entsprechenden Gruppe
# Meldedatum: Datum, wann der Fall dem Gesundheitsamt bekannt geworden ist
# Datenstand: Datum, wann der Datensatz zuletzt aktualisiert worden ist
# NeuerFall:
# 0: Fall ist in der Publikation für den aktuellen Tag und in der für den Vortag enthalten
# 1: Fall ist nur in der aktuellen Publikation enthalten
# -1: Fall ist nur in der Publikation des Vortags enthalten
# damit ergibt sich: Anzahl Fälle der aktuellen Publikation als Summe(AnzahlFall), wenn NeuerFall in (0,1); Delta zum Vortag als Summe(AnzahlFall) wenn NeuerFall in (-1,1)
# NeuerTodesfall:
# 0: Fall ist in der Publikation für den aktuellen Tag und in der für den Vortag jeweils ein Todesfall
# 1: Fall ist in der aktuellen Publikation ein Todesfall, nicht jedoch in der Publikation des Vortages
# -1: Fall ist in der aktuellen Publikation kein Todesfall, jedoch war er in der Publikation des Vortags ein Todesfall
# -9: Fall ist weder in der aktuellen Publikation noch in der des Vortages ein Todesfall
# damit ergibt sich: Anzahl Todesfälle der aktuellen Publikation als Summe(AnzahlTodesfall) wenn NeuerTodesfall in (0,1); Delta zum Vortag als Summe(AnzahlTodesfall) wenn NeuerTodesfall in (-1,1)
# Referenzdatum: Erkrankungsdatum bzw. wenn das nicht bekannt ist, das Meldedatum
# AnzahlGenesen: Anzahl der Genesenen in der entsprechenden Gruppe
# NeuGenesen:
# 0: Fall ist in der Publikation für den aktuellen Tag und in der für den Vortag jeweils Genesen
# 1: Fall ist in der aktuellen Publikation Genesen, nicht jedoch in der Publikation des Vortages
# -1: Fall ist in der aktuellen Publikation nicht Genesen, jedoch war er in der Publikation des Vortags Genesen
# -9: Fall ist weder in der aktuellen Publikation noch in der des Vortages Genesen
# damit ergibt sich: Anzahl Genesen der aktuellen Publikation als Summe(AnzahlGenesen) wenn NeuGenesen in (0,1); Delta zum Vortag als Summe(AnzahlGenesen) wenn NeuGenesen in (-1,1)

def get_date_range(dfs):
    min_dates = []
    for _, df in dfs.items():
        min_dates.append(date.fromisoformat(df.index.values.min()))
    min_date = min(min_dates)

    dates = []
    for i in range((date.today() - min_date).days + 1):
        dates.append((min_date + timedelta(days=i)).isoformat())

    return dates


class DataFetcher:
    @staticmethod
    def fetch_french_data():
        pass

    @staticmethod
    def fetch_german_data():
        start_date = date(2020, 1, 1)
        end_date = date.today() + timedelta(days=1)
        dfs = []
        for single_date in pd.date_range(start_date, end_date):
            print("readin: "+str(single_date))
            with urllib.request.urlopen(
                    "https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0//query?where=Meldedatum%3D%27" + str(
                            single_date.strftime(
                                    "%Y-%m-%d")) + "%27&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token=") as url:
                json_data = json.loads(url.read().decode())["features"]
            json_data = [x["attributes"] for x in json_data if "attributes" in x]
            if len(json_data) > 0:
                dfs.append(pd.DataFrame(json_data))
        return pd.concat(dfs)

    @staticmethod
    def fetch_italian_data():
        load_dotenv()

        df = pd.read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv")
        df.sort_values(by="data", inplace=True)
        return df

    @staticmethod
    def fetch_swiss_data():
        load_dotenv()
        parser = ConfigParser()
        parser.read("../covid19-cases-switzerland/sources.ini")
        cantons = list(map(str.upper, parser['cantonal']))

        dfs = {}
        for canton in cantons:
            dfs[canton] = (
                pd.read_csv(parser["cantonal"][canton.lower()]).groupby(["date"]).max()
            )

        # Append empty dates to all
        dates = get_date_range(dfs)

        df_cases = pd.DataFrame(float("nan"), index=dates, columns=cantons)
        df_fatalities = pd.DataFrame(float("nan"), index=dates, columns=cantons)
        df_hospitalized = pd.DataFrame(float("nan"), index=dates, columns=cantons)
        df_icu = pd.DataFrame(float("nan"), index=dates, columns=cantons)
        df_vent = pd.DataFrame(float("nan"), index=dates, columns=cantons)
        df_released = pd.DataFrame(float("nan"), index=dates, columns=cantons)

        for canton, df in dfs.items():
            for d in dates:
                if d in df.index:
                    df_cases[canton][d] = df["ncumul_conf"][d]
                    df_fatalities[canton][d] = df["ncumul_deceased"][d]
                    df_hospitalized[canton][d] = df["ncumul_hosp"][d]
                    df_icu[canton][d] = df["ncumul_ICU"][d]
                    df_vent[canton][d] = df["ncumul_vent"][d]
                    df_released[canton][d] = df["ncumul_released"][d]

        # Fill to calculate the correct totals for CH
        cantons = df_cases.columns

        canton_geo = {
            "AG": [47.3887506709017, 8.04829959908319],
            "AI": [47.3366249625858, 9.41305434728638],
            "AR": [47.382878725937, 9.27443680962626],
            "BE": [46.948021, 7.44821],
            "BL": [47.4854119772813, 7.72572990842221],
            "BS": [47.559662, 7.58053],
            "CH": [46.8095957, 7.103256],
            "FR": [46.8041803910762, 7.15974364171403],
            "GE": [46.2050579, 6.126579],
            "GL": [47.0209899551849, 8.97917017584858, ],
            "GR": [46.8521107310644, 9.53013993916933],
            "JU": [47.1711613, 7.7298485],
            "LU": [47.0471358210592, 8.32521536164421],
            "NE": [46.9947407, 6.8725506],
            "NW": [46.95725, 8.365905],
            "OW": [46.8984476369729, 8.17648286868192],
            "SG": [47.4221404863031, 9.37594858130343],
            "SH": [47.697472, 8.63223],
            "SO": [47.2083728153487, 7.53011089227976],
            "SZ": [47.0234391221245, 8.67474512239957],
            "TG": [47.5539632, 8.8730003],
            "TI": [46.1999320324997, 9.02266750657928],
            "UR": [46.886774, 8.634912],
            "VD": [46.552043140147, 6.65230780339362],
            "VS": [46.225759, 7.3302457],
            "ZG": [47.1499939485641, 8.52330213578886],
            "ZH": [47.369091, 8.53801]
        }

        df_cases.columns = df_cases.columns + "_cases"
        df_fatalities.columns = df_fatalities.columns + "_fatalities"
        df_hospitalized.columns = df_hospitalized.columns + "_hospitalized"
        df_icu.columns = df_icu.columns + "_icu"
        df_released.columns = df_released.columns + "_released"

        df_merged_temp = df_cases.merge(df_fatalities, how='outer', left_index=True, right_index=True)
        df_merged_temp = df_merged_temp.merge(df_hospitalized, how='outer', left_index=True, right_index=True)
        df_merged_temp = df_merged_temp.merge(df_icu, how='outer', left_index=True, right_index=True)
        df_merged_temp = df_merged_temp.merge(df_released, how='outer', left_index=True, right_index=True)
        df_merged_temp.reset_index(inplace=True)
        df_merged_temp.fillna(method='ffill', inplace=True)
        df_merged_temp.fillna(0, inplace=True)
        df_merged_temp = df_merged_temp.rename(columns={'index': 'date'})

        data = []
        for row_i in range(df_merged_temp.shape[0]):
            row = df_merged_temp.iloc[row_i]
            for canton in cantons:
                d = {"canton": canton}
                if not canton == "CH":
                    for key in ["cases", "fatalities", "hospitalized", "icu", "released"]:
                        d[key] = row[f"{canton}_{key}"]
                    d["geo_coordinates_2d"] = canton_geo[canton]
                    d["date"] = row["date"]
                    data.append(d)

        return pd.DataFrame(data)

    @staticmethod
    def fetch_us_data():
        path = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
        df = pd.read_csv(path)
        df["City"] = df["Admin2"]
        df.drop(columns=["Admin2"], inplace=True)
        df.sort_values("City", inplace=True)
        df = df[["UID", "FIPS", "City", "Province_State", "Lat", "Long_"] + [x for x in df.columns if "/" in x]]
        return df


if __name__ == "__main__":
    DataFetcher.fetch_us_data()

