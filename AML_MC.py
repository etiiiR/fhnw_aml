# ---
# title: "AML Challenge 2024"
# date: today
# author: "Etienne Roulet, Alexander Shanmugam"
# format: 
#     html:
#         toc: true
#         number_sections: true
#         page-layout: full
# editor_options: 
#     chunk_output_type: console
# bibliography: references.bib
# ---

# %% [markdown]
# ### Todos
#
# - 14. Quantitatives Untersuchen der Unterschiede von Top-N Kunden-Listen verschiedener Modelle.
# - 17. Beschreiben des Mehrwerts des “finalen” Modelles in der Praxis.
#
# - Markdown writing and explain what we have done
# - Clean up code: (Imports in central place, Comments, Code Refactor)
#

# %% [markdown]
# # Setup
# Die folgenden Code-Blöcke können genutzt werden, um die benötigten Abhängigkeiten zu installieren und zu importieren.

# %%
# %%capture
# %pip install -r ../requirements.txt

# %%
from itables import init_notebook_mode

init_notebook_mode()

# %%
# Laden der eingesetzten Libraries
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from IPython.display import display
from sklearn.linear_model import LinearRegression

import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    roc_curve,
    auc,
    make_scorer,
    confusion_matrix,
    ConfusionMatrixDisplay,
    fbeta_score,
)
import lime.lime_tabular

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# %%
# %%capture
# set theme ggplot for plots
plt.style.use("ggplot")
# set display options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# %%
# Funktion zur Bestimmung des Geschlechts und Berechnung des Geburtstags
def parse_details(birth_number):
    birth_number_str = str(
        birth_number
    )  # Konvertiere birth_number zu einem String, falls notwendig
    year_prefix = "19"
    month = int(birth_number_str[2:4])
    gender = "female" if month > 12 else "male"
    if gender == "female":
        month -= 50
    year = int(year_prefix + birth_number_str[:2])
    day = int(birth_number_str[4:6])
    birth_day = datetime(year, month, day)
    return gender, birth_day


# Berechnung des Alters basierend auf einem Basisjahr
def calculate_age(birth_date, base_date=datetime(1999, 12, 31)):
    return (
        base_date.year
        - birth_date.year
        - ((base_date.month, base_date.day) < (birth_date.month, birth_date.day))
    )


# Regression metrics
def regression_results(y_true, y_pred):
    print(
        "explained_variance: ",
        round(metrics.explained_variance_score(y_true, y_pred), 4),
    )
    print(
        "mean_squared_log_error: ",
        round(metrics.mean_squared_log_error(y_true, y_pred), 4),
    )
    print("r2: ", round(metrics.r2_score(y_true, y_pred), 4))
    print("MAE: ", round(metrics.mean_absolute_error(y_true, y_pred), 4))
    print("MSE: ", round(metrics.mean_squared_error(y_true, y_pred), 4))
    print("RMSE: ", round(np.sqrt(metrics.mean_squared_error(y_true, y_pred)), 4))


# %% [markdown]
# # Aufgabenstellung
# Inhalt der hier bearbeiteten und dokumentierten Mini-Challenge für das Modul «aml - Angewandtes Machine Learning» der FHNW ist die Entwicklung und Evaluierung von Aﬀinitätsmodellen für personalisierte Kreditkarten-Werbekampagnen im Auftrag einer Bank. Das Ziel der Authoren ist es also, mithilfe von Kunden- und Transaktionsdaten präzise Modelle zu erstellen, die die Wahrscheinlichkeit des Kreditkartenkaufs einer bestimmten Person vorhersagen.

# %% [markdown]
# # Laden der zur Verfügung gestellten Daten
# [//]: # (-.- .tabset)
#
# Zur Verfügung gestellt wurden 8 csv-Dateien von welchen die Beschreibung der erfassten Variablen unter dem folgenden Link eingesehen werden können: [PKDD'99 Discovery Challenge - Guide to the Financial Data Set](https://sorry.vse.cz/~berka/challenge/PAST/index.html). Nachfolgend werden diese csv-Dateien eingelesen.

# %%
account = pd.read_csv("./data/account.csv", sep=";", dtype={"date": "str"})
card = pd.read_csv("./data/card.csv", sep=";", dtype={"issued": "str"})
client = pd.read_csv("./data/client.csv", sep=";")
disp = pd.read_csv("./data/disp.csv", sep=";")
district = pd.read_csv("./data/district.csv", sep=";")
loan = pd.read_csv("./data/loan.csv", sep=";", dtype={"date": "str"})
order = pd.read_csv("./data/order.csv", sep=";")
trans = pd.read_csv("./data/trans.csv", sep=";", dtype={"date": "str", "bank": "str"})

# %% [markdown]
# # Datenaufbereitung & Explorative Datenanalyse
# Im folgenden Abschnitt werden die geladenen Daten separat so transformiert, dass jede Zeile einer Observation und jede Spalte einer Variable im entsprechenden Datenformat entspricht, also ins Tidy-Format gebracht.

# %%
data_frames = {}

# %% [markdown]
# ## Account
# [//]: # (-.- .tabset)
# Der Datensatz `accounts.csv` beinhaltet 4500 Observationen mit den folgenden Informationen über die Kontos der Bank: 
#    
# - `account_id`: die Kontonummer,  
# - `district_id`: den Standort der entsprechenden Bankfiliale,  
# - `frequency`: die Frequenz der Ausstellung von Kontoauszügen (monatlich, wöchentlich, pro Transaktion) und  
# - `date`: das Erstellungsdatum

# %%
account.info()

# %%
print("Anzahl fehlender Werte:", sum(account.isnull().sum()))
print("Anzahl duplizierter Einträge:", account.duplicated().sum())

# %% [markdown]
# ### Aufbereitung
# Nachfolgend wird die `date` Spalte des `account.csv`-Datensatzes in das entsprechende Datenformat geparsed und die Werte von `frequency` übersetzt und als Levels einer Kategorie definiert.

# %%
# parse date
account["date"] = pd.to_datetime(account["date"], format="%y%m%d")
# translate categories
account["frequency"] = account["frequency"].replace(
    {
        "POPLATEK MESICNE": "monthly",
        "POPLATEK TYDNE": "weekly",
        "POPLATEK PO OBRATU": "transactional",
    }
)

# convert column frequency to categorical
account["frequency"] = account["frequency"].astype("category")

# sample 5 random rows
account.sample(n=5)

# %% [markdown]
# ### Distrikt
# Hier zu sehen ist die Verteilung der Distrikte pro Bankkonto. Ersichtlich ist, dass im Distrikt 1 mit Abstand am meisten Bankkontos geführt werden. Die darauf folgenden Distrikte bewegen sich alle im Bereich zwischen ~250 - 50 Bankkonten.  

# %%
# plot the distribution of the district_ids and replace the id with it's name
plt.figure(figsize=(15, 6))
account["district_id"].value_counts().plot(kind="bar")
plt.title("Verteilung der Distrikte")
plt.xlabel("Distrikt")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# ### Frequenz
# Auf dieser Visualisierung zu sehen ist die Klassenverteilung der Frequenz der Ausstellung der Kontoauszüge. Die allermeisten Bankkonten besitzen eine monatliche Ausstellung.

# %%
# Verteilung der Frequenz visualisieren
plt.figure(figsize=(10, 6))
account["frequency"].value_counts().plot(kind="bar")
plt.title("Frequenz der Kontoauszüge")
plt.xlabel("Frequenz")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# ### Datum
# Der hier dargestellte Plot zeigt die Verteilung der Kontoerstellungsdaten. Das erste Konto wurde im Jahr 1993 und das neuste im 1998 erstellt.  

# %%
# plot date distribution
plt.figure(figsize=(10, 6))
plt.hist(account["date"], bins=20)
plt.title("Verteilung der Kontoerstellungsdaten")
plt.xlabel("Datum")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# ### Korrelation & weitere Informationen
# Die Korrelation sowie weitere Informationen zu den vorhandenen Daten können aus dem [SweetViz Report](./reports/accounts.html) entnommen werden. 

# %%
# append account data to dataframe collection
data_frames["account.csv"] = account

# %%
# # %%capture
# # generate sweetviz report
# svReport_account = sv.analyze(account)
# svReport_account.show_html(filepath="./reports/accounts.html", open_browser=False)

# %% [markdown]
# ## Card
# [//]: # (-.- .tabset)
#
# Der Datensatz `card.csv` beinhaltet 892 Observationen mit den folgenden Informationen über die von der Bank herausgegebenen Kreditkarten:  
#
# - `card_id`: die Kartennummer,  
# - `disp_id`: die Zuordnung zum entsprechenden Bankkonto und -inhaber (Disposition),  
# - `type`: die Art der Kreditkarte (junior, classic, gold) und  
# - `issued`: das Ausstellungsdatum

# %%
card.info()

# %%
print("Anzahl fehlender Werte:", sum(card.isnull().sum()))
print("Anzahl duplizierter Einträge:", card.duplicated().sum())

# %% [markdown]
# ### Aufbereitung
# Auch bei diesem Datensatz (`card.csv`) werden zunächst die Datentypen korrigiert um anschliessend die Inhalte entsprechend beschreiben zu können

# %%
# parse date
card["issued"] = pd.to_datetime(card["issued"].str[:6], format="%y%m%d")
card["issued"] = card["issued"].dt.to_period("M")
# convert type to categorical
card["type"] = card["type"].astype("category")

card.sample(n=5)

# %% [markdown]
# ### Kartentyp
# Hier dargestellt ist die Klassenverteilung der Kartentypen. Die meisten Karteninhaber besitzen eine klassische Kreditkarte, gefolgt von ~180 junior- und ~100 gold Karten.  

# %%
# plot distribution of type
plt.figure(figsize=(10, 6))
card["type"].value_counts().plot(kind="bar")
plt.title("Verteilung der Kartentypen")
plt.xlabel("Kartentyp")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# ### Ausstellungsdatum
# Hier dargestellt ist die Häufigkeit von Kreditkartenausstellungen pro Monat. Erkennbar ist eine steigende Tendenz mit einem Rückgang in den Monaten Februar - April 1997.  

# %%
# plot issued date per month and year
plt.figure(figsize=(15, 6))
card["issued"].value_counts().sort_index().plot(kind="bar")
plt.title("Verteilung der Ausstellungsdaten")
plt.xlabel("Datum")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# ### Korrelation & weitere Informationen
# Die Korrelation sowie weitere Informationen zu den vorhandenen Daten können aus dem [SweetViz Report](./reports/card.html) entnommen werden.

# %%
# append to dataframes collection
data_frames["card.csv"] = card

# %%
# # %%capture
# # generate sweetviz report
# svReport_card = sv.analyze(card)
# svReport_card.show_html(filepath="./reports/card.html", open_browser=False)

# %% [markdown]
# ## Client
# [//]: # (-.- .tabset)
#
# Der Datensatz `client.csv` beinhaltet 5369 Observationen mit den folgenden Informationen über die Kunden der Bank:  
#
# - `client_id`: die Kundennummer,  
# - `birth_number`: eine Kombination aus Geburtsdatum und Geschlecht sowie  
# - `district_id`: die Adresse   

# %%
client.info()

# %%
print("Anzahl fehlender Werte:", sum(client.isnull().sum()))
print("Anzahl duplizierter Einträge:", client.duplicated().sum())

# %% [markdown]
# ### Aufbereitung
# Die Spalte `birth_number` des `client.csv`-Datensatzes codiert 3 Features der Bankkunden: Geschlecht, Geburtsdatum und damit auch das Alter. Diese Informationen werden mithilfe der zuvor definierten Funktionen `parse_details()` und `calculate_age` extrahiert.

# %%
# Geburtstag & Geschlecht aus birth_number extrahieren
client["gender"], client["birth_day"] = zip(
    *client["birth_number"].apply(parse_details)
)
client["gender"] = client["gender"].astype("category")
# Alter berechnen
client["age"] = client["birth_day"].apply(calculate_age)

# Spalte birth_number entfernen
client = client.drop(columns=["birth_number"])

# Sample 5 random rows
client.sample(n=5)

# %% [markdown]
# ### Geschlecht
# Hier dargestellt ist die Verteilung des Geschlechts der Bankkunden. Das Geschlecht der erfassten Bankkunden ist fast gleichverteilt mit einem etwas kleineren Frauenanteil.  

# %%
# plot distribution of gender
plt.figure(figsize=(10, 6))
gender_distribution = client["gender"].value_counts().plot(kind="bar")
plt.title("Verteilung des Geschlechts der Bankkunden")
plt.xlabel("Geschlecht")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# ### Alter
# Nachfolgend abgebildet ist die Verteilung des Alters der Bankkunden. Die jüngste erfasste Person ist 12 Jahre alt und die älteste 88. 

# %%
# plot distribution of age
plt.figure(figsize=(10, 6))
client["age"].plot(kind="hist", bins=20)
plt.title("Verteilung des Alters der Bankkunden")
plt.xlabel("Alter")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# ### Korrelation & weitere Informationen
# Die Korrelation sowie weitere Informationen zu den vorhandenen Daten können aus dem [SweetViz Report](./reports/client.html) entnommen werden.

# %%
data_frames["client.csv"] = client

# %%
# # %%capture
# svReport_client = sv.analyze(client)
# svReport_client.show_html(filepath="./reports/client.html", open_browser=False)

# %% [markdown]
# ## Disp
# [//]: # (-.- .tabset)
#
# Der Datensatz `disp.csv` beinhaltet 5369 Observationen mit den folgenden Informationen über die Dispositionen der Bank:   
#
# - `disp_id`: der Identifikationsschlüssel der Disposition,  
# - `client_id`: die Kundennummer,  
# - `account_id`: die Kontonummer,  
# - `type`: die Art der Disposition (Inhaber, Benutzer)

# %%
disp.info()

# %%
print("Anzahl fehlender Werte:", sum(disp.isnull().sum()))
print("Anzahl duplizierter Einträge:", disp.duplicated().sum())

# %% [markdown]
# ### Aufbereitung
# Auch die Variablen des Datensatzes `disp.csv` werden in die korrekten Datentypen übertragen. 

# %%
# Spalte type als Kategorie speichern
disp["type"] = disp["type"].astype("category")

# random sample
disp.sample(n=5)

# %% [markdown]
# ### Typ der Disposition
# Hier dargestellt ist die Verteilung der Art der Dispositionen. 4500 Kunden sind Inhaber eines Kontos und 896 sind Disponenten. 

# %%
# plot distribution of kind
plt.figure(figsize=(10, 6))
disp["type"].value_counts().plot(kind="bar")
plt.title("Verteilung der Dispositionen")
plt.xlabel("Disposition")
plt.ylabel("Anzahl")
plt.show()

# %%
# remove disponents
disp = disp[disp["type"] == "OWNER"]

# %% [markdown]
# ### Korrelation & weitere Informationen
# Die Korrelation sowie weitere Informationen zu den vorhandenen Daten können aus dem [SweetViz Report](./reports/disp.html) entnommen werden.

# %%
data_frames["disp.csv"] = disp

# %%
# # %%capture
# svReport_disp = sv.analyze(disp)
# svReport_disp.show_html(filepath="./reports/disp.html", open_browser=False)

# %% [markdown]
# ## District
# [//]: # (-.- .tabset)
#
# Der Datensatz `district.csv` beinhaltet 77 Observationen mit den folgenden demografischen Informationen:   
#
# - `A1`: die ID des Distrikts,  
# - `A2`: der Name des Distrikts,  
# - `A3`: die Region,  
# - `A4`: die Anzahl der Einwohner,  
# - `A5`: die Anzahl der Gemeinden mit < 499 Einwohner,  
# - `A6`: die Anzahl der Gemeinden mit 500 - 1999 Einwohner,  
# - `A7`: die Anzahl der Gemeinden mit 2000 - 9999 Einwohner,  
# - `A8`: die Anzahl der Gemeinden mit >10000 Einwohner,  
# - `A9`: die Anzahl Städte,  
# - `A10`: das Verhältnis von städtischen Einwohnern,  
# - `A11`: das durchschnittliche Einkommen,  
# - `A12`: die Arbeitslosenrate vom Jahr 95,  
# - `A13`: die Arbeitslosenrate vom Jahr 96,  
# - `A14`: die Anzahl von Unternehmer pro 1000 Einwohner,  
# - `A15`: die Anzahl von begangenen Verbrechen im Jahr 95,  
# - `A16`: die Anzahl von begangenen Verbrechen im Jahr 96,   

# %%
district.info()

# %%
print("Anzahl fehlender Werte:", sum(district.isnull().sum()))
print("Anzahl duplizierter Einträge:", district.duplicated().sum())

# %% [markdown]
# ### Aufbereitung
# Zunächst werden die Spaltennamen in sprechendere übersetzt.

# %%
# Spalten umbenennen
district = district.rename(
    columns={
        "A1": "district_id",
        "A2": "district_name",
        "A3": "region",
        "A4": "num_of_habitat",
        "A5": "num_of_small_town",
        "A6": "num_of_medium_town",
        "A7": "num_of_big_town",
        "A8": "num_of_bigger_town",
        "A9": "num_of_city",
        "A10": "ratio_of_urban",
        "A11": "average_salary",
        "A12": "unemploy_rate95",
        "A13": "unemploy_rate96",
        "A14": "n_of_enterpren_per1000_inhabit",
        "A15": "no_of_crimes95",
        "A16": "no_of_crimes96",
    }
)[
    [
        "district_id",
        "district_name",
        "region",
        "num_of_habitat",
        "num_of_small_town",
        "num_of_medium_town",
        "num_of_big_town",
        "num_of_bigger_town",
        "num_of_city",
        "ratio_of_urban",
        "average_salary",
        "unemploy_rate95",
        "unemploy_rate96",
        "n_of_enterpren_per1000_inhabit",
        "no_of_crimes95",
        "no_of_crimes96",
    ]
]

district["region"] = district["region"].astype("category")
district["district_name"] = district["district_name"].astype("category")

# %% [markdown]
# Auffällig ist, dass nebst den Spalten `A2` (dem Namen) und `A3` (der Region) die Spalten `A12` und `A15` den Datentyp `object` erhalten. Das ist, weil jeweils ein fehlender Wert vorhanden ist, welcher mit einem `?` gekennzeichnet ist. 

# %%
# die fehlenden Werte anzeigen
district[district.isin(["?"]).any(axis=1)]

# %% [markdown]
# Wir gehen davon aus, dass es sich hier um effektiv fehlende Werte handelt und nicht um zensierte Daten, also Werte, für welche der exakte Wert fehlt, aber trotzdem Informationen vorhanden sind. In diesem Fall, wenn die Variable mit den fehlenden Werten eine hohe Korrelation mit anderen Prediktoren aufweist, bietet es sich an, KNN oder eine einfache lineare Regression für die Imputation anzuwenden. [@brancoSurveyPredictiveModeling2017] 
#
# Die Korrelationsmatrix des [SweetViz Reports](./reports/district.html) zeigt, dass `unemploy_rate95` stark mit `unemploy_rate96` und `no_of_crimes95` mit `no_of_crimes96` korreliert. 

# %%
# die ? ersetzen mit NaN
district = district.replace("?", np.nan)

# Datentyp korrigieren
district["no_of_crimes95"] = district["no_of_crimes95"].astype(float)
district["unemploy_rate95"] = district["unemploy_rate95"].astype(float)

# %%
# Korrelation zwischen Arbeitslosenquote 95 und 96
district[["unemploy_rate95", "unemploy_rate96"]].corr()

# %%
# Korrelation zwischen Anzahl Verbrechen 95 und 96
district[["no_of_crimes95", "no_of_crimes96"]].corr()

# %% [markdown]
# Demnach werden nachfolgend zwei lineare Regressions-Modelle trainiert, um die fehlenden Werte zu imputieren.

# %%
# Zeilen filtern, sodass keine fehlenden Werte vorhanden sind
district_no_na = district[district["unemploy_rate95"].notnull()]

# Lineares regressions Modell erstellen
lin_reg_unemploy = LinearRegression()

# Modell fitten
lin_reg_unemploy.fit(
    district_no_na["unemploy_rate96"].values.reshape(-1, 1),
    district_no_na["unemploy_rate95"].values,
)

# Modell evaluieren
regression_results(
    district_no_na["unemploy_rate95"],
    lin_reg_unemploy.predict(district_no_na["unemploy_rate96"].values.reshape(-1, 1)),
)

# %% [markdown]
# Der $R^2$ Wert von $0.9634$ versichert, damit ein stabiles Modell für die Imputation erreicht zu haben. 

# %%
# Lineares regressions Modell erstellen
lin_reg_crime = LinearRegression()

# Modell fitten
lin_reg_crime.fit(
    district_no_na["no_of_crimes96"].values.reshape(-1, 1),
    district_no_na["no_of_crimes95"].values,
)

# Modell evaluieren
regression_results(
    district_no_na["no_of_crimes95"],
    lin_reg_crime.predict(district_no_na["no_of_crimes96"].values.reshape(-1, 1)),
)

# %% [markdown]
# Auch hier mit einem $R^2$ Wert von $0.9969$ gehen wir davon aus, damit ein stabiles Modell für die Imputation erreicht zu haben. Somit werden nachfolgend die beiden Modelle genutzt, um die fehlenden Werte einzufüllen.

# %%
# Vorhersage der fehlenden Werte
district.loc[district["no_of_crimes95"].isnull(), "no_of_crimes95"] = (
    lin_reg_crime.predict(
        district[district["no_of_crimes95"].isnull()]["no_of_crimes96"].values.reshape(
            -1, 1
        )
    )
)

district.loc[district["unemploy_rate95"].isnull(), "unemploy_rate95"] = (
    lin_reg_unemploy.predict(
        district[district["unemploy_rate95"].isnull()][
            "unemploy_rate96"
        ].values.reshape(-1, 1)
    )
)

# %%
district.sample(n=5)

# %%
district.isnull().sum()

# %% [markdown]
# ### EDA
# Es gibt keine Duplikate und somit 77 unterschiedliche Namen der Distrikte. Diese sind auf 8 Regionen verteilt, wobei die meisten in south Moravia und die wenigsten in Prague liegen. Der Distrikt mit den wenigsten Einwohnern zählt 42821, im Vergleich zu demjenigen mit den meisten: 1204953, wobei die nächst kleinere Ortschaft 102609 Einwohner zählt. Weitere Informationen zu den vorhandenen Daten können aus dem [SweetViz Report](./reports/district.html) entnommen werden. 

# %%
data_frames["district.csv"] = district

# %%
# # %%capture
# svReport_district = sv.analyze(district)
# svReport_district.show_html(filepath="./reports/district.html", open_browser=False)

# %% [markdown]
# ## Loan
# [//]: # (-.- .tabset)
#
# Der Datensatz `loan.csv` beinhaltet 682 Observationen mit den folgenden Informationen über die vergebenen Darlehen der Bank:  
#
# - `loan_id`: ID des Darlehens,  
# - `account_id`: die Kontonummer,  
# - `date`: das Datum, wann das Darlehen gewährt wurde,  
# - `amount`: der Betrag,  
# - `duration`: die Dauer des Darlehens,  
# - `payments`: die höhe der monatlichen Zahlungen und  
# - `status`: der Rückzahlungsstatus (A: ausgeglichen, B: Vertrag abgelaufen aber nicht fertig bezahlt, C: laufender Vertrag und alles in Ordnung, D: laufender Vertrag und Kunde verschuldet)
#

# %%
loan.info()

# %%
print("Anzahl fehlender Werte:", sum(loan.isnull().sum()))
print("Anzahl duplizierter Einträge:", loan.duplicated().sum())

# %% [markdown]
# ### Aufbereitung
# Auch für den `loan.csv` Datensatz werden zunächst Datenformate korrigiert und Kategorien übersetzt. Anschliessend wird überprüft, ob ein Bankkonto mehrere Darlehen besitzt.  

# %%
# Datum parsen
loan["date"] = pd.to_datetime(loan["date"], format="%y%m%d")

# Kategorien übersetzen
loan["status"] = loan["status"].map(
    {
        "A": "contract finished",
        "B": "finished contract, loan not paid",
        "C": "running contract",
        "D": "client in debt",
    }
)

loan["status"] = loan["status"].astype("category")

# %%
# Anzahl der Darlehen pro Kontonummer berechnen
num_of_loan_df = (
    loan.groupby("account_id")
    .size()
    .reset_index(name="num_of_loan")
    .sort_values(by="num_of_loan", ascending=False)
)

# %%
# Überprüfen, ob jedes Konto nur ein Darlehen hat
num_of_loan_df["num_of_loan"].value_counts()

# %% [markdown]
# Von allen Bankkontos, die ein Darlehen aufgenommen haben, hat jedes Konto genau ein Darlehen zugewiesen.

# %%
# Sample 5 random rows from the joined DataFrame
display(loan.sample(n=5))

# %% [markdown]
# ### Ausstellungsdatum
# Nachfolgend dargestellt ist die Verteilung der Darlehensausstellungsdaten. das erste Darlehen wurde im Juli 1993 ausgestellt und das neuste im Dezember 1998. 

# %%
# plot distribution of date
plt.figure(figsize=(15, 6))
loan["date"].dt.to_period("M").value_counts().sort_index().plot(kind="bar")
plt.title("Verteilung der Darlehensausstellungsdaten")
plt.xlabel("Datum")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# ### Dauer
# Hier ersichtlich ist die Verteilung der Dauer der Darlehen. Sie ist fast gleichverteilt über die 5 möglichen Optionen. 

# %%
# plot duration distribution
plt.figure(figsize=(10, 6))
loan["duration"].value_counts().plot(kind="bar")
plt.title("Verteilung der Darlehensdauer")
plt.xlabel("Dauer")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# ### Betrag
# Hier dargestellt ist die Verteilung der Darlehensbeträge. Nur wenige Darlehensbeträge sind höher als 400000 wobei die meisten um die 100000 betragen. 

# %%
# plot amount
plt.figure(figsize=(10, 6))
loan["amount"].plot(kind="hist", bins=20)
plt.title("Verteilung der Darlehensbeträge")
plt.xlabel("Betrag")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# ### Status
# Der nachfolgende Plot zeigt die Klassenverteilung vom Darlehensstatus. Die meisten (~400) sind laufend und ok, rund 200 sind abgeschlossen, die Kunden von ~50 Darlehen sind verschuldet und etwas weniger wurden abgeschlossen, ohne fertig abbezahlt worden zu sein.  

# %%
# plot status distribution
plt.figure(figsize=(10, 6))
loan["status"].value_counts().plot(kind="bar")
plt.title("Verteilung der Darlehensstatus")
plt.xlabel("Status")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# ### Zahlungen
# Hier ersichtlich ist die Verteilung der monatlichen Zahlungen der Darlehen. Die kleinste monatliche Zahlung beträgt 304 und die höchste 9910.

# %%
# plot payments
plt.figure(figsize=(10, 6))
loan["payments"].plot(kind="hist", bins=20)
plt.title("Verteilung der monatlichen Zahlungen")
plt.xlabel("Zahlungen")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# ### Korrelation & weitere Informationen
# Die Korrelation sowie weitere Informationen zu den vorhandenen Daten können aus dem [SweetViz Report](./reports/loan.html) entnommen werden.

# %%
# Assign the resulting DataFrame to a dictionary for storage
data_frames["loan.csv"] = loan

# %%
# # %%capture
# svReport_loan = sv.analyze(loan)
# svReport_loan.show_html(filepath="./reports/loan.html", open_browser=False)

# %% [markdown]
# ## Order
# [//]: # (-.- .tabset)
#
# Der Datensatz `order.csv` beinhaltet 6471 Observationen mit den folgenden Informationen über die Daueraufträge eines Kontos:  
#
# - `order_id`: die Nummer des Dauerauftrags,  
# - `account_id`: die Kontonummer von welchem der Auftrag stammt,  
# - `bank_to`: die empfangende Bank,  
# - `account_to`: das empfangende Konto,  
# - `amount`: der Betrag,  
# - `k_symbol`: die Art des Auftrags (Versicherungszahlung, Haushalt, Leasing, Darlehen)
#

# %%
order.info()

# %%
print("Anzahl fehlender Werte:", sum(order.isnull().sum()))
print("Anzahl duplizierter Einträge:", order.duplicated().sum())

# %% [markdown]
# ### Aufbereitung
# Auch für `order.csv` werden die Kategorien zunächst übersetzt und fehlende Werte mit der Kategorie `unknown` ersetzt. Es bestehen deutlich mehr Daueraufträge als Bankkontos, was darauf hindeutet, dass ein Bankkonto mehrere Daueraufträge eingerichtet haben kann. Zur weiteren Verarbeitung der Daten wird das Format so geändert, dass pro Konto ein `order`-Eintrag existiert.  

# %%
# Kategorien übersetzen und fehlende Werte mit "unknown" füllen
order["k_symbol"] = (
    order["k_symbol"]
    .map(
        {
            "POJISTNE": "insurance_payment",
            "SIPO": "household",
            "UVER": "loan_payment",
            "LEASING": "leasing",
        }
    )
    .fillna("unknown")
)

order["k_symbol"] = order["k_symbol"].astype("category")

# %%
# Merge with 'account_id_df' to ensure all accounts are represented
order = pd.merge(account[["account_id"]], order, on="account_id", how="left")

# After merging, fill missing values that may have been introduced
order["k_symbol"] = order["k_symbol"].fillna("unknown")
order["amount"] = order["amount"].fillna(0)
order["has_order"] = ~order.isna().any(axis=1)

orders_pivot = order.pivot_table(
    index="account_id",
    columns="k_symbol",
    values="amount",
    aggfunc="sum",
    observed=False,
)

# Add prefix to column names
orders_pivot.columns = orders_pivot.columns

orders_pivot = orders_pivot.reset_index()

# NaN to 0
orders_pivot = orders_pivot.fillna(0)
# Sample 5 random rows from the merged DataFrame
orders_pivot.sample(n=5)

# %% [markdown]
# ### Empfangende Bank
# Die Verteilung der empfangenden Banken ist ziemlich ausgeglichen, wobei in 742 Observationen diese Angabe fehlt.  

# %% [markdown]
# ### Empfangendes Konto
# Auch bei den empfangenden Konten scheint es keine auffällige Konzentration bei wenigen Konten zu geben und bei 742 Observationen fehlt die Angabe ebenfalls.  

# %% [markdown]
# ### Betrag
# Der Betrag bewegt sich im Bereich zwischen 0 - 14882 mit einem Mittelwert von 2943 und einem Median von 2249. Die Verteilung ist also stark rechtsschief

# %% [markdown]
# ### Art
# Die meisten Daueraufträge sind betreffend dem Haushalt eingerichtet worden (3502), die wenigsten für Leasing (341).  

# %% [markdown]
# ### Korrelation & weitere Informationen
# Die Korrelation sowie weitere Informationen zu den vorhandenen Daten können aus dem [SweetViz Report](./reports/order.html) entnommen werden.

# %%
# Assuming data_frames is a dictionary for storing DataFrames
data_frames["order.csv"] = orders_pivot

# %%
# # %%capture
# svReport_order = sv.analyze(order)
# svReport_order.show_html(filepath="./reports/order.html", open_browser=False)

# %% [markdown]
# ## Trans
# [//]: # (-.- .tabset)
#
# Der Datensatz `trans.csv` beinhaltet 1056320 Observationen mit den folgenden Informationen über die Transaktionen eines Kontos:  
#
# - `trans_id`: die ID der Transaktion,  
# - `account_id`: die Kontonummer des ausführenden Kontos,  
# - `date`: das Datum,  
# - `type`: der Typ (Einzahlung, Bezug),  
# - `operation`: die Art der Transaktion (Bezug Kreditkarte, Bareinzahlung, Bezug über eine andere Bank, Bezug Bar, Überweisung)  
# - `amount`: der Betrag der Transaktion,  
# - `balance`: der Kontostand nach ausführung der Transaktion,  
# - `k_symbol`: die Klassifikation der Transaktion (Versicherungszahlung, Kontoauszug, Zinsauszahlung, Zinszahlung bei negativem Kontostand, Haushalt, Pension, Darlehensauszahlung),  
# - `bank`: die empfangende Bank und   
# - `account`: das empfangende Bankkonto
#

# %%
trans.info()

# %%
print("Anzahl fehlender Werte:", sum(trans.isnull().sum()))
print("Anzahl duplizierter Einträge:", trans.duplicated().sum())

# %% [markdown]
# ### Aufbereitung
# Die Kategorien für `type`, `operation` und `k_symbol` wurden übersetzt und die Datentypen korrigiert.  

# %%
trans["date"] = pd.to_datetime(trans["date"], format="%y%m%d")

# Update 'type' column
trans["type"] = trans["type"].replace(
    {"PRIJEM": "credit", "VYDAJ": "withdrawal", "VYBER": "withdrawal"}
)
trans["type"] = trans["type"].astype("category")

# Update 'operation' column
trans["operation"] = trans["operation"].replace(
    {
        "VYBER KARTOU": "credit card withdrawal",
        "VKLAD": "credit in cash",
        "PREVOD Z UCTU": "collection from another bank",
        "VYBER": "cash withdrawal",
        "PREVOD NA UCET": "remittance to another bank",
    }
)
trans["operation"] = trans["operation"].astype("category")

# Update 'k_symbol' column
trans["k_symbol"] = trans["k_symbol"].replace(
    {
        "POJISTNE": "insurance payment",
        "SLUZBY": "statement payment",
        "UROK": "interest credited",
        "SANKC. UROK": "sanction interest if negative balance",
        "SIPO": "household payment",
        "DUCHOD": "pension credited",
        "UVER": "loan payment",
    }
)
trans["k_symbol"] = trans["k_symbol"].astype("category")

# negate the amount if type is credit
trans.loc[trans["type"] == "withdrawal", "amount"] = trans.loc[
    trans["type"] == "withdrawal", "amount"
] * (-1)

# %%
# Sample 5 random rows from the DataFrame
trans.sample(n=5)

# %% [markdown]
# ### Zeitliche Entwicklung eines Kontos

# %%
# Plot Zeitliche Entwicklung des Konto-Saldos für die Konto nummer 19
account_19 = trans[trans["account_id"] == 19].copy()  # Create a copy of the DataFrame
# Ensure the date column is in datetime format
account_19["date"] = pd.to_datetime(account_19["date"])

# Sort the values by date
account_19 = account_19.sort_values("date")

plt.figure(figsize=(10, 6))
plt.plot(account_19["date"], account_19["balance"])
plt.title("Time evolution of balance for account number 19")
plt.xlabel("Date")
plt.ylabel("Balance")
plt.show()

# %%
# zoom the year 1995 of the plot
account_19_1995 = account_19[account_19["date"].dt.year == 1995]
# plot it
plt.figure(figsize=(10, 6))
plt.plot(account_19_1995["date"], account_19_1995["balance"])
plt.title("Time evolution of balance for account number 19 in 1995")
plt.xlabel("Date")
plt.ylabel("Balance")
plt.show()

# Wee see that there is a steep line in 1995-10 so there are two transactions, this we have to clean.

# %% [markdown]
# ### Korrelation & weitere Informationen
# Die Korrelation sowie weitere Informationen zu den vorhandenen Daten können aus dem [SweetViz Report](./reports/trans.html) entnommen werden.

# %%
# Assign to a dictionary if needed (similar to list assignment in R)
data_frames["trans.csv"] = trans

# %%
# # %%capture
# svReport_trans = sv.analyze(trans)
# svReport_trans.show_html(filepath="./reports/trans.html", open_browser=False)

# %% [markdown]
# # Kombinieren der Daten zu einem Modellierungsdatensatz
# Im nachfolgenden Abschnitt werden die Daten zu statischen (Kunden-) Daten und transaktionellen (Bankdienstleistungs-) Daten kombiniert um diese anschliessend zu einem Datensatz für die Modellierung zusammenzufügen.  
#  
# ## Stammdaten
# Die aufbereiteten Stammdaten aus den Dateien 
#
# - `disp.csv`  
# - `account.csv`  
# - `client.csv`  
# - `card.csv`  
# - `loan.csv`  
# - `order.csv`  
# - `districts.csv`  
#
# werden nachfolgend zu einem Datensatz kombiniert. 

# %%
# merge dataframes
static_data = (
    data_frames["disp.csv"]
    .merge(data_frames["account.csv"], on="account_id", validate="1:1", how="left")
    .merge(
        data_frames["client.csv"],
        on="client_id",
        validate="1:1",
        suffixes=("_account", "_client"),
        how="left",
    )
    .merge(
        data_frames["card.csv"],
        on="disp_id",
        validate="1:1",
        suffixes=("_disp", "_card"),
        how="left",
    )
    .merge(
        data_frames["loan.csv"],
        on="account_id",
        suffixes=("_account", "_loan"),
        validate="1:1",
        how="left",
    )
    .merge(data_frames["order.csv"], on="account_id", validate="1:1", how="left")
    .merge(
        data_frames["district.csv"].add_suffix("_account"),
        left_on="district_id_account",
        right_on="district_id_account",
        validate="m:1",
        how="left",
    )
    .merge(
        data_frames["district.csv"].add_suffix("_client"),
        left_on="district_id_client",
        right_on="district_id_client",
        validate="m:1",
        how="left",
    )
)

# %%
static_data["has_card"] = ~static_data["card_id"].isna()

# %%
static_data["type_card"] = static_data["type_card"].cat.add_categories(["none"])
static_data.loc[static_data["card_id"].isna(), "type_card"] = "none"

# %%
# get static_data status categories
static_data["status"] = static_data["status"].cat.add_categories(["none"])
static_data.loc[static_data["status"].isna(), "status"] = "none"

# %%
static_data.info()

# %%
print("Anzahl duplizierter Einträge:", static_data.duplicated().sum())

# %%
# fillna for payments, duration, amount
static_data["payments"] = static_data["payments"].fillna(0)
static_data["duration"] = static_data["duration"].fillna(0)
static_data["amount"] = static_data["amount"].fillna(0)

# %% [markdown]
# Damit wird ein Datensatz mit 4500 individuellen Kunden und 56 Spalten erzeugt. 892 dieser Kunden besitzen eine Kreditkarte und 682 haben einen Kredit aufgenommen. 

# %% [markdown]
# ## Entfernen der Junior Karteninhaber
# Kunden im jugendlichen Alter sind speziell interessante Kunden für eine Bank, da diese grundsätzlich noch keine bis wenige Bankdienstleistungen beziehen und somit flexibel sind. Es ist deshalb sehr vorteilhaft für ein Unternehmen diese für sich zu gewinnen, weshalb viele Banken für solche Kunden ganz spezifische Prozesse definieren. Das in dieser Aufgabenstellung gewünschte Modell würde in so einem Prozess nicht eingesetzt werden, weshalb die jugendlichen Kunden nachfolgend aus dem Datensatz entfernt werden.    

# %%
num_accounts_before = len(static_data)
# # Filter rows where 'card_type' does not contain 'junior' (case insensitive)
static_data = static_data[
    ~static_data["type_card"].str.contains("junior", case=False, na=False)
]
num_accounts_after = len(static_data)
num_junior_cards = num_accounts_before - num_accounts_after
print(f"Number of junior cards removed: {num_junior_cards}")

# %% [markdown]
# Durch diese Entfernung wurden 145 Kunden entfernt. 

# %% [markdown]
# ## Bewegungsdaten
# Um einen Datensatz zu erhalten, bei welchem jede Zeile eine Observation repräsentiert müssen die Transaktionen pro Kunde entsprechend aufgerollt werden. Das bedeutet, ein vordefiniertes Zeitfenster vor dem zu modellierenden Event zu definieren und die darin enthaltenen Daten in einer Zeile zu aggregieren. Das gesuchte Zeitfenster beinhaltet bestenfalls saisonale Gegebenheiten und stets einen Lag-Zeitraum, der die Verzögerung der Kaufentscheidung und Ausführung des Auftrags aufzeichnen soll. 
# Hier wird ein Rollup-Fenster inklusive Lag von 13 Monaten eingesetzt. 
#
# ### Käufer
# Für Kunden, die bereits eine Kreditkarte besitzen, ist es unkompliziert, das Rollup-Fenster zu identifizieren. 
#

# %%
# select all transactions from trans from date 1995-03-16 and account_id 150
trans[(trans["date"] == "1995-03-16") & (trans["account_id"] == 150)]

# %% [markdown]
# Aus dieser Tabelle ersichtlich ist, dass für den Kunden 150 zum Datum der Eröffnung des Kontos mehrere Transaktionen vorhanden sind und dass wenn die Beträge von dem Tag aufsummiert werden, der korrekte Kontostand resultiert (1900 + 900 = 2800). Deshalb wird nachfolgend davon ausgegangen, dass die Aufsummierung der Transaktionsbeträge zum korrekten Kontostand führt. 

# %%
# sort dataframe trans by account_id and date
first_row_per_account = trans.groupby("account_id")

# select rows where amount == balance
first_row_per_account = first_row_per_account.apply(
    lambda x: x[x["amount"] == x["balance"]].iloc[0], include_groups=False
).reset_index()

# %%
# show that there's one row per unique account_id in trans
first_row_per_account["account_id"].nunique() == trans["account_id"].nunique()

# %%
first_row_per_account.query("amount != balance")

# %% [markdown]
# Mit dem obigen Code wird zudem sichergestellt, dass diese Gegebenheit für alle Kunden gilt. Nachfolgend werden die Transaktionen aggregiert, sodass die Spalten  
#
# - `volume`: das Volumen, also die Summe der Ein- und Ausgaben auf dem Konto,  
# - `credit`: die Summe der Einnahmen,  
# - `withdrawal`: die Summe der Ausgaben,  
# - `n_transactions`: die Anzahl der getätigten Transaktionen und  
# - `balance`: der Kontostand
#
# pro Monat entstehen. Dieser Datensatz wird dann mithilfe der nachfolgend definierten Funktion `rollup_credit_card` aufgerollt. 

# %%
# Extract year and month from date to a new column 'year_month'
trans["year_month"] = trans["date"].dt.to_period("M")

# Group by 'account_id' and 'month', and calculate the sum of 'amount', 'credit', 'withdrawal' and 'n_transactions'
transactions_monthly = (
    trans.groupby(["account_id", "year_month"])
    .agg(
        volume=("amount", "sum"),
        credit=("amount", lambda x: x[x > 0].sum()),
        withdrawal=("amount", lambda x: x[x < 0].sum()),
        n_transactions=("amount", "count"),
    )
    .reset_index()
)

# Calculate cumulative sum of 'volume' for each account
transactions_monthly["balance"] = transactions_monthly.groupby("account_id")[
    "volume"
].cumsum()

# %%
# count unique account_ids in transactions_monthly
print(transactions_monthly["account_id"].nunique())
num_accounts_before = transactions_monthly["account_id"].nunique()


# %%
def rollup_credit_card(trans_monthly, account_card_issue_dates):
    # Add issue date and calculate months since card issue
    trans_monthly = pd.merge(trans_monthly, account_card_issue_dates, on="account_id")

    trans_monthly["months_before_card_issue"] = [
        (issued - year_month).n
        for issued, year_month in zip(
            trans_monthly["issued"], trans_monthly["year_month"]
        )
    ]

    # select only where months_before_card_issue > 0 and <= 13
    trans_monthly = trans_monthly[
        (trans_monthly["months_before_card_issue"] > 0)
        & (trans_monthly["months_before_card_issue"] <= 13)
    ]

    trans_monthly = trans_monthly.groupby("account_id").filter(lambda x: len(x) == 13)

    # Pivot wider
    trans_monthly = trans_monthly.pivot_table(
        index="account_id",
        columns="months_before_card_issue",
        values=["volume", "credit", "withdrawal", "n_transactions", "balance"],
    )
    trans_monthly.reset_index(inplace=True)
    trans_monthly.columns = [
        "_".join(str(i) for i in col) for col in trans_monthly.columns
    ]
    # rename account_id_ to account_id
    trans_monthly = trans_monthly.rename(columns={"account_id_": "account_id"})

    return trans_monthly


# %%
buyers = static_data[static_data["has_card"]]

# print number of buyers
print(buyers["account_id"].nunique())

# %%
transactions_rolled_up_buyers = rollup_credit_card(
    transactions_monthly, buyers.loc[:, ["account_id", "issued"]]
)

transactions_rolled_up_buyers.info()

# %%
# calculate the number of buyers lost by rolling up
lost_buyers = buyers[~buyers["account_id"].isin(transactions_rolled_up_buyers["account_id"])]
lost_buyers = lost_buyers["account_id"].nunique()
print(lost_buyers)

# %% [markdown]
# Und so entsteht ein Datensatz, welcher für 568 Kreditkartenkäufer die Merkmale `balance`, `credit`, `n_transactions`, `volume` und `withdrawal` für alle 13 Monate des Rollup-Fensters inklusive Lag beinhaltet. 

# %% [markdown]
# ### Nicht-Käufer
# Um die Transaktionen der Nicht-Käufer analog zu verarbeiten, wird ein fiktives Kaufdatum benötigt, welches als Ausgangslage für die Aufrollung dient.

# %%
# plot the issue date distribution of the buyers that where rolled up (586)
merged_data = transactions_rolled_up_buyers.merge(
    buyers, how="left", left_on="account_id", right_on="account_id"
)
plt.figure(figsize=(10, 6))
merged_data["issued"].value_counts().sort_index().plot(kind="bar")
plt.title("Verteilung der Ausstellungsdaten")
plt.xlabel("Datum")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# Hier dargestellt ist die Verteilung der Ausstellungsdaten der Kreditkarten. Zu erkennen ist ein klarer aufwärtstrend.

# %%
# from transactions_monthly plot the number of distinct account ids per month
plt.figure(figsize=(10, 6))
transactions_monthly["year_month"].value_counts().sort_index().plot(kind="bar")
plt.title("Anzahl der Konten pro Monat")
plt.xlabel("Monat")
plt.ylabel("Anzahl")
plt.show()

# %%
# calculate the correlation between transactions_monthly["year_month"] and merged_data["issued"].value_counts()
transactions_monthly["year_month"].value_counts().sort_index().corr(merged_data["issued"].value_counts().sort_index())


# %% [markdown]
# Hier dargestellt ist die Verteilung der Anzahl von eröffneten Konten pro Monat. Der hier beobachtete Aufwärtstrend der Anzahl erstellter Konten könnte ein maßgebender Einflussfaktor der Anzahl ausgestellter Kreditkarten pro Monat sein. Der Korrelationskoeffizient von 0.78 unterstreicht diese Beobachtung. Wir gehen davon aus, dass diese Gegebenheit von einem Klassifikationsmodell schnell overfitted wird, weshalb wir nachfolgend ein random sampling einsetzen, um das fiktive Issue-Date von Nicht-Käufern zu definieren.

# %%
def rollup_non_credit(trans_monthly, non_buyers, range):
    # set seed
    np.random.seed(43)
    # for each non buyer, find the date of the first transaction
    first_transaction_dates = (
        trans_monthly.groupby("account_id")["year_month"].min().reset_index()
    )
    first_transaction_dates.columns = ["account_id", "first_transaction_date"]

    # merge the first transaction dates with the non_buyers DataFrame
    non_buyers = non_buyers.merge(first_transaction_dates, on="account_id", how="left")

    # randomly sample a date from the range as issue date for each non buyer making sure that the random date is after the first transaction of the non buyer
    non_buyers["issued"] = non_buyers["first_transaction_date"].apply(
        lambda x: np.random.choice(range, 1)[0]
    )

    non_buyers_rolled_up = rollup_credit_card(
        trans_monthly, non_buyers.loc[:, ["account_id", "issued"]]
    )
    return non_buyers_rolled_up, non_buyers


# %%
# get the list of issue dates of buyers
issue_dates_buyers = buyers["issued"].unique()
n_non_buyers = static_data[~static_data["has_card"]]["account_id"].nunique()
transactions_rolled_up_non_buyers, non_buyers = rollup_non_credit(transactions_monthly,
                                                                  static_data[~static_data["has_card"]],
                                                                  issue_dates_buyers)
transactions_rolled_up_non_buyers.info()

# %% [markdown]
# So finden wir 1684 aufgerollte Datensätze für Nicht-Käufer. 

# %%
non_buyers_lost = n_non_buyers - transactions_rolled_up_non_buyers["account_id"].nunique()
print(non_buyers_lost)

# %%
# plot the issue date distribution of the non-buyers that where rolled up
plt.figure(figsize=(10, 6))
non_buyers["issued"].value_counts().sort_index().plot(kind="bar")
plt.title("Verteilung der Ausstellungsdaten")
plt.xlabel("Datum")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# ## Zusammenfügen der Daten
# Nachfolgend werden die Stammdaten mit den aufgerollten Bewegungsdaten zum Modellierungsdatensatz kombiniert.

# %%
transactions_rolled_up = pd.concat(
    [transactions_rolled_up_buyers, transactions_rolled_up_non_buyers]
)
# merge transactions_rolled_up and static data
X = pd.merge(static_data, transactions_rolled_up, on="account_id")

# %% [markdown]
# ### Entfernen von minderjährigen Kunden
# Kunden die zum Zeitpunkt des Erwerbs der Kreditkarte minderjährig waren, müssen entfernt werden, da diese Zielgruppe, wie bereits beschrieben, nicht modelliert werden soll.

# %%
num_before_underage_removal = X["account_id"].nunique()
X['issued'] = X['issued'].fillna(non_buyers["issued"])
X['issued'] = X['issued'].dt.to_timestamp()

time_to_compare = pd.Timedelta(days=6970)

# Filter underage accounts
X = X[(X['issued'] - X['birth_day']) >= time_to_compare]

# Calculate the number of accounts after filtering
num_accounts_after = X['account_id'].nunique()

# Calculate the number of underage accounts
num_underage_accounts = num_before_underage_removal - num_accounts_after
num_underage_accounts

# %% [markdown]
# 120 Kunden werden durch diesen Schritt entfernt.

# %%
y = X["has_card"]

X = X.drop(columns=["has_card", "card_id", "issued", "type_card", "loan_id", "date_loan", "disp_id", "client_id",
                    "district_id_account", "birth_day", "type_disp"])
# convert "date_account" to "days active"
X["date_account"] = (X["date_account"] - X["date_account"].min()).dt.days

# %%
# show NaNs in X
X.isnull().sum()

# %% [markdown]
# Zu sehen ist, dass der Datensatz komplett ist, also keine fehlenden Werte aufweist.

# %% [markdown]
# ## Explorative Datenanalyse des Modellierungssatzes
# Nachfolgend werden diverse Askepte des zusammengefügten Datensatzes untersucht
# ### Entfernte Konten
# In der Vorverarbeitung werden diverse Kundenkonten aus dem Datensatz entfernt. Die folgende Darstellung zeigt auf, wie viele in welchem Schritt entfernt werden.

# %%
num_junior_cards, lost_buyers, non_buyers_lost, num_underage_accounts, X.shape[0]

# %%
waterfall_data = {
    "step": ["Initial", "Junior Card Holders", "Lost Buyers", "Non-Buyers", "Underage Clients", "Final"],
    "count": [4500, -num_junior_cards, -lost_buyers, -non_buyers_lost, -num_underage_accounts, X.shape[0]],
}

waterfall_df = pd.DataFrame(waterfall_data)

blank = waterfall_df["count"].cumsum().shift(1).fillna(0)
step = blank.reset_index(drop=True).repeat(3).shift(-1)
step[1::3] = np.nan
blank[5] = 0

# %%
my_plot = waterfall_df.plot(kind='bar', stacked=True, bottom=blank, legend=False,
                            title="Entfernte Anzahl von Konten in verschiedenen Aufbereitungsschritten")
my_plot.plot(step.index, step.values, 'k')

# %%
display(waterfall_df)

# %% [markdown]
# Insgesamt werden also 2368 Konten in der Vorberarbeitung entfernt. 
# ### Verteilung Kartenbesitzer
# Nachfolgend wird die Verteilung der Kartenbesitzer aufgezeigt.

# %%
# plot distribution of has_card
plt.figure(figsize=(10, 6))
y.value_counts().plot(kind="bar")
plt.title("Verteilung der Kartenbesitzer")
plt.xlabel("Kartenbesitzer")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# Klar ersichtlich ist, dass es deutlich mehr Nicht-Kartenbesitzer als Kartenbesitzer gibt. Unbalancierten Daten erschweren die Modellierung erheblich, weshalb nachfolgend SMOTE (Synthetic Minority Over-sampling Technique) eingesetzt wird, um die Daten zu balancieren.

# %%
from imblearn.over_sampling import SMOTE

# x get dummy variables for category
X = pd.get_dummies(X, drop_first=True)

sm = SMOTE(random_state=43)
X_res, y_res = sm.fit_resample(X, y)

# %%
# plot distribution of has_card
plt.figure(figsize=(10, 6))
y_res.value_counts().plot(kind="bar")
plt.title("Verteilung der Kartenbesitzer")
plt.xlabel("Kartenbesitzer")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# Durch den Einsatz von SMOTE konnte der Datensatz ausbalanciert werden.
# ### Konten 14 und 18
# In der Aufgabenbeschreibung wurde explizit verlangt, die Konten 14 und 18 zu untersuchen. Nachfolgend dargestellt ist der Saldo und das Volumen der beiden Konten. 

# %%
# Filter the data for accounts 14 and 18
account_data = X[X["account_id"].isin([14, 18])]
# Reshape the DataFrame for easier plotting
months = [f"Month {i}" for i in range(1, 14)]
balances = [f"balance_{i}" for i in range(1, 14)]
volumes = [f"volume_{i}" for i in range(1, 14)]

# Melt the DataFrame for balances and volumes
balance_data = account_data.melt(
    id_vars="account_id", value_vars=balances, var_name="Month", value_name="Balance"
)
volume_data = account_data.melt(
    id_vars="account_id", value_vars=volumes, var_name="Month", value_name="Volume"
)

# Convert 'Month' from string to integer for proper sorting
balance_data["Month"] = balance_data["Month"].str.extract(r"(\d+)").astype(int)
volume_data["Month"] = volume_data["Month"].str.extract(r"(\d+)").astype(int)

# Sort data by account and month
balance_data = balance_data.sort_values(by=["account_id", "Month"])
volume_data = volume_data.sort_values(by=["account_id", "Month"])
# Plotting balance data
plt.figure(figsize=(14, 7))
for key, grp in balance_data.groupby("account_id"):
    plt.plot(grp["Month"], grp["Balance"], label=f"Account {key} Balances")
plt.title("Monthly Balances for Accounts 14 and 18")
plt.xlabel("Month")
plt.ylabel("Balance")
plt.legend()
plt.show()

# Plotting volume data
plt.figure(figsize=(14, 7))
for key, grp in volume_data.groupby("account_id"):
    plt.plot(grp["Month"], grp["Volume"], label=f"Account {key} Volumes")
plt.title("Monthly Transaction Volumes for Accounts 14 and 18")
plt.xlabel("Month")
plt.ylabel("Volume")
plt.legend()
plt.show()


# %% [markdown]
# ## Feature Engineering

# %%
# Function to calculate features
def calculate_features(df, prefix):
    monthly_values = df[[f"{prefix}_{i}" for i in range(1, 13)]]
    # needs to be a small constant to avoid division by zero
    epsilon = 1e-7  # small constant
    features = {
        f"{prefix}_mean": monthly_values.mean(axis=1),
        f"{prefix}_min": monthly_values.min(axis=1),
        f"{prefix}_max": monthly_values.max(axis=1),
        f"{prefix}_mad": monthly_values.sub(monthly_values.mean(axis=1), axis=0)
        .abs()
        .mean(axis=1),
        f"{prefix}_mean_ratio_last3_first3": (
            monthly_values[[f"{prefix}_{i}" for i in range(10, 13)]].mean(axis=1)
            / (
                monthly_values[[f"{prefix}_{i}" for i in range(1, 4)]].mean(axis=1)
                + epsilon
            )
        ),
    }

    if prefix in ["credit", "withdrawal"]:
        features[f"{prefix}_sum"] = monthly_values.sum(axis=1)
    if prefix in ["balance", "credit"]:
        features[f"{prefix}_std"] = monthly_values.std(axis=1)

    return features


# List of column prefixes for required calculations
columns_to_process = ["balance", "credit", "n_transactions", "withdrawal"]

# Generating features for each prefix and merging them
all_features = {}
for prefix in columns_to_process:
    all_features.update(calculate_features(X, prefix))

# Creating the final dataframe with new features
df_features = pd.DataFrame(all_features)


X_feature_engineered = pd.concat([X_res, df_features], axis=1)
display(X_feature_engineered.head(5))


# %%
# Remove the Variable that can lead to data leakage
def clean_data(df):
    # Define unnecessary columns
    unnecessary_cols = [
        "disp_id",
        "client_id",
        "account_id",
        "type_card",
        "card_id",
        "loan_id",
        "district_id_account",
        "district_id_client",
    ]
    # Drop these columns if they exist in the dataframe
    df_cleaned = df.drop(columns=[col for col in unnecessary_cols if col in df.columns])
    return df_cleaned


X_res = clean_data(X_res)
X_feature_engineered = clean_data(X_feature_engineered)

# %%
display(X_feature_engineered.head(5))
display(X_res.head(5))


# %%
# only print the columns with missing values
X_res.isnull().sum()[X_res.isnull().sum() > 0]
X_feature_engineered.isnull().sum()[X_feature_engineered.isnull().sum() > 0]


# %%
# impute missing values with knn imputation but in dataframe
# import KNNImputer
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_res = pd.DataFrame(imputer.fit_transform(X_res), columns=X_res.columns)
X_feature_engineered = pd.DataFrame(
    imputer.fit_transform(X_feature_engineered), columns=X_feature_engineered.columns
)


# %%
# only print the columns with missing values
X_res.isnull().sum()[X_res.isnull().sum() > 0]
X_feature_engineered.isnull().sum()[X_feature_engineered.isnull().sum() > 0]


# %%
# Nomalize the data and standardize the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_res = pd.DataFrame(scaler.fit_transform(X_res), columns=X_res.columns)
X_feature_engineered = pd.DataFrame(
    scaler.fit_transform(X_feature_engineered), columns=X_feature_engineered.columns
)


# %% [markdown]
# # Evaluations Daten

# %%
# Assuming 'X' is your DataFrame and 'has_card' is the target variable
# y_features = X_feature_engineered["has_card"]

# X_feature_engineered.drop("has_card", axis=1, inplace=True)
# we use kfold for cross validation and then the X_test and y_test are used for evaluation on never seen data
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.1, random_state=42, stratify=y_res
)

X_train_features, X_test_features, y_train_features, y_test_features = train_test_split(
    X_feature_engineered, y_res, test_size=0.1, random_state=42, stratify=y_res
)

# %% [markdown]
# ## Drop Featrues

# %%
X_train.dtypes

# %% [markdown]
# # Modeling und Model Selection
# Für die Model Selection benutzen wir einen StratifiedKFold mit 10 Folds in dem wir nur den Train split Folden, denn später brauchen wir die Test Daten für den Error Assesment.

# %%


class ModelEvaluator:
    def __init__(self, models, param_grid, X, y, X_test, y_test, selected_fields=None):
        """
        Initialize the evaluator with models, their parameter grids, and data.

        :param models: dict of (name, model) pairs
        :param param_grid: dict of (name, param_grid) pairs for GridSearch
        :param X: Feature matrix
        :param y: Target vector
        :param selected_fields: Fields selected for training
        """
        self.benchmark_results = {}
        self.models = models
        self.param_grid = param_grid
        self.X = X[selected_fields]
        self.y = y
        self.eval_data = X_test
        self.eval_target = y_test
        self.fitted_models = {}
        self.best_models = {}
        self.cv_predictions = {}

    def get_benchmark_results(self):
        return self.benchmark_results

    def fit_models(self):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for name, model in self.models.items():
            pipeline = self.create_pipeline(model)

            # Prefix the parameters with the step name 'model'
            grid_search_params = {}
            for param, values in self.param_grid[name].items():
                grid_search_params = {
                    f"model__{param}": values
                    for param, values in self.param_grid[name].items()
                }

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=grid_search_params,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(self.X, self.y)
            print(f"Best parameters for {name}: {grid_search.best_params_}")

            best_pipeline = grid_search.best_estimator_

            metrics = {
                "roc_auc": "roc_auc",
                "precision": "precision",
                "recall": "recall",
                "accuracy": "accuracy",
                "f1": make_scorer(fbeta_score, beta=1),
                "kappa": make_scorer(cohen_kappa_score),
                "mcc": make_scorer(matthews_corrcoef),
            }

            self.benchmark_results[name] = {}
            all_cv_preds = np.zeros(len(self.y))

            results = cross_validate(
                best_pipeline,
                self.X,
                self.y,
                cv=cv,
                scoring=metrics,
                return_estimator=True,
                n_jobs=-1,
                verbose=0,
            )

            for metric_name in metrics.keys():
                self.benchmark_results[name][metric_name] = np.mean(
                    results["test_" + metric_name]
                )
                print(
                    f"{name}: {metric_name} = {np.mean(results['test_' + metric_name]):.2f}"
                )

            for train_idx, test_idx in cv.split(self.X, self.y):
                best_pipeline.fit(self.X.iloc[train_idx], self.y.iloc[train_idx])
                all_cv_preds[test_idx] = best_pipeline.predict_proba(
                    self.X.iloc[test_idx]
                )[:, 1]

            self.cv_predictions[name] = all_cv_preds
            self.fitted_models[name] = best_pipeline.fit(self.X, self.y)

    def evaluate_models(self):
        if not self.fitted_models:
            self.fit_models()
        return self.benchmark_results

    def plot_roc_curves(self):
        if not self.fitted_models:
            self.fit_models()

        plt.figure(figsize=(10, 8))
        for name in self.fitted_models:
            y_scores = self.cv_predictions[name]
            fpr, tpr, _ = roc_curve(self.y, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (area = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.show()

    def create_pipeline(self, model):
        categorical_cols = self.X.select_dtypes(include=["category", "object"]).columns
        numeric_cols = self.X.select_dtypes(include=["int64", "float64"]).columns

        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    def compare_top_n_customers(self, model_name, n=100):
        print(f"Comparing top {n} customers for {model_name}")
        model = self.fitted_models[model_name]
        probabilities = model.predict_proba(self.X)[:, 1]
        top_n_indices = np.argsort(probabilities)[::-1][:n]

        plt.figure()
        plt.hist(probabilities[top_n_indices], bins=20, alpha=0.75)
        plt.title(f"Histogram of top {n} customers' probabilities for {model_name}")
        plt.xlabel("Probability")
        plt.ylabel("Frequency")
        plt.show()

    def plot_confusion_matrices(self):
        if not self.fitted_models:
            self.fit_models()

        for name, model in self.fitted_models.items():
            plt.style.use("default")
            y_pred = model.predict(self.X)
            cm = confusion_matrix(self.y, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix for {name}")
            plt.show()
            plt.style.use("ggplot")


class MetricsBenchmarker:
    def __init__(self):
        """
        Initialize the benchmarker with models and data.

        :param models: dict of (name, model) pairs
        :param X: Feature matrix
        :param y: Target vector
        :param selected_fields: Fields selected for training
        """
        self.benchmark_results = {}
        self.evals = []

    def add_evaluator(self, evaluator: ModelEvaluator):
        self.evals.append(evaluator)

    def set_benchmark_results(self):
        for eval in self.evals:
            self.benchmark_results.update(eval.get_benchmark_results())

    def display_benchmark_results_table(self):
        """
        Display a table of benchmark results.
        """
        results_df = pd.DataFrame(self.benchmark_results).T
        display(results_df)

    def plot_benchmark_results_bar_chart(self):
        """
        Plot a bar chart of benchmark results.
        """
        results_df = pd.DataFrame(self.benchmark_results).T
        results_df.plot(kind="bar", figsize=(10, 6))
        plt.title("Benchmark Results")
        plt.ylabel("Score")
        plt.show()



# %%
# Example usage
from sklearn.linear_model import LogisticRegression

# Define models and their parameter grids
models = {
    "Baseline Logistic Regression": LogisticRegression(solver="liblinear"),
}
param_grid = {
    "Baseline Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
}

selected_fields = (
    ["age"]
    + [col for col in X_train.columns if "gender" in col]
    + [col for col in X_train.columns if "region_client" in col]
    + [f"volume_{i}" for i in range(1, 14)]
    + [f"balance_{i}" for i in range(1, 14)]
)


evaluator_baseline = ModelEvaluator(
    models,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test,
    selected_fields=selected_fields,
)
evaluator_baseline.evaluate_models()
evaluator_baseline.plot_roc_curves()
evaluator_baseline.plot_confusion_matrices()
evaluator_baseline.compare_top_n_customers("Baseline Logistic Regression", n=100)

# %%
# Define models and their parameter grids
models = {
    "Logistic Regression Features": LogisticRegression(solver="liblinear"),
}
param_grid = {
    "Logistic Regression Features": {"C": [0.01, 0.1, 1, 10]},
}

selected_fields = X_train_features.columns

evaluator = ModelEvaluator(
    models,
    param_grid,
    X_test_features,
    y_test_features,
    X_train_features,
    y_train_features,
    selected_fields=selected_fields,
)
evaluator.evaluate_models()
evaluator.plot_roc_curves()
evaluator.compare_top_n_customers("Logistic Regression Features", n=100)


# Assuming X and y are defined
#

# %% [markdown]
# ## Overfitting because of jagged ROC curve needs Regularization

# %%
# Define models and their parameter grids
models = {
    "Logistic Regression Features added": LogisticRegression(solver="liblinear"),
}
param_grid = {
    "Logistic Regression Features added": {"C": [0.001, 0.01, 0.1, 1, 10]},
}

selected_fields = X_train_features.columns

# Fix the not converging models with LassoCV
# for model_name, model in models.items():
#    models[model_name] = Pipeline(
#        [
#            ("scaler", StandardScaler()),
#            (
#                "feature_selection",
#                SelectFromModel(LassoCV(alphas=[0.01, 0.1, 1, 10], max_iter=10000)),
#            ),
#            ("model", model),
#        ]
#    )


evaluator = ModelEvaluator(
    models,
    param_grid,
    X_test_features,
    y_test_features,
    X_train_features,
    y_train_features,
    selected_fields=selected_fields,
)


evaluator.evaluate_models()
evaluator.plot_roc_curves()
evaluator.compare_top_n_customers("Logistic Regression Features added", n=100)


# Assuming X and y are defined

# %%
# Usage example:
# Define models and their parameter grids
models = {
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),
}

param_grid = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.1, 0.01, 0.001],
    },
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
    "Decision Tree": {
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
    },
    "AdaBoost": {},
}


# Fix the not converging models with LassoCV
# for model_name, model in models.items():
#    models[model_name] = Pipeline(
#        [("feature_selection", SelectFromModel(LassoCV(max_iter=1500))), ("model", model)]
#    )

selected_fields = X_train_features.columns  # add the new features of df_features

evaluator_models = ModelEvaluator(
    models,
    param_grid,
    X_train_features,
    y_train_features,
    X_test_features,
    y_test_features,
    selected_fields=selected_fields,
)
results = evaluator_models.evaluate_models()
evaluator_models.plot_roc_curves()
# evaluator_models.plot_confusion_matrices()
# compare top n customers for all models

# %%
# compare top n customers for all models
for model_name in models.keys():
    evaluator_models.compare_top_n_customers(model_name, n=100)

# %% [markdown]
# ## Results Comparision

# %%
benchmark = MetricsBenchmarker()
benchmark.add_evaluator(evaluator_baseline)
benchmark.add_evaluator(evaluator_models)
benchmark.add_evaluator(evaluator)
benchmark.set_benchmark_results()
benchmark.display_benchmark_results_table()
benchmark.plot_benchmark_results_bar_chart()

# %%
# best model
# Define weights for the metrics
weights = {"roc_auc": 0.25, "precision": 0.25, "recall": 0.25, "f1": 0.25}

# Calculate weighted scores for each model
weighted_scores = {}
for model, metrics in benchmark.benchmark_results.items():
    weighted_score = sum(
        weights[metric] * score
        for metric, score in metrics.items()
        if metric in weights
    )
    weighted_scores[model] = weighted_score

# Find the best model based on weighted score
best_model_name = max(weighted_scores, key=weighted_scores.get)
best_model_score = weighted_scores[best_model_name]

# %% [markdown]
# # Model Assesment

# %%
# Evaluate the best model with the test set
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)


best_model = evaluator_models.fitted_models[best_model_name]
y_pred = best_model.predict(X_test_features)

print(f"Evaluation of the best model ({best_model_name}) using X_test:")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix for {best_model_name} on X_test")
plt.show()

# ROC Curve
y_scores = best_model.predict_proba(X_test_features)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f"{best_model_name} (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve for {best_model_name} on X_test")
plt.legend(loc="lower right")
plt.show()

# Other Metrics
accuracy = accuracy_score(y_test_features, y_pred)
precision = precision_score(y_test_features, y_pred)
recall = recall_score(y_test_features, y_pred)
f1 = f1_score(y_test_features, y_pred)
kappa = cohen_kappa_score(y_test_features, y_pred)
mcc = matthews_corrcoef(y_test_features, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Cohen Kappa: {kappa:.2f}")
print(f"Matthews Correlation Coefficient: {mcc:.2f}")

# %% [markdown]
# # Erklärbare Modelle

# %% [markdown]
# ## Reduziere Modell für Erklärbarkeit

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
import statsmodels.api as sm

# Ensure all data in X_train_features are numeric
X_train_features = X_train_features.apply(pd.to_numeric, errors="coerce")

# Drop rows with any NaN values in X_train_features and align y_res
X_train_features = X_train_features.dropna()
y_res_aligned = y_res.loc[X_train_features.index]

X_train_reduced = X_train_features.copy()
X_test_reduced = X_test_features.copy()
y_test_reduced = y_test_features.copy()
y_train_reduced = y_train_features.copy()

# Convert to DataFrame with feature names for consistency
X_train_reduced = pd.DataFrame(X_train_reduced, columns=X_train_features.columns)
X_test_reduced = pd.DataFrame(X_test_reduced, columns=X_train_features.columns)

# Add a constant column for the intercept
X_train_reduced = sm.add_constant(X_train_reduced)
X_test_reduced = sm.add_constant(X_test_reduced)

# Convert to numeric to avoid dtype issues
X_train_reduced = X_train_reduced.apply(pd.to_numeric)
X_test_reduced = X_test_reduced.apply(pd.to_numeric)

# Apply Lasso (L1) regularization for feature selection
lasso_model = LogisticRegressionCV(
    cv=5, penalty="l1", solver="liblinear", random_state=42, Cs=np.logspace(-4, 0, 50)
)
lasso_model.fit(X_train_reduced, y_train_reduced)

# Get the features with non-zero coefficients
coef = pd.Series(lasso_model.coef_[0], index=X_train_reduced.columns)
selected_features = coef[coef != 0].index.tolist()
print("Selected features after Lasso:", selected_features)

lasso_selected_fields = selected_features


# %% [markdown]
# ### Logistic Regression Reduziert

# %% [markdown]
# #### Wir machen kein Model Selection mehr sondern nur ein Model Assesment für Explainable AI

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import lime.lime_tabular
import shap

# Step 1: Create and Train a Reduced Model
selected_fields_reduced = lasso_selected_fields

# Ensure all data in selected fields are numeric
X_feature_engineered[selected_fields_reduced] = X_feature_engineered[
    selected_fields_reduced
].apply(pd.to_numeric)

X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    X_feature_engineered[selected_fields_reduced],
    y_res,
    test_size=0.1,
    random_state=42,
    stratify=y_res,
)

# Convert to DataFrame with feature names for consistency
X_train_reduced = pd.DataFrame(X_train_reduced, columns=selected_fields_reduced)
X_test_reduced = pd.DataFrame(X_test_reduced, columns=selected_fields_reduced)

reduced_model = LogisticRegression(solver="liblinear")
reduced_model.fit(X_train_reduced, y_train_reduced)

y_pred_reduced = reduced_model.predict(X_test_reduced)
cm = confusion_matrix(y_test_reduced, y_pred_reduced)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Reduced Model")
plt.show()

y_scores_reduced = reduced_model.predict_proba(X_test_reduced)[:, 1]
fpr, tpr, _ = roc_curve(y_test_reduced, y_scores_reduced)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f"Reduced Model (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Reduced Model")
plt.legend(loc="lower right")
plt.show()


# Step 2: Create a Wrapper Function for LIME
def predict_proba_with_feature_names(X):
    X_df = pd.DataFrame(X, columns=selected_fields_reduced)
    return reduced_model.predict_proba(X_df)


# Explain the Model with LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_reduced.values,
    feature_names=selected_fields_reduced,
    class_names=["No Card", "Card"],
    mode="classification",
)

for i in range(15):
    exp = explainer.explain_instance(
        data_row=X_test_reduced.iloc[i].values,
        predict_fn=predict_proba_with_feature_names,
    )
    exp.show_in_notebook(show_table=True)

# Step 3: Explain the Model with SHAP
# Summarize the background data using shap.sample
background_data = shap.sample(X_train_reduced, 100)

explainer_shap = shap.KernelExplainer(reduced_model.predict_proba, background_data)
shap_values = explainer_shap.shap_values(X_test_reduced)

shap.initjs()

# Print shapes to debug
print("SHAP values shape:", np.array(shap_values).shape)
print("X_test_reduced shape:", X_test_reduced.shape)

# Verify dimensions (consider removing the extra dimension if it exists)
instance_index = 0  # Change the instance index if needed
positive_class_index = 1

if len(shap_values.shape) > 2:  # Check for extra dimension
    shap_values = shap_values[
        :, :, positive_class_index
    ]  # Select positive class values

assert len(shap_values[instance_index]) == X_test_reduced.shape[1], "Dimension mismatch"

# Use only the SHAP values for the positive class (index 1)
shap.force_plot(
    explainer_shap.expected_value[positive_class_index],
    shap_values[instance_index],
    X_test_reduced.iloc[instance_index],
)
shap.summary_plot(shap_values, X_test_reduced, feature_names=selected_fields_reduced)


# %% [markdown]
# ## Interpretation von den Resultaten

# %% [markdown]
# - todo

# %% [markdown]
# # Convert Notebook

# %%
# %%capture
import subprocess
import pathlib
import os

try:
    file_path = pathlib.Path(os.path.basename(__file__))
except:
    file_path = pathlib.Path("AML_MC.ipynb")

# Check the file extension
if file_path.suffix == ".py":
    # If it's a Python script, convert it to a notebook
    try:
        subprocess.check_output(["jupytext", "--to", "notebook", str(file_path)])
        print("Converted to notebook.")
    except subprocess.CalledProcessError as e:
        print("Conversion failed. Error message:", e.output)
elif file_path.suffix == ".ipynb":
    # If it's a notebook, convert it to a Python script with cell markers
    try:
        subprocess.check_output(["jupytext", "--to", "py:percent", str(file_path)])
        print("Converted to Python script.")
    except subprocess.CalledProcessError as e:
        print("Conversion failed. Error message:", e.output)
else:
    print("Unsupported file type.")

# %%
import subprocess
import pathlib
import os

try:
    file_path = pathlib.Path(os.path.basename(__file__))
except:
    file_path = pathlib.Path("AML_MC.ipynb")

# Check the file extension
if file_path.suffix == ".qmd":
    # If it's a Python script, convert it to a notebook
    try:
        os.system("quarto convert AML_MC.qmd")
        print("Converted to notebook.")
    except subprocess.CalledProcessError as e:
        print("Conversion failed. Error message:", e.output)
elif file_path.suffix == ".ipynb":
    # If it's a notebook, convert it to a Python script with cell markers
    try:
        # quatro convert ipynb to qmd
        os.system("quarto convert AML_MC.ipynb")
        print("Converted to qmd.")
    except subprocess.CalledProcessError as e:
        print("Conversion failed. Error message:", e.output)
else:
    print("Unsupported file type.")

# %%
import os


os.system("quarto render AML_MC.ipynb --to html --embed-resources")

# %%
