# %% [raw]
# title:  AML Challenge 2024
# date: "Generated: {{ datetime.now().strftime('%Y-%m-%d') }}"
# author: Etienne Roulet, Alexander Shanmugam
# output:
#     general:
#         input_jinja: true
#     html:
#         code_folding: hide
#         code_tools: true
#         theme: readable

# %% [markdown]
# # Setup
# Die folgenden Code-Blöcke können genutzt werden, um die benötigten Abhängigkeiten zu installieren und zu importieren.

# %%
# %%capture
# %pip install -r ../requirements.txt

# %%
# %%capture
# %load_ext pretty_jupyter

# %%
# %%capture
# Laden der eingesetzten Libraries
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import sweetviz as sv
from IPython.display import display
from itables import init_notebook_mode
from sklearn.linear_model import LinearRegression

init_notebook_mode()


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
    print('explained_variance: ', round(metrics.explained_variance_score(y_true, y_pred), 4))
    print('mean_squared_log_error: ', round(metrics.mean_squared_log_error(y_true, y_pred), 4))
    print('r2: ', round(metrics.r2_score(y_true, y_pred), 4))
    print('MAE: ', round(metrics.mean_absolute_error(y_true, y_pred), 4))
    print('MSE: ', round(metrics.mean_squared_error(y_true, y_pred), 4))
    print('RMSE: ', round(np.sqrt(metrics.mean_squared_error(y_true, y_pred)), 4))


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
# # Transformationen & Explorative Datenanalyse
# Im folgenden Abschnitt werden die geladenen Daten separat so transformiert, dass jede Zeile einer Observation und jede Spalte einer Variable im entsprechenden Datenformat entspricht, also ins Tidy-Format gebracht.

# %%
data_frames = {}

# %% [markdown]
# ## Account
# [//]: # (-.- .tabset)
# Der Datensatz `accounts.csv` beinhaltet 4500 Observationen mit den folgenden Informationen über die Kontos der Bank:  
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
# ### Transformation
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

# append account data to dataframe collection
data_frames["account.csv"] = account

# sample 5 random rows
account.sample(n=5)

# %%
# %%capture
# generate sweetviz report
svReport_account = sv.analyze(account)
svReport_account.show_html(filepath="./reports/accounts.html", open_browser=False)

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
# Die Korrelation sowie weitere Informationen zu den vorhandenen Daten können aus dem [SweetViz Report](./reports/account.html) entnommen werden. 

# %% [markdown]
# ## Card
# [//]: # (-.- .tabset)
#
# Der Datensatz `card.csv` beinhaltet 892 Observationen mit den folgenden Informationen über die von der Bank herausgegebenen Kreditkarten:  
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
# ### Transformation
# Auch bei diesem Datensatz (`card.csv`) werden zunächst die Datentypen korrigiert um anschliessend die Inhalte entsprechend beschreiben zu können

# %%
# parse date
card["issued"] = pd.to_datetime(card["issued"].str[:6], format="%y%m%d")
# convert type to categorical
card["type"] = card["type"].astype("category")
# append to dataframes collection
data_frames["card.csv"] = card

card.sample(n=5)

# %%
# %%capture
# generate sweetviz report
svReport_card = sv.analyze(card)
svReport_card.show_html(filepath="./reports/card.html", open_browser=False)

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
card["issued"].dt.to_period("M").value_counts().sort_index().plot(kind="bar")
plt.title("Verteilung der Ausstellungsdaten")
plt.xlabel("Datum")
plt.ylabel("Anzahl")
plt.show()

# %% [markdown]
# Die Korrelation sowie weitere Informationen zu den vorhandenen Daten können aus dem [SweetViz Report](./reports/account.html) entnommen werden.

# %% [markdown]
# ## Client
# [//]: # (-.- .tabset)
#
# Der Datensatz `client.csv` beinhaltet 5369 Observationen mit den folgenden Informationen über die Kunden der Bank:  
# - `client_id`: die Kundennummer, 
# - `birth_number`: eine Kombination aus Geburtsdatum und Geschlecht sowie
# - `district_id`: die Adresse  

# %%
client.info()

# %%
print("Anzahl fehlender Werte:", sum(client.isnull().sum()))
print("Anzahl duplizierter Einträge:", client.duplicated().sum())

# %% [markdown]
# ### Transformation
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

data_frames["client.csv"] = client

# Sample 5 random rows
client.sample(n=5)

# %%
# %%capture
svReport_client = sv.analyze(client)
svReport_client.show_html(filepath="./reports/client.html", open_browser=False)

# %% [markdown]
# ### Geschlecht
# Hier dargestellt ist die Verteilung des Geschlechts der Bankkunden. Das Geschlecht der erfassten Bankkunden ist fast gleichverteilt mit einem etwas kleineren Frauenanteil.  

# %%
# plot distribution of gender
plt.figure(figsize=(10, 6))
gender_distribution = client['gender'].value_counts().plot(kind='bar')
plt.title('Verteilung des Geschlechts der Bankkunden')
plt.xlabel('Geschlecht')
plt.ylabel('Anzahl')
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

# %% [markdown]
# ## Disp
# [//]: # (-.- .tabset)
#
# Der Datensatz `disp.csv` beinhaltet 5369 Observationen mit den folgenden Informationen über die Dispositionen der Bank:  
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
# ### Transformation
# Auch die Variablen des Datensatzes `disp.csv` werden in die korrekten Datentypen übertragen. 

# %%
# Spalte type als Kategorie speichern 
disp["type"] = disp["type"].astype("category")

data_frames["disp.csv"] = disp

# random sample
disp.sample(n=5)

# %%
# %%capture
svReport_disp = sv.analyze(disp)
svReport_disp.show_html(filepath="./reports/disp.html", open_browser=False)

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

# %% [markdown]
# ## District
# [//]: # (-.- .tabset)
#
# Der Datensatz `district.csv` beinhaltet 77 Observationen mit den folgenden demografischen Informationen:  
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
# ### Transformation
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
# Wir gehen davon aus, dass es sich hier um effektiv fehlende Werte handelt und nicht um zensierte Daten, also Werte, für welche der exakte Wert fehlt, aber trotzdem Informationen vorhanden sind. In diesem Fall, wenn die Variable mit den fehlenden Werten eine hohe Korrelation mit anderen Prediktoren aufweist, bietet es sich an, KNN oder eine einfache lineare Regression für die Imputation anzuwenden. [1] 
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
regression_results(district_no_na["unemploy_rate95"],
                   lin_reg_unemploy.predict(district_no_na["unemploy_rate96"].values.reshape(-1, 1)))

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
regression_results(district_no_na["no_of_crimes95"],
                   lin_reg_crime.predict(district_no_na["no_of_crimes96"].values.reshape(-1, 1)))

# %% [markdown]
# Auch hier mit einem $R^2$ Wert von $0.9969$ gehen wir davon aus, damit ein stabiles Modell für die Imputation erreicht zu haben. Somit werden nachfolgend die beiden Modelle genutzt, um die fehlenden Werte einzufüllen.

# %%
# Vorhersage der fehlenden Werte
district.loc[district["no_of_crimes95"].isnull(), "no_of_crimes95"] = lin_reg_crime.predict(
    district[district["no_of_crimes95"].isnull()]["no_of_crimes96"].values.reshape(-1, 1)
)

district.loc[district["unemploy_rate95"].isnull(), "unemploy_rate95"] = lin_reg_unemploy.predict(
    district[district["unemploy_rate95"].isnull()]["unemploy_rate96"].values.reshape(-1, 1)
)

# %%
data_frames["district.csv"] = district

district.sample(n=5)

# %%
district.isnull().sum()

# %% [markdown]
# ### EDA
# Es gibt keine Duplikate und somit 77 unterschiedliche Namen der Distrikte. Diese sind auf 8 Regionen verteilt, wobei die meisten in south Moravia und die wenigsten in Prague liegen. Der Distrikt mit den wenigsten Einwohnern zählt 42821, im Vergleich zu demjenigen mit den meisten: 1204953, wobei die nächst kleinere Ortschaft 102609 Einwohner zählt. Weitere Informationen zu den vorhandenen Daten können aus dem [SweetViz Report](./reports/district.html) entnommen werden. 

# %%
# %%capture
svReport_district = sv.analyze(district)
svReport_district.show_html(filepath="./reports/district.html", open_browser=False)

# %% [markdown]
# ## Loan
# [//]: # (-.- .tabset)
#
# Der Datensatz `loan.csv` beinhaltet 682 Observationen mit den folgenden Informationen über die vergebenen Darlehen der Bank:  
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
# ### Transformation
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
# Assign the resulting DataFrame to a dictionary for storage
data_frames["loan.csv"] = loan

# Sample 5 random rows from the joined DataFrame
display(loan.sample(n=5))

# %%
# %%capture
svReport_loan = sv.analyze(loan)
svReport_loan.show_html(filepath="./reports/loan.html", open_browser=False)

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

# %% [markdown]
# ## Order
# [//]: # (-.- .tabset)
#
# Der Datensatz `order.csv` beinhaltet 6471 Observationen mit den folgenden Informationen über die Daueraufträge eines Kontos:  
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
# ### Transformation
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
    index="account_id", columns="k_symbol", values="amount", aggfunc="sum", observed=False
)

# Add prefix to column names
orders_pivot.columns = orders_pivot.columns

orders_pivot = orders_pivot.reset_index()
# Assuming data_frames is a dictionary for storing DataFrames
data_frames["order.csv"] = orders_pivot

# NaN to 0
data_frames["order.csv"] = data_frames["order.csv"].fillna(0)
# Sample 5 random rows from the merged DataFrame
data_frames["order.csv"].sample(n=5)

# %%
# %%capture
svReport_order = sv.analyze(order)
svReport_order.show_html(filepath="./reports/order.html", open_browser=False)

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

# %% [markdown]
# ## Trans
# [//]: # (-.- .tabset)
#
# Der Datensatz `trans.csv` beinhaltet 1056320 Observationen mit den folgenden Informationen über die Transaktionen eines Kontos:  
# - `trans_id`: die ID der Transaktion,
# - `account_id`: die Kontonummer des ausführenden Kontos,
# - `date`: das Datum,
# - `type`: der Typ (Einzahlung, Bezug)
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
# ### Transformation
# Die Kategorien für `type`, `operation` und `k_symbol` wurden übersetzt und die Datentypen korrigiert.  

# %%
trans["date"] = pd.to_datetime(trans["date"], format="%y%m%d")

# Update 'type' column
trans["type"] = trans["type"].replace({"PRIJEM": "credit", "VYDAJ": "withdrawal"})
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
trans.loc[trans['type'] == 'credit', 'amount'] = trans.loc[trans['type'] == 'credit', 'amount'] * (-1)

# %%
# Assign to a dictionary if needed (similar to list assignment in R)
data_frames["trans.csv"] = trans

# Sample 5 random rows from the DataFrame
trans.sample(n=5)

# %%
# %%capture
svReport_trans = sv.analyze(trans)
svReport_trans.show_html(filepath="./reports/trans.html", open_browser=False)

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

# %% [markdown]
# # Datenaufbereitung
#
# ## Statische Daten

# %%
# merge dataframes
static_data = (
    data_frames["disp.csv"]
    .add_suffix("_disp")
    .merge(
        data_frames["account.csv"].add_suffix("_account"),
        left_on="account_id_disp",
        right_on="account_id_account",
        how="left",
    )
    .merge(
        data_frames["card.csv"].add_suffix("_card"),
        left_on="disp_id_disp",
        right_on="disp_id_card",
        how="left",
    )
    .merge(
        data_frames["loan.csv"].add_suffix("_loan"),
        left_on="account_id_disp",
        right_on="account_id_loan",
        how="left",
    )
    .merge(
        data_frames["order.csv"].add_suffix("_order"),
        left_on="account_id_disp",
        right_on="account_id_order",
        how="left",
    )
)

# %%
static_data.columns

# %%
cols_to_replace_na = [
    "household_order",
    "insurance_payment_order",
    "loan_payment_order",
    "leasing_order",
    "unknown_order",
]

static_data[cols_to_replace_na] = static_data[
    cols_to_replace_na
].fillna(0)

# %% [markdown]
# ## Dropping of Junior Cards that are not on the edge to a normal card Analyse
#

# %%
# join district and client left join on district_id
static_data = static_data.merge(
    data_frames["district.csv"],
    left_on="district_id_account",
    right_on="district_id",
    how="left",
)

static_data

# %%
# merge client with suffix
static_data = static_data.merge(
    data_frames["client.csv"].add_suffix("_client"),
    left_on="client_id_disp",
    right_on="client_id_client",
    how="left",
)

# %%
static_data["has_card"] = ~static_data["card_id_card"].isna()

# Filter rows where 'has_card' is True
filtered_data = static_data[static_data["has_card"]]

# Check if there are duplicated 'account_id' in the filtered data
duplicated_account_id = filtered_data["account_id_account"].duplicated().sum()

print(duplicated_account_id)

# %% [markdown]
#
# ## Junior Cards removal

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

display(static_data)

# Filter rows where 'card_type' contains 'junior' (case insensitive)
junior_cards = static_data[
    static_data["type_card"].str.contains("junior", case=False, na=False)
]

display(junior_cards)

# Calculate age at issue
junior_cards["age_at_issue"] = (
                                       junior_cards["issued_card"] - junior_cards["birth_day_client"]
                               ).dt.days // 365

# Plot histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=junior_cards, x="age_at_issue", bins=20)
plt.title("Age distribution at issue date of junior cards")
plt.xlabel("Age at issue date")
plt.ylabel("Number of cards")
plt.show()

# %% [markdown]
# In the advertising campaign, we do not want to promote children's/junior cards (for whatever reasons). First, I looked at the distribution of age at issuance. Here I see that there are not many junior cards, nor are the cards issued at a late age.

# %%
num_accounts_before = len(static_data)
# Filter rows where 'card_type' does not contain 'junior' (case insensitive)
non_transactional_data = static_data[
    ~static_data["type_card"].str.contains("junior", case=False, na=False)
]
num_accounts_after = len(non_transactional_data)
num_junior_cards = num_accounts_before - num_accounts_after
print(f"Number of junior cards removed: {num_junior_cards}")

# %% [markdown]
# ## Transaktionen

# %% [markdown]
# ## Zusammenfügen der Daten

# %%
# %%capture
import subprocess
import pathlib

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
# Update html output
# jupyter nbconvert --to html --template pj AML_MC.ipynb

# %% [markdown]
# # Referenzen
# - [1] [Applied Predictive Modelling](http://link.springer.com/10.1007/978-1-4614-6849-3)
