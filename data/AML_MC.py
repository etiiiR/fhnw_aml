# %%
title:  AML Challenge 2024
date: "Generated: {{ datetime.now().strftime('%Y-%m-%d') }}"
author: Etienne Roulet Alex Sha
output:
    general:
        input_jinja: true
    html:
        code_folding: hide
        code_tools: true
        theme: readable

# %%
%pip install -r ../requirements.txt

# %%
%load_ext pretty_jupyter

# %% [markdown]
# # Introduction
# 
# In diesem Notebook wenden wir Applied Machine Learning (AML) Techniken an, um effektive Strategien für personalisierte Kreditkarten-Werbekampagnen zu entwickeln. Unser Ziel ist es, mithilfe von Kunden- und Transaktionsdaten präzise Modelle zu erstellen, die die Wahrscheinlichkeit des Kreditkartenkaufs vorhersagen.

# %% [markdown]
# ## Lib Importing
# 

# %%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from itables import init_notebook_mode
from datetime import datetime

init_notebook_mode(all_interactive=True)

# %% [markdown]
# ## Load the Data
# 
# 

# %%
account = pd.read_csv("account.csv", sep=";", dtype={"date": "str"})
account["date"] = pd.to_datetime(account["date"], format="%y%m%d")

card = pd.read_csv("card.csv", sep=";", dtype={"issued": "str"})
# Man kann die Zeit weglassen da immer 00:00:00
card["issued"] = pd.to_datetime(card["issued"].str[:6], format="%y%m%d")

client = pd.read_csv("client.csv", sep=";")
disp = pd.read_csv("disp.csv", sep=";")
district = pd.read_csv("district.csv", sep=";")

loan = pd.read_csv("loan.csv", sep=";", dtype={"date": "str"})
loan["date"] = pd.to_datetime(loan["date"], format="%y%m%d")


order = pd.read_csv("order.csv", sep=";")

trans = pd.read_csv("trans.csv", sep=";", dtype={"date": "str", "bank": "str"})
trans["date"] = pd.to_datetime(trans["date"], format="%y%m%d")
trans
# count 'NaN' in each column from trans

# %% [markdown]
# ## EDA

# %% [markdown]
# ### Account

# %%
account

# %% [markdown]
# ### Card

# %%
card

# %% [markdown]
# ### Client

# %%
client

# %% [markdown]
# ### Disp

# %%
disp

# %% [markdown]
# ### District

# %%
district

# %% [markdown]
# ### Loan

# %%
loan

# %% [markdown]
# ### Order

# %%
order

# %% [markdown]
# ### Trans

# %%
trans

# %% [markdown]
# ## Transformations

# %%
data_frames = {}

# %% [markdown]
# ### Account
# 

# %%
# Frequency Transformation
account["frequency"] = account["frequency"].replace(
    {
        "POPLATEK MESICNE": "monthly issuance",
        "POPLATEK TYDNE": "weekly issuance",
        "POPLATEK PO OBRATU": "issuance after transaction",
    }
)

# Rename Column
account = account.rename(columns={"frequency": "issuance_statement_frequency"})

# Convert Date Column to datetime format
account["date"] = pd.to_datetime(account["date"])

# Assuming 'data_frames' is a dictionary of DataFrames
data_frames["account.csv"] = account

# Sample 5 random rows
account.sample(n=5)

# %% [markdown]
# ### Card

# %%
card["issued"] = pd.to_datetime(card["issued"], format="mixed")
data_frames["card.csv"] = card

# %% [markdown]
# ### Client

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


# Anwenden der Funktionen und Erstellen neuer Spalten
client["gender"], client["birth_day"] = zip(
    *client["birth_number"].apply(parse_details)
)
client["age"] = client["birth_day"].apply(calculate_age)

data_frames["client.csv"] = client

# Auswahl spezifischer Spalten für die finale DataFrame (optional, je nach Bedarf)
# Sample 5 random rows
client.sample(n=5)

# %% [markdown]
# ### Disp

# %%
data_frames["disp.csv"] = disp

# random sample
disp.sample(n=5)

# %% [markdown]
# ### District
# 

# %% [markdown]
# - A1 district_id/district code
# - A2 district name
# - A3 region
# - A4 no. of inhabitants
# - A5 no. of municipalities with inhabitants < 499
# - A6 no. of municipalities with inhabitants 500-1999 A7 no. of municipalities with inhabitants 2000-9999
# - A8 no. of municipalities with inhabitants >10000
# - A9 no. of cities
# - A10 ratio of urban inhabitants
# - A11 average salary
# - A12 unemploymant rate ’95
# - A13 unemploymant rate ’96
# - A14 no. of enterpreneurs per 1000 inhabitants
# - A15 no. of commited crimes ’95
# - A16 no. of commited crimes ’96

# %%
import pandas as pd

# Assuming 'district' is your pandas DataFrame

# Renaming and selecting columns
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

data_frames["district.csv"] = district

district.sample(n=5)
district

# %%
# find the ? in the district dataframe
district[district.isin(["?"]).any(axis=1)]

# %%
# replace the ? with NaN
district = district.replace("?", np.nan)

# %%
# replace the NaN with the mean of the column no_of_crimes95 and unemploy_rate95
district["no_of_crimes95"] = district["no_of_crimes95"].astype(float)
district["unemploy_rate95"] = district["unemploy_rate95"].astype(float)
district["no_of_crimes95"] = district["no_of_crimes95"].fillna(
    district["no_of_crimes95"].mean()
)
district["unemploy_rate95"] = district["unemploy_rate95"].fillna(
    district["unemploy_rate95"].mean()
)
# check if there are still NaN values in no_of_crimes95 and unemploy_rate95
district[district.isin([np.nan]).any(axis=1)]

# %% [markdown]
# ### Loan

# %%
# Convert the 'date' column to datetime format
loan["date"] = pd.to_datetime(loan["date"], format="mixed")

# Mutate the 'status' column based on conditions
loan["status"] = loan["status"].map(
    {
        "A": "contract finished",
        "B": "finished contract, loan not paid",
        "C": "running contract",
        "D": "client in debt",
    }
)

# Group by 'account_id', calculate the number of loans, and sort the results
num_of_loan_df = (
    loan.groupby("account_id")
    .size()
    .reset_index(name="num_of_loan")
    .sort_values(by="num_of_loan", ascending=False)
)

# Display the resulting DataFrame
num_of_loan_df

# %%
# Perform an inner join between 'loan' and 'num_of_loan_df' on 'account_id'
loan = pd.merge(loan, num_of_loan_df, on="account_id", how="inner")

# Assign the resulting DataFrame to a dictionary for storage
data_frames["loan.csv"] = loan

# Sample 5 random rows from the joined DataFrame
loan.sample(n=100)

# %% [markdown]
# ### Order
# 

# %%
# Assuming 'order' and 'account' DataFrames are already loaded

# Correctly map and fill missing values in 'k_symbol' column
order["k_symbol"] = (
    order["k_symbol"]
    .map({"POJISTNE": "Insurance Payment", "SIPO": "Household", "UVER": "Loan Payment"})
    .fillna("UNKNOWN")
)

# Merge with 'account_id_df' to ensure all accounts are represented
order = pd.merge(account[["account_id"]], order, on="account_id", how="left")

# After merging, fill missing values that may have been introduced
order["k_symbol"] = order["k_symbol"].fillna("UNKNOWN")
order["amount"] = order["amount"].fillna(0)
order["has_order"] = ~order.isna().any(axis=1)

# Aggregate 'amount' information
aggregated_amount = (
    order.groupby("account_id")
    .agg(
        sum_amount=("amount", "sum"),
        mean_amount=("amount", "mean"),
        median_amount=("amount", "median"),
        min_amount=("amount", "min"),
        max_amount=("amount", "max"),
        num_of_orders=("amount", lambda x: (x != 0).sum()),
    )
    .reset_index()
)
aggregated_amount["has_order"] = aggregated_amount["sum_amount"] != 0

# Create dummies for 'k_symbol' and ensure aggregation by 'account_id'
dummies_k_symbol = pd.get_dummies(
    order[["account_id", "k_symbol"]], columns=["k_symbol"], prefix="", prefix_sep=""
)
dummies_k_symbol = dummies_k_symbol.groupby("account_id").sum().reset_index()

# Merge 'aggregated_amount' and 'dummies_k_symbol'
merged_order = pd.merge(
    aggregated_amount, dummies_k_symbol, on="account_id", how="left"
)

# Assuming data_frames is a dictionary for storing DataFrames
data_frames["order.csv"] = merged_order

# Sample 5 random rows from the merged DataFrame
merged_order.sample(n=5)

# %% [markdown]
# ### Trans
# 

# %%
# Convert 'date' from string to datetime
trans["date"] = pd.to_datetime(trans["date"])

# Update 'type' column
trans["type"] = trans["type"].replace({"PRIJEM": "credit", "VYDAJ": "withdrawal"})

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

# Assign to a dictionary if needed (similar to list assignment in R)

data_frames["trans.csv"] = trans

# Sample 5 random rows from the DataFrame
trans.sample(n=1000)
trans

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

# %% [markdown]
# Wee see that there is a steep line in 1995-10 so there are two transactions, this we have to clean.

# %% [markdown]
# ## D&Q

# %%
# Check for missing values in each DataFrame
for df_name, df in data_frames.items():
    print(f"Missing values in {df_name}:")
    print(df.isna().sum().sum())  # Sum of all missing values in the DataFrame

# %% [markdown]
# ## Dropping of Junior Cards that are not on the edge to a normal card
# 

# %%
data_frames.keys()

# %%
# join the dataframes


# %%
!jupytext --to notebook clean-tech-rag.py
# commxand with os
os.system("jupytext --to notebook AML_MC.py")

# %%
!jupyter nbconvert --to html --template pj "clean-tech-rag.ipynb"
# command with os
os.system("jupyter nbconvert --to html --template pj AML_MC.ipynb")


