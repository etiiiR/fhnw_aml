# %% [raw]
# title:  AML Challenge 2024
# date: "Generated: {{ datetime.now().strftime('%Y-%m-%d') }}"
# author: Etienne Roulet Alex Sha
# output:
#     general:
#         input_jinja: true
#     html:
#         code_folding: hide
#         code_tools: true
#         theme: readable

# %%
# %pip install -r ../requirements.txt

# %%
# %load_ext pretty_jupyter

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
from IPython.display import display

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
order

# %%
# Assuming 'order' and 'account' DataFrames are already loaded

# Correctly map and fill missing values in 'k_symbol' column
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

# Merge with 'account_id_df' to ensure all accounts are represented
order = pd.merge(account[["account_id"]], order, on="account_id", how="left")

# After merging, fill missing values that may have been introduced
order["k_symbol"] = order["k_symbol"].fillna("unknown")
order["amount"] = order["amount"].fillna(0)
order["has_order"] = ~order.isna().any(axis=1)

orders_pivot = order.pivot_table(
    index="account_id", columns="k_symbol", values="amount", aggfunc="sum"
)

# Add prefix to column names
orders_pivot.columns = orders_pivot.columns


orders_pivot = orders_pivot.reset_index()
# Assuming data_frames is a dictionary for storing DataFrames
data_frames["order.csv"] = orders_pivot

# NaN to 0
data_frames["order.csv"] = data_frames["order.csv"].fillna(0)
# Sample 5 random rows from the merged DataFrame
data_frames["order.csv"].sample(n=10)

# %%
data_frames["order.csv"].columns

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


# %%
# merge dataframes


non_transactional_data = (
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
non_transactional_data.columns

# %% With the merge we got NaN values in the columns, we have to clean them
cols_to_replace_na = [
    "household_order",
    "insurance_payment_order",
    "loan_payment_order",
    "leasing_order",
    "unknown_order",
]

non_transactional_data[cols_to_replace_na] = non_transactional_data[
    cols_to_replace_na
].fillna(0)


# %%
# ## Dropping of Junior Cards that are not on the edge to a normal card Analyse
#
# join disctrict and client left join on district_id
non_transactional_data = non_transactional_data.merge(
    data_frames["district.csv"],
    left_on="district_id_account",
    right_on="district_id",
    how="left",
)

# %%
non_transactional_data

# %%
# merge client with suffix
non_transactional_data = non_transactional_data.merge(
    data_frames["client.csv"].add_suffix("_client"),
    left_on="client_id_disp",
    right_on="client_id_client",
    how="left",
)


# %%
non_transactional_data["has_card"] = ~non_transactional_data["card_id_card"].isna()

# Filter rows where 'has_card' is True
filtered_data = non_transactional_data[non_transactional_data["has_card"]]

# Check if there are duplicated 'account_id' in the filtered data
duplicated_account_id = filtered_data["account_id_account"].duplicated().sum()

print(duplicated_account_id)



# %% [markdown]
# ### Junior Cards removal

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta

display(non_transactional_data)

# Filter rows where 'card_type' contains 'junior' (case insensitive)
junior_cards = non_transactional_data[
    non_transactional_data["type_card"].str.contains("junior", case=False, na=False)
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
num_accounts_before = len(non_transactional_data)
# Filter rows where 'card_type' does not contain 'junior' (case insensitive)
non_transactional_data = non_transactional_data[
    ~non_transactional_data["type_card"].str.contains("junior", case=False, na=False)
]
num_accounts_after = len(non_transactional_data)
num_junior_cards = num_accounts_before - num_accounts_after
print(f"Number of junior cards removed: {num_junior_cards}")


# %% [markdown]
# ## Convert the Notebook always to a py file and vis versa

# %%
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
# command with os
os.system("jupyter nbconvert --to html --template pj AML_MC.ipynb")


