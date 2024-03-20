import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_file_path = "../data/raw_data/2024-03-06-11-28_temperature_data.csv"

# read in data, ignore irrelevant 'result' and 'table' columns
df = pd.read_csv(csv_file_path, skiprows=3)
df = df[df["_field"] == "temperature"]
# parse date columns to 'datetime' obj's
df["_time"] = pd.to_datetime(df["_time"])

# clean data
values = df.dropna().reset_index(drop=True)  # This will remove rows with any NaN values
# Box plot
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["_value"])
plt.title("Boxplot of CO2 Concentration")
plt.xlabel("CO2 Concentration (ppm)")
plt.show()

# Histogram
plt.figure(figsize=(10, 5))
sns.histplot(df["_value"], kde=True)  # KDE will add a density plot on top
plt.title("Histogram of CO2 Concentration")
plt.xlabel("CO2 Concentration (ppm)")
plt.show()
