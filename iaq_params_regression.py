import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


# Function to load and clean individual datasets
def load_and_clean_data(file_path, new_column_name):
    df = pd.read_csv(file_path, comment="#", header=0)
    df["_time"] = pd.to_datetime(df["_time"])
    df.set_index("_time", inplace=True)
    df = df[["_value"]]  # Keep only the measurement values
    df.rename(
        columns={"_value": new_column_name}, inplace=True
    )  # Rename the _value column
    return df


data_files = {
    "CO2": "../data/raw_data/victors_office/2024-03-06-11-22_co2_data.csv",
    "temperature": "../data/raw_data/victors_office/2024-03-06-11-28_temperature_data.csv",
    "humidity": "../data/raw_data/victors_office/2024-03-06-11-26_humidity_data.csv",
}

df = pd.DataFrame()
for parameter, file_path in data_files.items():
    temp_df = load_and_clean_data(file_path, parameter)
    df = temp_df if df.empty else df.join(temp_df, how="inner")

print(df.head())
print(df.describe())
print(df.isnull().sum())
df.dropna(inplace=True)
df["time_of_day"] = df.index.hour + df.index.minute / 60

# Visualization: Pairplot to visualize relationships
sns.pairplot(df)
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()

# Adding a constant for the intercept
X = sm.add_constant(df[["CO2", "time_of_day"]])

# Separate regressions for each dependent variable
results = {}
dependent_vars = ["temperature", "humidity"]
for var in dependent_vars:
    model = sm.OLS(df[var], X).fit()
    results[var] = model
    print(f"Results for {var}:")
    print(model.summary())
    print("\n\n")

# Splitting dataset for training and testing using sklearn
X_train, X_test, Y_train, Y_test = train_test_split(
    df[["CO2", "time_of_day"]],
    df[["temperature", "humidity"]],
    test_size=0.2,
    random_state=42,
)

# Creating and fitting the Multivariate Linear Regression model
multi_lm = LinearRegression()
multi_lm.fit(X_train, Y_train)

# Make predictions
Y_pred = multi_lm.predict(X_test)

# Evaluation
mse_temperature = mean_squared_error(Y_test["temperature"], Y_pred[:, 0])
mse_humidity = mean_squared_error(Y_test["humidity"], Y_pred[:, 1])

print(f"Mean Squared Error for Temperature: {mse_temperature}")
print(f"Mean Squared Error for Humidity: {mse_humidity}")

r2_temperature = r2_score(Y_test["temperature"], Y_pred[:, 0])
r2_humidity = r2_score(Y_test["humidity"], Y_pred[:, 1])

print(f"R^2 Score for Temperature: {r2_temperature}")
print(f"R^2 Score for Humidity: {r2_humidity}")
