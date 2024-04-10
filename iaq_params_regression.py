import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


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


# Load and clean datasets
co2_df = load_and_clean_data("../data/raw_data/2024-03-06-11-22_co2_data.csv", "CO2")
temp_df = load_and_clean_data(
    "../data/raw_data/2024-03-06-11-28_temperature_data.csv", "temperature"
)
humidity_df = load_and_clean_data(
    "../data/raw_data/2024-03-06-11-26_humidity_data.csv", "humidity"
)

# Sequentially join the dataframes
df = co2_df.join(temp_df, how="inner").join(humidity_df, how="inner")

# Check the first few rows to ensure data looks correct
print(df.head())

# Quick overview
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Drop any rows with missing values (if any)
df.dropna(inplace=True)

# Adding 'time_of_day' as a feature
df["time_of_day"] = df.index.hour + df.index.minute / 60

# Visualization: Pairplot to visualize relationships
sns.pairplot(df)
plt.show()

# Multivariate Regression Analysis with sklearn
X = df[["CO2", "time_of_day"]]
Y = df[["temperature", "humidity"]]

# Splitting dataset for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Creating and fitting the Multivariate Linear Regression model
multi_lm = linear_model.LinearRegression()
multi_lm.fit(X_train, Y_train)

# Make predictions
Y_pred = multi_lm.predict(X_test)

# Evaluation
# Since we have two dependent variables, we evaluate the model for each
mse_temperature = mean_squared_error(Y_test["temperature"], Y_pred[:, 0])
mse_humidity = mean_squared_error(Y_test["humidity"], Y_pred[:, 1])

print(f"Mean Squared Error for Temperature: {mse_temperature}")
print(f"Mean Squared Error for Humidity: {mse_humidity}")

# Correct way to compute R^2 Score for each variable separately
r2_temperature = r2_score(Y_test["temperature"], Y_pred[:, 0])
r2_humidity = r2_score(Y_test["humidity"], Y_pred[:, 1])

print(f"R^2 Score for Temperature: {r2_temperature}")
print(f"R^2 Score for Humidity: {r2_humidity}")
