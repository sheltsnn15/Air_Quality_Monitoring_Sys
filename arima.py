import matplotlib.pyplot as plt
import time
import pandas as pd
import pmdarima as pm
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt


def handle_missing_values(df, method="ffill"):  # Handle missing values in DataFrame.
    print("Handling missing values:")
    print("Initial missing value count:")
    print(df.isnull().sum())  # Printing the count of missing values.

    start_time = time.time()  # Start timing
    if method == "ffill":
        df.fillna(
            method=method, inplace=True
        )  # Filling missing values with the previous valid value (forward fill).
    elif method == "mean":
        df.fillna(
            df.mean(), inplace=True
        )  # Filling missing values with the mean of the column.
    else:
        df.dropna(inplace=True)  # Dropping rows with missing values.
    end_time = time.time()  # End timing
    print(
        "Time taken for handling missing values: {:.4f} seconds".format(
            end_time - start_time
        )
    )

    print(
        "After handling missing values:"
    )  # Showing the count of missing values after handling.
    print(df.isnull().sum())  # Printing the count of missing values after handling.
    return df


def transform_data(df):  # Transform data and check for stationarity.
    print("Transforming data:")
    start_time = time.time()  # Start timing
    result = adfuller(
        df["_value"]
    )  # Using the Augmented Dickey-Fuller test to check for stationarity.
    print("ADF Statistic: %f" % result[0])  # Printing the ADF Statistic.
    print("p-value: %f" % result[1])  # Printing the p-value.
    # Return a flag indicating whether the data was differenced, so the apply_arima function knows the state of the
    # data it's working with
    was_differenced = False

    if result[1] > 0.05:  # Checking if the p-value is greater than 0.05.
        print("Data is not stationary, need to difference")
        df["_value"] = (
            df["_value"].diff().dropna()
        )  # Performing differencing on the data.
        was_differenced = True
    else:
        print("Data is stationary, no differencing needed")
    end_time = time.time()  # End timing
    print(
        "Time taken for transforming data: {:.4f} seconds".format(end_time - start_time)
    )

    return df["_value"], was_differenced


def apply_arima(original_df, transformed_df, was_differenced, forecast_steps=180):
    print("Applying ARIMA:")
    start_time = time.time()  # Start timing

    # Automated parameter selection using pmdarima
    auto_arima_model = pm.auto_arima(
        transformed_df,
        seasonal=False,
        suppress_warnings=True,
        stepwise=False,  # Consider stepwise=True for faster execution if applicable
        n_jobs=-1,  # Use all available CPU cores
    )
    order = auto_arima_model.order
    print(f"Selected ARIMA order: {order}")

    # Fitting ARIMA model
    model = ARIMA(transformed_df, order=order)
    model_fit = model.fit()
    print(model_fit.summary())

    # Forecasting the next steps ahead
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(
        start=transformed_df.index[-1], periods=forecast_steps + 1, freq="T"
    )[1:]
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    # If the data was differenced, inverse the transformation
    if was_differenced:
        forecast_mean = forecast_mean.cumsum() + original_df["_value"].iloc[-1]

    end_time = time.time()  # End timing
    print("Time taken for applying ARIMA: {:.4f} seconds".format(end_time - start_time))

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(original_df.index, original_df["_value"], label="Observed")
    plt.plot(forecast_index, forecast_mean, label="Forecast")
    plt.fill_between(
        forecast_index,
        forecast_conf_int.iloc[:, 0],
        forecast_conf_int.iloc[:, 1],
        color="k",
        alpha=0.1,
    )
    plt.title("ARIMA Forecast")
    plt.xlabel("Time")
    plt.ylabel("CO2 Concentration")
    plt.legend()
    plt.show()

    # Return the fitted model
    return model_fit


def load_data(
    file_path,
):  #     Load data from a CSV file and perform initial preprocessing.
    try:
        df = pd.read_csv(
            file_path, comment="#", parse_dates=["_time"], index_col="_time", skiprows=3
        )
        df.drop(columns=["Unnamed: 0", "result"], inplace=True)
        return df
    except Exception as e:
        raise e


def visualize_data(
    df, aggregation="weekly"
):  #     Visualize data for exploratory analysis.
    plt.figure(figsize=(10, 5))
    plot_acf(df["_value"], lags=1440)
    plt.title("Autocorrelation Function")
    plt.show()

    plt.figure(figsize=(10, 5))
    plot_pacf(df["_value"], lags=1440)
    plt.title("Partial Autocorrelation Function")
    plt.show()

    if aggregation == "daily":
        grouped = df["_value"].groupby(df.index.hour).agg(list).apply(pd.Series).T
        title = "Hourly Variance in CO2 Levels (Daily)"
        period = 24  # Daily period
    elif aggregation == "weekly":
        grouped = df["_value"].groupby(df.index.dayofweek).agg(list).apply(pd.Series).T
        title = "Weekly Variance in CO2 Levels"
        period = 10080  # Weekly period
    else:
        raise ValueError("Invalid aggregation. Please choose 'daily' or 'weekly'.")

    grouped.boxplot()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("CO2 Concentration")
    plt.show()

    if aggregation == "weekly":
        decomposition = seasonal_decompose(
            df["_value"], model="additive", period=period
        )
        decomposition.plot()
        plt.title("Seasonal Decomposition (Weekly)")
        plt.show()


def evaluate_forecast(validation_df, forecast_mean):
    # Comparing actual vs predicted values
    comparison_df = pd.DataFrame(
        {"actual": validation_df["_value"], "predicted": forecast_mean}
    )
    print(comparison_df)

    # Calculating the Mean Absolute Error (MAE)
    mae = np.mean(np.abs(comparison_df["predicted"] - comparison_df["actual"]))
    print(f"Mean Absolute Error: {mae}")

    # Calculating the Root Mean Squared Error (RMSE)
    mse = mean_squared_error(comparison_df["actual"], comparison_df["predicted"])
    rmse = sqrt(mse)
    print(f"Root Mean Squared Error: {rmse}")

    # Plotting actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df["actual"], label="Actual")
    plt.plot(comparison_df["predicted"], label="Predicted", alpha=0.7)
    plt.title("Actual vs Predicted CO2 Levels")
    plt.xlabel("Time")
    plt.ylabel("CO2 Concentration")
    plt.legend()
    plt.show()

    return mae, rmse


def main():
    file_path = "../data/raw_data/2024-03-06-11-22_co2_data.csv"

    try:
        df = load_data(file_path)
        cleaned_df = handle_missing_values(df)
        cleaned_df.index = pd.to_datetime(cleaned_df.index)

        # Define the training dataset
        # Use the first 80% of the dataset for training
        split_point = int(len(cleaned_df) * 0.8)
        training_df = cleaned_df.iloc[:split_point]

        # Define the validation dataset
        validation_df = cleaned_df.iloc[split_point:]

        visualize_data(training_df)

        transformed_series, was_differenced = transform_data(training_df)

        # Apply ARIMA on the training dataset
        model_fit = apply_arima(training_df, transformed_series, was_differenced)

        # Generate forecasts for the validation set
        forecast_steps = len(
            validation_df
        )  # Set forecast steps to the size of validation set
        forecast_results = model_fit.get_forecast(steps=forecast_steps)
        forecast_mean = forecast_results.predicted_mean

        # Evaluate the forecast
        mae, rmse = evaluate_forecast(validation_df, forecast_mean)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
