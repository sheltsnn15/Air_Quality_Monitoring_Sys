import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def handle_missing_values(df, method="ffill"):  # Handle missing values in DataFrame.
    print("Handling missing values:")
    print("Initial missing value count:")
    print(df.isnull().sum())  # Printing the count of missing values.

    if method == "ffill":
        df.fillna(
            method="ffill", inplace=True
        )  # Filling missing values with the previous valid value (forward fill).
    elif method == "mean":
        df.fillna(
            df.mean(), inplace=True
        )  # Filling missing values with the mean of the column.
    else:
        df.dropna(inplace=True)  # Dropping rows with missing values.

    print(
        "After handling missing values:"
    )  # Showing the count of missing values after handling.
    print(df.isnull().sum())  # Printing the count of missing values after handling.
    return df


def transform_data(df):  # Transform data and check for stationarity.
    print("Transforming data:")
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

    return df["_value"], was_differenced


def apply_arima(original_df, transformed_df, was_differenced):
    print("Applying ARIMA:")
    print(transformed_df.head())

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
    forecast_steps = 5  # You can change this to forecast more steps ahead
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(
        start=transformed_df.index[-1], periods=forecast_steps + 1, freq="T"
    )[1:]
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    # If the data was differenced, inverse the transformation
    if was_differenced:
        forecast_mean = forecast_mean.cumsum() + original_df["_value"].iloc[-1]

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


def main():
    file_path = "../data/raw_data/2024-03-06-11-22_co2_data.csv"

    # Step 1: Read CSV
    """ @parameters used in read csv func
    The comment='#' parameter helps pandas ignore the commented lines in the CSV file
    parse_dates with index_col will treat the _time column as a datetime index
    """
    df = pd.read_csv(
        file_path, comment="#", parse_dates=["_time"], index_col="_time", skiprows=3
    )
    df.drop(columns=["Unnamed: 0", "result"], inplace=True)

    cleaned_df = handle_missing_values(df)

    # Convert the index to a PeriodIndex with minutely frequency BEFORE the transformation
    cleaned_df.index = pd.to_datetime(cleaned_df.index)

    # Proceed with transforming data and applying ARIMA
    transformed_series, was_differenced = transform_data(cleaned_df)

    # Pass both the original cleaned data frame and the transformed series
    apply_arima(cleaned_df, transformed_series, was_differenced)


if __name__ == "__main__":
    main()
