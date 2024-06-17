import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

from data_handling import (
    load_measured_iaq_data,
    preprocess_data,
    resample_data,
    get_maximum_weeks_available,
)
from time_series_modeling import (
    fit_time_series_model,
    generate_forecast,
    check_stationarity,
    evaluate_forecast,
    adjust_forecast_start_date,
    save_evaluation_metrics,
    save_forecast_to_csv,
)
from visualisation import (
    visualize_data_analysis,
    plot_forecast_vs_actual,
    visualize_boxplots,
    plot_acf_pacf,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_accuracy_over_training_sizes(
    logger, file_path, start_weeks=4, end_weeks=10, step_weeks=1, forecast_horizon=24
):
    """
    Evaluate the prediction accuracy of an ARIMA model over different training sizes.
    Args:
        logger: A logging.Logger object for logging messages.
        file_path: The path to the CSV file containing the time series data.
        start_weeks: The starting size of the training data in weeks.
        end_weeks: The maximum size of the training data in weeks, adjusted not to exceed dataset size.
        step_weeks: The step size to increase the training data size for each iteration.
        forecast_horizon: The number of hours to forecast ahead from the end of each training period.
    """
    accuracies = []
    training_times = []  # List to store training times
    training_sizes = range(start_weeks, end_weeks + 1, step_weeks)

    raw_data = load_measured_iaq_data(file_path)
    cleaned_data = preprocess_data(raw_data, method="ffill")
    resampled_data = resample_data(cleaned_data, resample_frequency="D")

    # Plot ACF and PACF for the entire series to understand data dependencies
    plot_acf_pacf(
        resampled_data, "Overall Data - "
    )  # Assuming '_value' is the column name

    # Ensure end_weeks does not exceed the dataset size
    max_weeks = get_maximum_weeks_available(resampled_data)
    if end_weeks > max_weeks:
        logger.warning(
            f"End weeks parameter exceeds the dataset size. Adjusting to {max_weeks} weeks."
        )
        end_weeks = max_weeks

    for weeks in training_sizes:
        training_start_date = resampled_data.index.min()
        training_end_date = training_start_date + pd.Timedelta(weeks=weeks)

        # Adjust the forecast start date to the next period
        forecast_start_date = adjust_forecast_start_date(training_end_date)
        forecast_end_date = forecast_start_date + pd.Timedelta(
            hours=forecast_horizon - 1
        )

        # Check if there's enough data for forecasting
        if forecast_end_date > resampled_data.index.max():
            logger.info(
                f"Insufficient data for forecasting beyond {weeks} weeks of training. Stopping."
            )
            break

        training_data = resampled_data.loc[training_start_date:training_end_date]

        stationary_data, _ = check_stationarity(training_data)
        model_fit, training_time = fit_time_series_model(
            stationary_data, seasonal_period=24
        )

        training_times.append(training_time)  # Store the training time

        forecasted_values, _ = generate_forecast(
            model_fit, stationary_data, forecast_horizon, forecast_start_date
        )
        actual_forecast_data = resampled_data.loc[forecast_start_date:forecast_end_date]

        if len(forecasted_values) != len(actual_forecast_data):
            forecasted_values = forecasted_values[: len(actual_forecast_data)]

        accuracy = mean_absolute_error(actual_forecast_data, forecasted_values)
        accuracies.append(accuracy)
        logger.info(f"Weeks of training: {weeks}, MAE: {accuracy:.4f}")

    # Plotting results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:red"
    ax1.set_xlabel("Weeks of Training Data")
    ax1.set_ylabel("Prediction Accuracy (MAE)", color=color)
    ax1.plot(training_sizes, accuracies, marker="o", linestyle="-", color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color = "tab:blue"
    ax2.set_ylabel("Training Time (seconds)", color=color)
    ax2.plot(training_sizes, training_times, marker="s", linestyle="--", color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Prediction Accuracy and Training Time vs. Training Size")
    plt.grid(True)
    plt.show()


def main(logger, file_path):
    logger.info("Starting the forecasting process...")

    # Load Data
    raw_data = load_measured_iaq_data(file_path)

    # Inform the user about the dataset's time frame and interval
    dataset_start_date = raw_data.index.min()
    dataset_end_date = raw_data.index.max()
    data_collection_interval = pd.infer_freq(raw_data.index)
    logger.info(f"Dataset starts from: {dataset_start_date}")
    logger.info(f"Dataset ends on: {dataset_end_date}")
    logger.info(
        f"Data is collected at {data_collection_interval if data_collection_interval else 'irregular'} intervals."
    )

    # Preprocess Data
    cleaned_data = preprocess_data(raw_data, method="ffill")

    # Handling Outliers
    # cleaned_data["_value"] = handle_outliers(
    #    cleaned_data["_value"], method="cap"
    # )  # Adjust method as needed

    # User Interaction for Resampling and Analysis Period
    resample_frequency = input(
        "Enter the resampling frequency you want to choose (H for hourly, D for daily, W for weekly, M for monthly): "
    ).upper()
    weeks_interval = int(
        input("Enter the size of the dataset in weeks intervals for analysis: ")
    )
    training_start_date_input = input(
        "Enter the training start date (YYYY-MM-DD) that is available in the dataset: "
    )
    training_start_date = pd.to_datetime(training_start_date_input)

    # Check if the dataset's datetime index is tz-aware and adjust training_start_date accordingly
    if raw_data.index.tz is not None:
        training_start_date = training_start_date.tz_localize(raw_data.index.tz)

    # Define the mapping from resample frequency to typical seasonal period
    frequency_to_seasonal_period = {
        "H": 24,  # Daily seasonality in hourly data
        "D": 7,  # Weekly seasonality in daily data
        "W": 52,  # Yearly seasonality in weekly data
    }

    suggested_seasonal_period = frequency_to_seasonal_period.get(resample_frequency, 1)
    seasonal_period = int(
        input(
            f"Enter the seasonal period (suggested: {suggested_seasonal_period} based on {resample_frequency} frequency): "
        )
        or suggested_seasonal_period
    )

    # Resample Data to User-Specified Frequency
    new_resampled_data = resample_data(
        cleaned_data, resample_frequency=resample_frequency
    )

    visualize_boxplots(
        new_resampled_data,  # Ensure "_value" is the correct column name
        title="Distribution of Time Series Data - " + resample_frequency,
        frequency=resample_frequency[0],
    )

    # Select Training Period based on User Input
    training_end_date = training_start_date + pd.Timedelta(weeks=weeks_interval)
    training_data = new_resampled_data.loc[training_start_date:training_end_date]

    # Check Stationarity and Fit Model
    stationary_data, differencing_needed = check_stationarity(training_data)
    training_data_final = stationary_data if differencing_needed else training_data
    model_fit, _ = fit_time_series_model(training_data_final, seasonal_period)

    # Forecast based on User-Specified Forecasting Steps
    forecasting_steps = int(
        input(
            "Enter the number of steps to forecast (e.g., 24 for the next 24 hours if hourly): "
        )
    )
    forecast_start_date = adjust_forecast_start_date(training_end_date)
    forecasted_values, forecast_confidence_interval = generate_forecast(
        model_fit, training_data_final, forecasting_steps, forecast_start_date
    )
    print("Forcasted vals: ", forecasted_values.head())

    forecast_end_date = forecast_start_date + pd.Timedelta(hours=forecasting_steps - 1)

    print(
        "Forcasted start date: ",
        forecast_start_date,
        " Forcested end date: ",
        forecast_end_date,
    )
    # Evaluate Forecast
    actual_forecast_period_data = new_resampled_data.loc[
        forecast_start_date:forecast_end_date
    ]
    mae, rmse = evaluate_forecast(actual_forecast_period_data, forecasted_values)
    logger.info(f"Forecast Evaluation - MAE: {mae}, RMSE: {rmse}")

    # After forecasting and evaluation
    save_forecast_to_csv(forecasted_values, forecast_confidence_interval)
    save_evaluation_metrics(mae, rmse)

    # Fetch the extended actual data for visualization directly from new_resampled_data
    extended_actual_data = new_resampled_data.loc[training_start_date:forecast_end_date]

    plot_forecast_vs_actual(
        extended_actual_data,
        forecasted_values,
        forecast_confidence_interval,
        "Training + Forecast Period - ",
    )

    visualize_data_analysis(
        extended_actual_data,
        f"{resample_frequency[0]} Training Data - ",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    co2_raw_data_file_path = (
        "../data/raw_data/victors_office/2024-03-06-11-22_co2_data.csv"
    )
    humidity_raw_data_file_path = (
        "../data/raw_data/victors_office/2024-03-06-11-26_humidity_data.csv"
    )
    temp_raw_data_file_path = (
        "../data/raw_data/victors_office/2024-03-06-11-28_temperature_data.csv"
    )

    # main(logger, humidity_raw_data_file_path)
    evaluate_accuracy_over_training_sizes(
        logger=logger, file_path=co2_raw_data_file_path
    )
