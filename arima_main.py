import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

from data_handling import load_data, preprocess_data, resample_data
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
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_accuracy_over_training_sizes(
    logger, file_path, start_weeks=1, end_weeks=10, step_weeks=1, forecast_horizon=24
):
    """
    Evaluate the prediction accuracy of an ARIMA model over different training sizes.

    This function iteratively increases the size of the training dataset, forecasts a fixed interval
    ahead (e.g., 24 hours), and records the prediction accuracy for each training size.

    Args:
        logger: A logging.Logger object for logging messages.
        file_path: The path to the CSV file containing the time series data.
        start_weeks: The starting size of the training data in weeks.
        end_weeks: The maximum size of the training data in weeks.
        step_weeks: The step size to increase the training data by each iteration.
        forecast_horizon: The number of hours to forecast ahead from the end of each training period.

    The function plots the relationship between training sizes (in weeks) and prediction accuracy.
    """
    accuracies = []
    training_sizes = range(start_weeks, end_weeks + 1, step_weeks)

    raw_data = load_data(file_path)
    cleaned_data = preprocess_data(raw_data, method="ffill")

    resample_frequency = "D"
    new_resampled_data = resample_data(
        cleaned_data, resample_frequency=resample_frequency
    )

    for weeks in training_sizes:
        training_start_date = new_resampled_data.index.min()
        training_end_date = training_start_date + pd.Timedelta(weeks=weeks)

        forecast_start_date = training_end_date + pd.Timedelta(hours=1)
        forecast_end_date = forecast_start_date + pd.Timedelta(
            hours=forecast_horizon - 1
        )

        training_data = new_resampled_data.loc[
            training_start_date:training_end_date, "_value"
        ]
        stationary_data, _ = check_stationarity(training_data)

        model_fit = fit_time_series_model(stationary_data, seasonal_period=24)

        forecasted_values, _ = generate_forecast(
            model_fit, stationary_data, forecast_horizon, forecast_start_date
        )

        # Align forecasted_values with actual data based on timestamps
        forecasted_values.index = pd.date_range(
            start=forecast_start_date, periods=len(forecasted_values), freq="D"
        )

        actual_forecast_data = new_resampled_data.loc[
            forecast_start_date:forecast_end_date, "_value"
        ]

        # Ensure both datasets are not empty and have overlapping periods
        if (
            not actual_forecast_data.empty
            and not forecasted_values.empty
            and actual_forecast_data.index.intersection(forecasted_values.index).any()
        ):
            common_index = forecasted_values.index.intersection(
                actual_forecast_data.index
            )
            forecasted_values_aligned = forecasted_values.loc[common_index]
            actual_forecast_data_aligned = actual_forecast_data.loc[common_index]

            accuracy = mean_absolute_error(
                actual_forecast_data_aligned, forecasted_values_aligned
            )
            accuracies.append(accuracy)
            logger.info(f"Weeks of training: {weeks}, Accuracy: {accuracy}")
        else:
            logger.warning(
                f"Weeks of training: {weeks}, no overlapping data for forecast and actual data. Skipping accuracy calculation."
            )

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(list(training_sizes), accuracies, marker="o", linestyle="-")
    plt.xlabel("Weeks of Training Data")
    plt.ylabel("Prediction Accuracy (MAE)")
    plt.title("Prediction Accuracy vs. Training Size")
    plt.grid(True)
    plt.show()


def main(logger, file_path):
    logger.info("Starting the forecasting process...")

    # Load Data
    raw_data = load_data(file_path)

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
        new_resampled_data["_value"],  # Ensure "_value" is the correct column name
        title="Distribution of Time Series Data - " + resample_frequency,
        frequency=resample_frequency[
            0
        ],  # Assuming resample_frequency is correctly set to one of 'H', 'D', 'W', 'M'
    )

    # Select Training Period based on User Input
    training_end_date = training_start_date + pd.Timedelta(weeks=weeks_interval)
    training_data = new_resampled_data.loc[
        training_start_date:training_end_date, "_value"
    ]

    # Check Stationarity and Fit Model
    stationary_data, differencing_needed = check_stationarity(training_data)
    training_data_final = stationary_data if differencing_needed else training_data
    model_fit = fit_time_series_model(training_data_final, seasonal_period)

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
        forecast_start_date:forecast_end_date, "_value"
    ]
    mae, rmse = evaluate_forecast(actual_forecast_period_data, forecasted_values)
    logger.info(f"Forecast Evaluation - MAE: {mae}, RMSE: {rmse}")

    # After forecasting and evaluation
    save_forecast_to_csv(forecasted_values, forecast_confidence_interval)
    save_evaluation_metrics(mae, rmse)

    # Fetch the extended actual data for visualization directly from new_resampled_data
    extended_actual_data = new_resampled_data.loc[
        training_start_date:forecast_end_date, "_value"
    ]

    # Plot Forecast vs Actual
    plot_forecast_vs_actual(
        extended_actual_data,
        forecasted_values,
        forecast_confidence_interval,
        "Training + Forecast Period - ",
    )

    # Visualizations
    visualize_data_analysis(
        extended_actual_data,
        f"{resample_frequency[0]} Training Data - ",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    co2_raw_data_file_path = "../data/raw_data/2024-03-06-11-22_co2_data.csv"
    humidity_raw_data_file_path = "../data/raw_data/2024-03-06-11-26_humidity_data.csv"
    temp_raw_data_file_path = "../data/raw_data/2024-03-06-11-28_temperature_data.csv"

    # main(logger, humidity_raw_data_file_path)
    evaluate_accuracy_over_training_sizes(
        logger=logger,
        file_path=co2_raw_data_file_path,
        start_weeks=4,  # Adjusting to start from 4 weeks, roughly a month
        end_weeks=52,  # Example: extending to a year for comprehensive analysis
        step_weeks=4,  # Incrementing in monthly intervals
        forecast_horizon=24,  # Forecasting 24 hours ahead
    )
