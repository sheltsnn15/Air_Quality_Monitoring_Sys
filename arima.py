import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import time
import logging
from joblib import dump

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean the time series data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned time series data.
    """
    try:
        logger.info("Loading data from file...")
        df = pd.read_csv(
            file_path, parse_dates=["_time"], index_col="_time", skiprows=3
        )
        df.drop(
            columns=[col for col in ["Unnamed: 0", "result"] if col in df.columns],
            inplace=True,
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    return df


def preprocess_data(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Preprocess data by applying the specified method for handling missing values.

    Args:
        df (pd.DataFrame): Time series data.
        method (str): Method for handling missing values. Options: 'ffill', 'mean', or 'drop'.

    Returns:
        pd.DataFrame: Preprocessed time series data.
    """
    try:
        logger.info("Preprocessing data...")
        preprocessed_df = df.copy()
        if method == "ffill":
            preprocessed_df.ffill(inplace=True)
        elif method == "mean":
            preprocessed_df.fillna(preprocessed_df.mean(), inplace=True)
        else:
            preprocessed_df.dropna(inplace=True)
    except Exception as e:
        logger.error(f"Failed during data preprocessing: {e}")
        raise
    return preprocessed_df


def check_stationarity(
    series: pd.Series, alpha: float = 0.05
) -> tuple[pd.Series, bool]:
    """
    Check for stationarity in the time series data using the Augmented Dickey-Fuller test.

    Args:
        series (pd.Series): Time series data.
        alpha (float): Significance level.

    Returns:
        pd.Series: Stationary time series data.
        bool: True if differencing is needed, False otherwise.
    """
    result = adfuller(series, autolag="AIC")
    logger.info(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    if result[1] > alpha:
        logger.info("Data is not stationary. Differencing will be applied.")
        return series.diff().dropna(), True
    else:
        logger.info("Data is stationary. No differencing needed.")
        return series, False


def resample_data(df: pd.DataFrame, resample_frequency: str = "H") -> pd.DataFrame:
    """
    Resample the time series data to the specified frequency by taking the mean of values for each period.

    Args:
        df (pd.DataFrame): Time series data.
        resample_frequency (str): The frequency for resampling.
                                  'H' for hourly, 'D' for daily, 'W' for weekly.

    Returns:
        pd.DataFrame: Data resampled to the specified frequency.
    """
    valid_frequencies = ["H", "D", "W"]
    if resample_frequency not in valid_frequencies:
        raise ValueError(
            f"Resample frequency must be one of {valid_frequencies}, got '{resample_frequency}'"
        )

    logger.info(f"Resampling data to {resample_frequency} frequency...")

    # Resample '_value' data and forward fill to handle missing values
    resampled_df = df["_value"].resample(resample_frequency).mean().ffill().to_frame()

    return resampled_df


def fit_time_series_model(time_series: pd.Series, seasonal_period: int) -> SARIMAX:
    """
    Fit a time series model (ARIMA or SARIMAX) based on the detected seasonality.

    Args:
        df (pd.Series): Time series data.
        forecast_steps (int): Number of steps to forecast.
        seasonal_period (int): Length of the seasonal period.

    Returns:
        SARIMAX or ARIMA: Fitted time series model.
    """
    try:
        # Logging: Indicate the start of model fitting process
        logger.info("Fitting time series model...")
        start_time = time.time()

        # Use pmdarima's auto_arima to automatically select the best model parameters
        model = pm.auto_arima(
            time_series,
            seasonal=seasonal_period is not None,
            m=seasonal_period or 1,
            suppress_warnings=True,
            stepwise=False,
            n_jobs=8,
        )

        # Check if seasonal order is not (0, 0, 0, 0) to determine whether to use SARIMAX or ARIMA
        # (0, 0, 0, 0) represents no seasonality in SARIMAX model (no seasonal differencing)
        # If seasonal_order is not (0, 0, 0, 0), it indicates the presence of seasonality
        # In such cases, SARIMAX model is used, otherwise ARIMA model is used
        if model.seasonal_order != (0, 0, 0, 0):
            logger.info("SARIMAX model selected.")
            # Fit SARIMAX model
            model_fit = SARIMAX(
                time_series, order=model.order, seasonal_order=model.seasonal_order
            ).fit(disp=False)
        else:
            logger.info("ARIMA model selected.")
            # Fit ARIMA model
            model_fit = ARIMA(time_series, order=model.order).fit(disp=False)

        # Logging: Indicate the completion of model fitting process along with the time taken
        logger.info(
            f"Model fitting completed in {time.time() - start_time:.2f} seconds."
        )

        # Save the fitted model
        dump(model_fit, "fitted_model.joblib")

    except Exception as e:
        # Logging: Log error if model fitting fails
        logger.error(f"Model fitting failed: {e}")
        raise
    return model_fit


def visualize_seasonal_decompose(
    series: pd.Series, title_prefix: str = "", period: int = 24
) -> None:
    """
    Perform and plot seasonal decomposition of the series using Plotly to assist in identifying seasonality.

    Args:
        series (pd.Series): Time series data.
        title_prefix (str): Optional prefix for the plot title to indicate the context.
        period (int): The period of the seasonality.
    """
    decomposition = seasonal_decompose(series, model="additive", period=period)

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"{title_prefix}Observed",
            f"{title_prefix}Trend",
            f"{title_prefix}Seasonal",
            f"{title_prefix}Residual",
        ),
    )

    # Plot the observed data
    fig.add_trace(
        go.Scatter(
            x=series.index, y=decomposition.observed, mode="lines", name="Observed"
        ),
        row=1,
        col=1,
    )

    # Plot the trend component
    fig.add_trace(
        go.Scatter(x=series.index, y=decomposition.trend, mode="lines", name="Trend"),
        row=2,
        col=1,
    )

    # Plot the seasonal component
    fig.add_trace(
        go.Scatter(
            x=series.index, y=decomposition.seasonal, mode="lines", name="Seasonal"
        ),
        row=3,
        col=1,
    )

    # Plot the residual component
    fig.add_trace(
        go.Scatter(
            x=series.index, y=decomposition.resid, mode="lines", name="Residual"
        ),
        row=4,
        col=1,
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"{title_prefix}Seasonal Decomposition",
        template="plotly_white",
    )

    fig.show()


def plot_acf_pacf(series: pd.Series, title_prefix: str = ""):
    """
    Plot Autocorrelation and Partial Autocorrelation.

    Args:
        series (pd.Series): Time series data or model residuals.
        title_prefix (str): Optional prefix for the plot titles to indicate if these are raw data or residuals.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_acf(series, ax=axes[0])
    plot_pacf(series, ax=axes[1])
    axes[0].set_title(f"{title_prefix}Autocorrelation")
    axes[1].set_title(f"{title_prefix}Partial Autocorrelation")
    plt.tight_layout()
    plt.show()


def visualize_boxplots(
    series: pd.Series, title: str = "Boxplot of Time Series Data", frequency: str = "H"
):
    """
    Generate boxplots or scatter plots of the time series data for specified frequency: Hourly, Daily, Weekly, or Monthly.

    Args:
        series (pd.Series): Time series data with a datetime index.
        title (str): Title for the plot.
        frequency (str): Frequency for aggregating data. 'H' for hourly, 'D' for daily, 'W' for weekly, 'M' for monthly.
    """
    df_for_boxplot = series.copy().to_frame()
    df_for_boxplot["Year"] = df_for_boxplot.index.year
    df_for_boxplot["Month"] = df_for_boxplot.index.month
    df_for_boxplot["Day"] = df_for_boxplot.index.day
    df_for_boxplot["Hour"] = df_for_boxplot.index.hour
    if frequency == "W":
        df_for_boxplot["Week"] = df_for_boxplot.index.isocalendar().week

    # Map frequency to plot data and labels
    freq_to_label = {"M": "Month", "W": "Week", "D": "Day", "H": "Hour"}
    x_label = freq_to_label.get(frequency, "Time")

    plt.figure(figsize=(12, 6))
    if frequency in ["D", "H"]:  # Use scatter plot for dense data points
        sns.stripplot(
            data=df_for_boxplot,
            x=freq_to_label[frequency],
            y=series.name,
            jitter=0.25,
            size=2.5,
        )
    else:  # Use boxplot for less dense data points
        sns.boxplot(data=df_for_boxplot, x=freq_to_label[frequency], y=series.name)
    plt.title(title)
    plt.xlabel(x_label)
    plt.xticks(rotation=45)  # Helps with readability, especially for large datasets
    plt.grid(True)
    plt.show()


def visualize_data_analysis(series: pd.Series, title_prefix: str = ""):
    """
    Perform comprehensive data visualizations, including raw data plotting,
    autocorrelation, partial autocorrelation, and seasonal decomposition.

    Args:
        series (pd.Series): Time series data with a datetime index.
        title_prefix (str): Optional prefix for the plot titles to provide context.
    """
    # Plot the raw data
    plt.figure(figsize=(12, 4))
    plt.plot(series, label=f"{title_prefix}Raw Data")
    plt.title(f"{title_prefix}Time Series Data")
    plt.legend()
    plt.show()

    # ACF and PACF plots
    plot_acf_pacf(series, title_prefix)

    # Seasonal decomposition
    visualize_seasonal_decompose(series, title_prefix)


def handle_outliers(series, method="cap"):
    """
    Handle outliers in a pandas Series using the Interquartile Range (IQR) method.

    Args:
        series (pd.Series): Time series data.
        method (str): Method for handling outliers ('remove', 'cap', 'none').

    Returns:
        pd.Series: Series with handled outliers.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    if method == "cap":
        return series.clip(lower=lower_bound, upper=upper_bound)
    elif method == "remove":
        return series[(series >= lower_bound) & (series <= upper_bound)]
    else:
        return series  # 'none' method, do not handle outliers


def generate_forecast(
    model_fit, series: pd.Series, forecast_steps: int, start_date: pd.Timestamp
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Generate forecast from the fitted time series model.

    Args:
        model_fit (SARIMAX or ARIMA): Fitted time series model.
        series (pd.Series): Time series data.
        forecast_steps (int): Number of steps to forecast.
        start_date (pd.Timestamp): Start date of the forecast period.

    Returns:
        tuple[pd.Series, pd.DataFrame]: Tuple containing forecasted mean values and confidence intervals.
    """
    forecast_results = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_results.predicted_mean
    forecast_conf_int = forecast_results.conf_int()

    # Adjust index for forecast mean and confidence interval to match the start date and frequency
    forecast_index = pd.date_range(
        start=start_date, periods=forecast_steps, freq=pd.infer_freq(series.index)
    )  # Adjust 'D' as needed
    forecast_mean.index = forecast_index
    forecast_conf_int.index = forecast_index

    return forecast_mean, forecast_conf_int


def save_forecast_to_csv(
    forecast_mean: pd.Series,
    forecast_conf_int: pd.DataFrame,
    file_name: str = "forecast.csv",
):
    # Combine forecast mean and confidence intervals into a single DataFrame
    forecast_df = pd.concat([forecast_mean, forecast_conf_int], axis=1)
    forecast_df.columns = [
        "Forecast",
        "Confidence_Interval_Lower",
        "Confidence_Interval_Upper",
    ]

    # Save the DataFrame to a CSV file
    forecast_df.to_csv(file_name)
    logger.info(f"Forecast saved to {file_name}")


def save_evaluation_metrics(
    mae: float, rmse: float, metrics_file: str = "evaluation_metrics.csv"
):
    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({"MAE": [mae], "RMSE": [rmse]})

    # Check if the file exists to append or write
    if os.path.exists(metrics_file):
        with open(metrics_file, "a") as f:
            metrics_df.to_csv(f, header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)

    logger.info(f"Evaluation metrics saved to {metrics_file}")


def plot_forecast_vs_actual(
    series: pd.Series,
    forecast_mean: pd.Series,
    forecast_conf_int: pd.DataFrame,
    title_prefix: str = "",
):
    """
    Plot actual data vs forecasted data with confidence intervals using Plotly.

    Args:
        series (pd.Series): Actual time series data.
        forecast_mean (pd.Series): Forecasted mean values.
        forecast_conf_int (pd.DataFrame): Forecasted confidence intervals.
        title_prefix (str): Optional prefix for the plot title to provide context.
    """
    # Creating a Plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Adding actual data series
    fig.add_trace(
        go.Scatter(x=series.index, y=series, mode="lines", name="Actual Data"),
        secondary_y=False,
    )

    # Adding forecast mean series
    fig.add_trace(
        go.Scatter(
            x=forecast_mean.index, y=forecast_mean, mode="lines", name="Forecast"
        ),
        secondary_y=False,
    )

    # Adding forecast confidence interval area
    fig.add_trace(
        go.Scatter(
            x=forecast_conf_int.index.tolist() + forecast_conf_int.index[::-1].tolist(),
            y=forecast_conf_int.iloc[:, 0].tolist()
            + forecast_conf_int.iloc[:, 1][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval",
        ),
        secondary_y=False,
    )

    # Setting plot titles and labels
    fig.update_layout(
        title=f"{title_prefix}Actual vs Forecasted Data",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
        template="plotly_white",
    )

    fig.show()


def adjust_forecast_start_date(training_end_date: pd.Timestamp) -> pd.Timestamp:
    """
    Adjust the start date for the forecast to avoid weekends.

    Args:
        training_end_date (pd.Timestamp): The end date of the training period.

    Returns:
        pd.Timestamp: Adjusted start date for the forecast.
    """
    forecast_start_date = training_end_date + pd.Timedelta(days=1)
    return forecast_start_date


def evaluate_forecast(
    actual_data: pd.Series, forecast_data: pd.Series
) -> tuple[float, float]:
    """
    Evaluate and log the performance of the forecast using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
    This method ensures that both series have the same date index for a fair comparison and handles cases where data might be missing.

    Args:
        actual_data (pd.Series): The actual observed values.
        forecast_data (pd.Series): The forecasted values.

    Returns:
        float: Mean Absolute Error (MAE).
        float: Root Mean Squared Error (RMSE).
    """
    # Ensure both Series are not empty and share the same index for fair comparison
    if not actual_data.empty and not forecast_data.empty:
        common_dates = actual_data.index.intersection(forecast_data.index)
        if not common_dates.empty:
            mae = mean_absolute_error(
                actual_data.loc[common_dates], forecast_data.loc[common_dates]
            )
            rmse = sqrt(
                mean_squared_error(
                    actual_data.loc[common_dates], forecast_data.loc[common_dates]
                )
            )
            logger.info(f"Forecast Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        else:
            logger.info(
                "No overlapping dates between actual and forecast data for evaluation."
            )
            mae, rmse = float("nan"), float("nan")
    else:
        logger.info("One or both of the Series are empty, cannot perform evaluation.")
        mae, rmse = float("nan"), float("nan")

    return mae, rmse


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
        f"{resample_frequency[
            0
        ]} Training Data - ",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    co2_raw_data_file_path = "../data/raw_data/2024-03-06-11-22_co2_data.csv"
    humidity_raw_data_file_path = "../data/raw_data/2024-03-06-11-26_humidity_data.csv"
    temp_raw_data_file_path = "../data/raw_data/2024-03-06-11-28_temperature_data.csv"

    main(logger, humidity_raw_data_file_path)
