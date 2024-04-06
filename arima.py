import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.dates as mdates
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
            n_jobs=-1,
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
        # dump(model_fit, "fitted_model.joblib")

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


def forecast_evaluation(
    actual_data: pd.Series, forecasted_data: pd.Series
) -> tuple[float, float]:
    """
    Evaluate forecast accuracy using MAE, RMSE, and optionally MAPE.

    Args:
        actual_data (pd.Series): Actual time series data.
        forecasted_data (pd.Series): Forecasted time series data.

    Returns:
        float: Mean Absolute Error (MAE).
        float: Root Mean Squared Error (RMSE).
    """
    mae = mean_absolute_error(actual_data, forecasted_data)
    rmse = sqrt(mean_squared_error(actual_data, forecasted_data))
    logger.info(f"Evaluation Metrics:\nMAE: {mae}\nRMSE: {rmse}")
    return mae, rmse


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


def forecast_and_plot(
    model_fit,
    series: pd.Series,
    forecast_steps: int,
    start_date: pd.Timestamp,
    title_prefix: str = "",
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Generate forecast from the model, dynamically handle frequency mismatches,
    and plot against actual data with enhanced visualization using Plotly,
    including a title prefix for context.

    Args:
        model_fit (SARIMAX or ARIMA): Fitted time series model.
        series (pd.Series): Time series data.
        forecast_steps (int): Number of steps to forecast.
        start_date (pd.Timestamp): Start date of the forecast period.
        title_prefix (str): Optional prefix for the plot title to provide context.
    """
    forecast_results = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_results.predicted_mean
    forecast_conf_int = forecast_results.conf_int()

    # Adjust index for forecast mean and confidence interval to match the start date and frequency
    forecast_index = pd.date_range(
        start=start_date, periods=forecast_steps, freq=pd.infer_freq(series.index)
    )
    forecast_mean.index = forecast_index
    forecast_conf_int.index = forecast_index

    # Creating a Plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Adding actual data series
    fig.add_trace(
        go.Scatter(x=series.index, y=series, mode="lines+markers", name="Actual Data"),
        secondary_y=False,
    )

    # Adding forecast mean series
    fig.add_trace(
        go.Scatter(
            x=forecast_mean.index,
            y=forecast_mean,
            mode="lines+markers",
            name="Forecast",
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
            showlegend=True,
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

    # Displaying the plot
    fig.show()

    return forecast_mean, forecast_conf_int


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


def evaluate_forecast_performance(actual_data: pd.Series, forecast_mean: pd.Series):
    """
    Evaluate and log the performance of the forecast using MAE and RMSE.

    Args:
        actual_data (pd.Series): The actual observed values.
        forecast_mean (pd.Series): The forecasted values.
    """
    # Ensure both Series have the same date index for fair comparison
    if not actual_data.empty and not forecast_mean.empty:
        common_dates = actual_data.index.intersection(forecast_mean.index)
        mae = mean_absolute_error(
            actual_data.loc[common_dates], forecast_mean.loc[common_dates]
        )
        rmse = sqrt(
            mean_squared_error(
                actual_data.loc[common_dates], forecast_mean.loc[common_dates]
            )
        )
        logger.info(f"Forecast Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    else:
        logger.info("No actual data available for the forecasted period to evaluate.")

    return mae, rmse


def main():
    logger.info("Starting the forecasting process...")
    file_path = "../data/raw_data/2024-03-06-11-22_co2_data.csv"

    # Load Data
    raw_data = load_data(file_path)
    logger.info(f"Original data shape: {raw_data.shape}")

    # Preprocess Data
    cleaned_data = preprocess_data(raw_data, method="ffill")
    logger.info(f"Cleaned data shape: {cleaned_data.shape}")

    # Resample Data to Hourly Frequency
    hourly_data = resample_data(cleaned_data, resample_frequency="H")
    logger.info(f"Hourly data shape: {hourly_data.shape}")

    # Select Training Period
    training_start_date = hourly_data.index.min()
    training_end_date = training_start_date + pd.Timedelta(weeks=3)  # Adjust as needed
    training_data = hourly_data.loc[
        training_start_date:training_end_date, "_value"
    ]  # Adjust column name if needed
    logger.info(f"Training data shape: {training_data.shape}")

    # Check Stationarity and Fit Model
    stationary_data, differencing_needed = check_stationarity(training_data)
    if differencing_needed:
        training_data = stationary_data
    model_fit = fit_time_series_model(
        training_data, seasonal_period=24
    )  # Using 24 for hourly data to capture daily seasonality

    # Forecast
    forecast_steps = 24  # Next 24 hours
    forecast_start_date = adjust_forecast_start_date(training_end_date)
    forecasted_values, forecast_confidence_interval = forecast_and_plot(
        model_fit, training_data, forecast_steps, forecast_start_date
    )

    # Evaluate Forecast
    actual_data = hourly_data.loc[
        forecast_start_date : forecast_start_date
        + pd.Timedelta(hours=forecast_steps - 1),
        "_value",
    ]  # Adjust column name if needed
    mae, rmse = evaluate_forecast_performance(actual_data, forecasted_values)
    logger.info(f"Forecast Evaluation - MAE: {mae}, RMSE: {rmse}")

    # Visualizations
    visualize_data_analysis(training_data, "Hourly Training Data - ")
    plot_acf_pacf(forecasted_values - actual_data, "Hourly Forecast Residuals - ")


if __name__ == "__main__":
    main()
