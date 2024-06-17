# time_series_modeling.py
import os

import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import time
from joblib import dump
import logging
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from math import sqrt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fit_time_series_model(time_series: pd.Series, seasonal_period: int) -> SARIMAX:
    """
    Fit a time series model (ARIMA or SARIMAX) based on the detected seasonality.

    Args:
        time_series (pd.Series): Time series data.
        seasonal_period (int): Length of the seasonal period.

    Returns:
        SARIMAX or ARIMA: Fitted time series model.
    """
    try:
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
            D=0,
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
            model_fit = ARIMA(time_series, order=model.order).fit()

        training_time = time.time() - start_time
        logger.info(f"Model fitting completed in {training_time:.2f} seconds.")

        # Save the fitted model
        # dump(model_fit, "fitted_model.joblib")

    except Exception as e:
        logger.error(f"Model fitting failed: {e}")
        raise
    return model_fit, training_time


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
    )
    forecast_mean.index = forecast_index
    forecast_conf_int.index = forecast_index

    return forecast_mean, forecast_conf_int


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


def save_forecast_to_csv(
    forecast_mean: pd.Series,
    forecast_conf_int: pd.DataFrame,
    file_name: str = "forecast.csv",
):
    """
    Save forecast mean and confidence intervals to a CSV file.
    """
    forecast_df = pd.concat([forecast_mean, forecast_conf_int], axis=1)
    forecast_df.columns = [
        "Forecast",
        "Confidence_Interval_Lower",
        "Confidence_Interval_Upper",
    ]
    forecast_df.to_csv(file_name)
    logger.info(f"Forecast saved to {file_name}")


def save_evaluation_metrics(
    mae: float, rmse: float, metrics_file: str = "evaluation_metrics.csv"
):
    """
    Save MAE and RMSE evaluation metrics to a CSV file.
    """
    metrics_df = pd.DataFrame({"MAE": [mae], "RMSE": [rmse]})
    if os.path.exists(metrics_file):
        with open(metrics_file, "a") as f:
            metrics_df.to_csv(f, header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Evaluation metrics saved to {metrics_file}")


def adjust_forecast_start_date(training_end_date: pd.Timestamp) -> pd.Timestamp:
    """
    Adjust the start date for the forecast to avoid weekends.
    """
    forecast_start_date = training_end_date + pd.Timedelta(days=1)
    return forecast_start_date
