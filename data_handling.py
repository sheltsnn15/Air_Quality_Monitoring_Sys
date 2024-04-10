# data_processing.py
import pandas as pd
import logging
from statsmodels.tsa.stattools import adfuller

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


def get_maximum_weeks_available(data, start_date):
    """
    Calculate the maximum number of weeks available in the dataset from the start_date.

    Args:
        data (pd.DataFrame): The time series dataset.
        start_date (pd.Timestamp): The date from which to start counting.

    Returns:
        int: The maximum number of weeks available in the dataset from the start_date.
    """
    end_date = data.index.max()
    duration = end_date - start_date
    max_weeks = duration.days // 7  # Convert the duration to weeks
    return max_weeks
