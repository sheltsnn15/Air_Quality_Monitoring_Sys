# data_handling.py
from datetime import datetime
import os

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_measured_iaq_data(file_path: str) -> pd.DataFrame:
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


def clean_survey_data(survey_df):
    # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    survey_df = survey_df.copy()
    """
    Cleans survey data for analysis from the provided Excel format.
    Standardizes Likert-scale values from 1-5 to -2 to 2 and computes the average score.
    Also, applies standardization to binary responses, treating 'Yes' as +1 and 'No' as -1.

    Parameters:
    survey_df (pd.DataFrame): The survey data as a pandas DataFrame.

    Returns:
    pd.DataFrame: DataFrame with the standardized and averaged responses for each question.
    """

    # Ensure columns for scoring are present
    expected_cols = [1, 2, 3, 4, 5, "Yes", "No"]
    missing_cols = [col for col in expected_cols if col not in survey_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Replace NaNs in numerical columns where scoring happens
    survey_df.loc[:, [1, 2, 3, 4, 5, "Yes", "No"]] = survey_df.loc[
        :, [1, 2, 3, 4, 5, "Yes", "No"]
    ].fillna(0)

    # Helper function to calculate weighted scores safely
    def calculate_weighted_score(row, weights):
        if row.sum() == 0:
            return None  # Avoid division by zero
        return (row * weights).sum() / row.sum()

    weights = pd.Series([-2, -1, 0, 1, 2], index=[1, 2, 3, 4, 5])
    survey_df.loc[:, "Comfort_Humidity_Score"] = survey_df[[1, 2, 3, 4, 5]].apply(
        calculate_weighted_score, axis=1, weights=weights
    )
    survey_df.loc[:, "Level_Alertness_Score"] = survey_df[[1, 2, 3, 4, 5]].apply(
        calculate_weighted_score, axis=1, weights=weights
    )
    survey_df.loc[:, "Air_Freshness_Score"] = survey_df[[1, 2, 3, 4, 5]].apply(
        calculate_weighted_score, axis=1, weights=weights
    )

    # Standardize binary responses by mapping 'Yes' to +1 and 'No' to -1, and compute a net score
    survey_df["Temperature_Discomfort_Score"] = (
        survey_df["Yes"] * 1 + survey_df["No"] * -1
    ) / (survey_df["Yes"] + survey_df["No"]).replace(
        0, pd.NA
    )  # Prevent division by zero

    return survey_df


def load_and_categorize_survey_data(directory):
    survey_data_by_room = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx"):
                date_segment, room_code, time_range = extract_survey_details(root, file)
                df = pd.read_excel(os.path.join(root, file), skiprows=1)
                timestamp = generate_timestamp(date_segment, time_range)

                # Detect the start of each question block by the pattern in 'Unnamed: 0'
                question_starts = df[
                    df["Unnamed: 0"].notnull() & df["Question"].notnull()
                ].index

                # Process each question block
                for start in question_starts:
                    # Assuming the end index based on the observed pattern (5 rows per question)
                    end = start + 5  # Adjust if necessary

                    # Extract the block of rows for the question
                    question_block = df.iloc[start:end]

                    # Clean the data
                    cleaned_data = clean_survey_data(question_block)

                    # Drop columns with all NaN values
                    cleaned_data.dropna(axis=1, how="all", inplace=True)

                    # Assign the timestamp
                    cleaned_data.loc[:, "Timestamp"] = timestamp

                    # Append to the room's DataFrame
                    if room_code not in survey_data_by_room:
                        survey_data_by_room[room_code] = pd.DataFrame()
                    survey_data_by_room[room_code] = pd.concat(
                        [survey_data_by_room[room_code], cleaned_data],
                        ignore_index=True,
                    )

    # Reset index for all DataFrames in the dictionary after all data has been processed
    for room in survey_data_by_room:
        survey_data_by_room[room].reset_index(drop=True, inplace=True)

    return survey_data_by_room


def extract_survey_details(root, file):
    date_segment = os.path.basename(root)
    room_time = file.replace(".xlsx", "").split("-")
    room_code, time_range = room_time[0], room_time[1].split("_")
    return date_segment, room_code, time_range


def generate_timestamp(date_segment, time_range):
    formatted_date_segment = datetime.strptime(date_segment, "%d-%m-%Y").strftime(
        "%Y-%m-%d"
    )
    datetime_str = f"{formatted_date_segment}T{time_range[0]}:00:00.000000000Z"
    return pd.to_datetime(datetime_str)


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

    logger.info(f"Resampling data to {resample_frequency} frequency...")

    # Resample '_value' data and forward fill to handle missing values
    resampled_df = df["_value"].resample(resample_frequency).mean().ffill()

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


def get_maximum_weeks_available(data):
    """
    Calculate the maximum number of weeks available in the dataset from the start_date.

    Args:
        data (pd.DataFrame): The time series dataset.

    Returns:
        int: The maximum number of weeks available in the dataset from the start_date.
    """
    start_date = data.index.min()  # Automatically use the earliest date in the dataset
    end_date = data.index.max()
    duration = end_date - start_date
    max_weeks = duration.days // 7  # Convert the duration to weeks
    return max_weeks
