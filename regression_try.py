import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_handling import (
    preprocess_data,
    resample_data,
    load_and_categorize_survey_data,
    load_measured_iaq_data,
)


# Constants
SENSOR_DATA_DIRECTORY = "../data/raw_data/lecture_rooms/"
SURVEY_DATA_DIRECTORY = "../data/Surveys/"
OPERATIONAL_HOURS = range(9, 17)  # 9 AM to 5 PM
TIMEZONE = "UTC"
NEAREST_HOUR_TOLERANCE = pd.Timedelta("1H")


def plot_actual_survey_vs_measured_co2(iaq_data, survey_data, data_type):
    # Sort the survey data by Timestamp
    survey_data_sorted = survey_data.sort_values("Timestamp")

    # Merge the CO2 data with the survey data on the Timestamp
    merged_data = pd.merge_asof(
        iaq_data.sort_values("Timestamp"),  # Sort the IAQ data as well,
        survey_data_sorted,
        on="Timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1H"),
    )

    # Create subplots for each surveyed data type
    num_plots = len(data_types)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(15, 7 * num_plots))

    for ax, data_type in zip(axes, data_types):
        color = "tab:green"
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{data_type} Level", color=color)
        ax.scatter(
            merged_data["Timestamp"],
            merged_data["_value"],
            color=color,
            label=f"Measured CO2 Data",
            alpha=0.7,
        )
        ax.tick_params(axis="y", labelcolor=color)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper left")

        ax2 = ax.twinx()  # Create a second axes that shares the same x-axis
        color = "tab:red"
        ax2.set_ylabel(
            f"{data_type} Score",
            color=color,
        )
        ax2.scatter(
            merged_data["Timestamp"],
            merged_data[data_type],
            color=color,
            label=f"Perceived {data_type}",
            alpha=0.7,
            marker="x",
        )
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.legend(loc="upper right")

    plt.title("Measured CO2 vs. Surveyed Perceptions Over Time")
    fig.tight_layout()  # To make sure that the labels don't get cut off
    plt.grid(True)
    plt.show()


def calculate_correlation(iaq_data, perceived_data):
    combined_df = pd.merge_asof(
        iaq_data.sort_index(),
        perceived_data.sort_index(),
        on="Timestamp",
        direction="nearest",
        tolerance=NEAREST_HOUR_TOLERANCE,
    )
    return combined_df.corr().iloc[0, 1]


def load_and_preprocess_iaq_data():
    iaq_data = {"CO2": [], "humidity": [], "temperature": []}
    for subdir, _, files in os.walk(SENSOR_DATA_DIRECTORY):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(subdir, file)
                data_type = (
                    "CO2"
                    if "co2" in file.lower()
                    else ("humidity" if "humidity" in file.lower() else "temperature")
                )
                iaq_data_piece = load_measured_iaq_data(file_path)
                iaq_data_piece = preprocess_data(iaq_data_piece)
                iaq_data_piece.index.rename("Timestamp", inplace=True)
                iaq_data[data_type].append(iaq_data_piece)
    return iaq_data


def resample_iaq_data(iaq_data, resample_frequency="H"):
    hourly_iaq_data = {}
    for data_type, data_list in iaq_data.items():
        combined_data = pd.concat(data_list)
        combined_data.index = pd.to_datetime(combined_data.index)
        hourly_data = resample_data(
            combined_data, resample_frequency=resample_frequency
        )
        hourly_iaq_data[data_type] = hourly_data
    return hourly_iaq_data


if __name__ == "__main__":
    start_date = "2024-03-01"
    end_date = "2024-04-30"
    room_identifier = "C214"

    # Load and preprocess IAQ data from the specified directory
    iaq_data = load_and_preprocess_iaq_data()

    # Resample the IAQ data to hourly data for better comparison with survey data
    hourly_iaq_data = resample_iaq_data(iaq_data, "H")

    # Load and categorize actual survey data
    survey_data_by_room = load_and_categorize_survey_data(SURVEY_DATA_DIRECTORY)
    # Confirm that 'Comfort_Humidity_Score' is in the DataFrame
    print(
        survey_data_by_room[room_identifier][
            ["Timestamp", "Comfort_Humidity_Score"]
        ].head()
    )

    data_types = [
        "Comfort_Humidity_Score",
        "Level_Alertness_Score",
        "Air_Freshness_Score",
    ]
    plot_actual_survey_vs_measured_co2(
        hourly_iaq_data["CO2"].reset_index(),
        survey_data_by_room[room_identifier].reset_index(),
        data_types,
    )
