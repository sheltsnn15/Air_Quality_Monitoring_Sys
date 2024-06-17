import sched
import time
import pandas as pd
from data_handling import load_measured_iaq_data, preprocess_data, resample_data
import paho.mqtt.client as mqtt

scheduler = sched.scheduler(time.time, time.sleep)


def write_log(message):
    with open("system_log.txt", "a") as log_file:
        log_file.write(f"{message}\n")


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully")
    else:
        print(f"Failed to connect, return code {rc}")


def on_disconnect(client, userdata, rc):
    print("Disconnected")


def set_fan_state(client, state):
    message_payload = '{"state": "' + state + '"}'
    client.publish("zigbee2mqtt/silvercrest_zigbee_plug/set", message_payload)
    write_log(f"Fan state set to {state}, MQTT Payload: {message_payload}")


def check_and_update(client, scheduler, data_files, thresholds, interval):
    write_log("Running scheduled check of IAQ data...")
    for key, file_path in data_files.items():
        write_log(f"Loading and processing {key} data...")
        df = load_measured_iaq_data(file_path)
        df = preprocess_data(df, method="ffill")
        df = resample_data(df, resample_frequency="5min")
        print(df.head())
        current_value = df.iloc[-1]
        print(current_value)

        write_log(f"Latest {key} value: {current_value}")
        if current_value > thresholds[key]:
            set_fan_state(client, "ON")
            break
    else:
        set_fan_state(client, "OFF")

    write_log(f"Scheduling next check in {interval} seconds...")
    scheduler.enter(
        interval,
        1,
        check_and_update,
        (client, scheduler, data_files, thresholds, interval),
    )


def load_and_prepare_data(data_files):
    prepared_data = {}
    for key, file_path in data_files.items():
        print(f"Loading and processing {key} data...")
        data = load_measured_iaq_data(file_path)
        data = preprocess_data(data)
        data = resample_data(data, resample_frequency="5min")
        prepared_data[key] = data
    return prepared_data


def check_and_update(client, scheduler, prepared_data, thresholds, interval):
    print("Checking IAQ data against thresholds...")
    for key, df in prepared_data.items():
        current_value = df.iloc[-1]
        print(f"Latest {key} value: {current_value}")

        if current_value > thresholds[key]:
            set_fan_state(client, "ON")
            break
    else:
        set_fan_state(client, "OFF")

    print(f"Scheduling next check in {interval} seconds...")
    scheduler.enter(
        interval,
        1,
        check_and_update,
        (client, scheduler, prepared_data, thresholds, interval),
    )


def main():
    client = mqtt.Client()
    client.connect("localhost", 1883, 60)
    client.loop_start()

    data_files = {
        "CO2": "../data/raw_data/victors_office/2024-03-06-11-22_co2_data.csv",
        "temperature": "../data/raw_data/victors_office/2024-03-06-11-28_temperature_data.csv",
        "humidity": "../data/raw_data/victors_office/2024-03-06-11-26_humidity_data.csv",
    }

    thresholds = {"CO2": 1000, "humidity": 60, "temperature": 25}
    prepared_data = load_and_prepare_data(data_files)

    interval = 300  # Check every 5 minutes
    scheduler.enter(
        0, 1, check_and_update, (client, scheduler, prepared_data, thresholds, interval)
    )
    scheduler.run()


if __name__ == "__main__":
    main()
