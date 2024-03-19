import time, logging, requests, os, pdb
from sensirion_i2c_driver import LinuxI2cTransceiver, I2cConnection
from sensirion_i2c_scd import Scd4xI2cDevice
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS


# Configure Logging

# Configure Logging to log to a file
logging.basicConfig(
    level=logging.INFO,
    filename="/home/thepinalyser/.log/iaq_monitoring_script.log",  # Log to this file
    filemode="a",  # Append to the file, do not overwrite
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE_ALTITUDE = 132
API_KEY = "eb1be9209cd5488680621512232008"
LOCATION = "Cork, Ireland"

# InfluxDB configurations
INFLUXDB_TOKEN = "iazhNS3IVmiHZljvizszVb38to1FTtK1Vvkv-hBTR_pBQWgfkN76SZmWjbs68SGCrpIBKySCFzupT1vLrUxF6g=="
INFLUXDB_ORG = "Shelton"
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_BUCKET = "IAQ_Monitoring_Sys"


class InfluxDBHandler:
    def __init__(self):
        self.client = InfluxDBClient(
            url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG
        )

    def write_to_influxdb(self, co2, temperature, humidity):
        write_api = self.client.write_api(write_options=SYNCHRONOUS)
        # pdb.set_trace()  # Set a breakpoint here
        point = (
            Point("sensor_data")
            .tag("location", LOCATION)
            .field("co2", co2.co2)
            .field("temperature", temperature.degrees_celsius)
            .field("humidity", humidity.percent_rh)
        )
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)


class SensorHandler:
    def __init__(self, i2c_transceiver):
        self.scd4x = Scd4xI2cDevice(I2cConnection(i2c_transceiver))

    def set_sensor_settings(self):
        self.scd4x.stop_periodic_measurement()
        self.scd4x.set_sensor_altitude(DEVICE_ALTITUDE)

    def start_periodic_measurement(self):
        self.scd4x.start_periodic_measurement()

    def read_measurement(self):
        return self.scd4x.read_measurement()


class WeatherAPIHandler:
    @staticmethod
    def fetch_atmospheric_pressure(api_key, location):
        url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
        response = requests.get(url)
        response_data = response.json() if response.status_code == 200 else None
        # pdb.set_trace()  # Set a breakpoint here
        return response_data.get("current", {}).get("pressure_mb", None)


class MainApplication:
    def __init__(self):
        self.influxdb_handler = InfluxDBHandler()

    def run(self):
        try:
            with LinuxI2cTransceiver("/dev/i2c-1") as i2c_transceiver:
                sensor_handler = SensorHandler(i2c_transceiver)
                # weather_handler = WeatherAPIHandler()

                time.sleep(1)  # Ensure sensor is in idle state

                sensor_handler.set_sensor_settings()

                # est_ambient_pressure = weather_handler.fetch_atmospheric_pressure(API_KEY, LOCATION)
                # if est_ambient_pressure is not None:
                # logger.info(f"Estimated Atmospheric Pressure in {LOCATION} mb: {est_ambient_pressure}")
                # sensor_handler.scd4x.set_ambient_pressure(int(est_ambient_pressure))  # Convert to Pa

                sensor_handler.start_periodic_measurement()

                while True:
                    time.sleep(60)  # Sleep for 60 seconds (1 minute)

                    co2, temperature, humidity = sensor_handler.read_measurement()
                    # logger.info(f"CO2: {co2}, Temperature: {temperature}, Humidity: {humidity}")

                    # Write data to InfluxDB
                    self.influxdb_handler.write_to_influxdb(co2, temperature, humidity)

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    app = MainApplication()
    app.run()

