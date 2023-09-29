import time
import logging
import requests
from sensirion_i2c_driver import LinuxI2cTransceiver, I2cConnection
from sensirion_i2c_scd import Scd4xI2cDevice

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEVICE_ALTITUDE = 132
API_KEY = 'eb1be9209cd5488680621512232008'
LOCATION = 'Cork, Ireland'


# Define a function to fetch atmospheric pressure using the weather API
def get_atmospheric_pressure(api_key, location):
    url = f'https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}'

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        atmospheric_pressure = data['current']['pressure_mb']
        return atmospheric_pressure
    else:
        logger.error(f'Error: {response.status_code}')
        return None


def main():
    try:
        with LinuxI2cTransceiver('/dev/i2c-1') as i2c_transceiver:
            scd4x = Scd4xI2cDevice(I2cConnection(i2c_transceiver))

            time.sleep(1)  # Ensure sensor is in idle state

            scd4x.stop_periodic_measurement()  # Stop any ongoing measurement

            logger.info("SCD41 Serial Number: {}".format(
                scd4x.read_serial_number()))

            scd4x.set_sensor_altitude(132)

            est_ambient_pressure = int(
                get_atmospheric_pressure(API_KEY, LOCATION))
            if est_ambient_pressure is not None:
                logger.info("Estimated Atmospheric Pressure in {} mb: {}".format(
                    LOCATION, est_ambient_pressure))

                # Set the estimated ambient pressure
                scd4x.set_ambient_pressure(
                    est_ambient_pressure)  # Convert to Pa

            scd4x.start_periodic_measurement()

            for _ in range(60):
                time.sleep(5)
                co2, temperature, humidity = scd4x.read_measurement()
                logger.info("CO2: {}, Temperature: {}, Humidity: {}".format(
                    co2, temperature, humidity))
    except Exception as e:
        logger.error("An error occured: {}".format(str(e)))


if __name__ == "__main__":
    main()
