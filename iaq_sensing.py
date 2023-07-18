import time
from sensirion_i2c_driver import LinuxI2cTransceiver, I2cConnection
from sensirion_i2c_scd import Scd4xI2cDevice


# Connect to the I²C 1 port /dev/i2c-1
with LinuxI2cTransceiver('/dev/i2c-1') as i2c_transceiver:
    # Create SCD4x device
    scd4x = Scd4xI2cDevice(I2cConnection(i2c_transceiver))

    # ensure SCD4x is in idle state
    time.sleep(1)

    # Make sure measurement is stopped, else we can't read serial number or
    # start a new measurement
    scd4x.stop_periodic_measurement()

    print("scd41 Serial Number: {}".format(scd4x.read_serial_number()))

    scd4x.start_periodic_measurement()

    # Measure every second for one minute
    for _ in range(60):
        time.sleep(5)
        co2, temperature, humidity = scd4x.read_measurement()
        # use default formatting for printing output:
        print("{}, {}, {}".format(co2, temperature, humidity))
