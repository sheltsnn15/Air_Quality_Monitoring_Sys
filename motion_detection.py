#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Create I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Create an ADS1115 ADC object
ads = ADS.ADS1115(i2c)

# Define the analog input channel (0 to 3, depending on your wiring)
channel = AnalogIn(ads, ADS.P0)

# Main loop
while True:
    # Read the analog value
    analog_value = channel.value
    # Convert the analog value to voltage (optional)
    voltage = (analog_value / 32767.0) * 4.096  # 4.096V is the default range for ADS1115
    # Print the results
    print(f"Analog Value: {analog_value}, Voltage: {voltage} V")
    time.sleep(1)  # Delay for 1 second between readings

