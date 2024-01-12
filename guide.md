## SCD4x Sensor Commands

The SCD4x sensor communicates with other devices, like microcontrollers, using specific commands sent over the I2C communication protocol. Here are the key commands:

### 1. Start Periodic Measurement (`start_periodic_measurement`)

- **Purpose**: Initiates periodic measurements, which means the sensor will regularly provide data on CO2 concentration, temperature, and humidity.
- **Command**: `0x21b1`
- **Usage**: You use this command to start the sensor's measurement cycle.

### 2. Read Measurement (`read_measurement`)

- **Purpose**: Reads the actual sensor data, including CO2 concentration, temperature, and humidity. You can read this data only once per measurement cycle.
- **Command**: `0xec05`
- **Usage**: After starting a measurement, you use this command to obtain the measurement results.

### 3. Stop Periodic Measurement (`stop_periodic_measurement`)

- **Purpose**: Stops the periodic measurement. Useful for conserving power or changing sensor settings.
- **Command**: `0x3f86`
- **Usage**: You send this command when you want to pause the regular measurements temporarily.

## Advanced Features

### 1. Persist Settings (`persist_settings`)

- **Purpose**: Stores configuration settings like temperature offset or sensor altitude in the sensor's memory, making them stay even after you turn off the sensor.
- **Command**: `0x3615`
- **Usage**: Use this when you want to save specific settings for future use.

### 2. Get Serial Number (`get_serial_number`)

- **Purpose**: Retrieves a unique serial number from the sensor. This helps identify the sensor and verify its presence.
- **Command**: `0x3682`
- **Usage**: You can use this command to get a unique ID for each sensor.

### 3. Perform Self-Test (`perform_self_test`)

- **Purpose**: Runs a self-test to check if the sensor functions correctly and if it's receiving power correctly.
- **Command**: `0x3639`
- **Usage**: Useful for verifying that the sensor is working as expected.

### 4. Perform Factory Reset (`perform_factory_reset`)

- **Purpose**: Resets all sensor settings to their default values and erases calibration history.
- **Command**: `0x3632`
- **Usage**: You'd use this command if you want to start fresh with the sensor's default settings.

### 5. Reinit (`reinit`)

- **Purpose**: Reinitializes the sensor, reloading user settings from memory. It can help if you're encountering issues.
- **Command**: `0x3646`
- **Usage**: Use this when you need to refresh the sensor's settings.

## CRC Checksum Calculation

The CRC checksum is a way to ensure that data sent to or from the sensor isn't corrupted during transmission. Here's how it works:

- **Name**: CRC-8
- **Width**: 8 bits
- **Polynomial**: `0x31` (x^8 + x^5 + x^4 + 1)
- **Initialization**: `0xFF`
- **Reflect input**: False
- **Reflect output**: False
- **Final XOR**: `0x00`

In simple terms, the CRC checksum is a value computed from the data you send or receive. You can use a provided code snippet to calculate this checksum. If the checksum matches on both ends (sender and receiver), it indicates that the data is intact and hasn't been altered during transmission.

The example code given calculates this checksum for you.

This information should help you understand how to communicate with the SCD4x sensor and verify the accuracy of the data you receive using CRC checksums.
