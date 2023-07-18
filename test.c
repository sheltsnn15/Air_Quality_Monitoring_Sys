#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>

#define I2C_BUS "/dev/i2c-1"
#define SCD4X_ADDRESS 0x62

// I2C Commands
#define CMD_STOP_PERIODIC_MEASUREMENT 0x3F86
#define CMD_START_PERIODIC_MEASUREMENT 0x21B1
#define CMD_READ_MEASUREMENT 0xEC05
#define CMD_READ_SERIAL_NUMBER 0x3682

// Delay Constants
#define DELAY_IDLE_STATE 1000000           // 1 second
#define DELAY_MEASUREMENT_COMPLETION 50000 // 50 milliseconds

// Open the I2C bus file for communication
int openI2C()
{
    int file = open(I2C_BUS, O_RDWR);
    if (file < 0)
    {
        perror("Failed to open I2C bus");
        exit(1);
    }
    return file;
}

// Set the I2C address for communication
void setI2CAddress(int file, int address)
{
    if (ioctl(file, I2C_SLAVE, address) < 0)
    {
        perror("Failed to set I2C address");
        exit(1);
    }
}

// Write data to the I2C device
void writeI2C(int file, unsigned char *data, int length)
{
    if (write(file, data, length) != length)
    {
        perror("Failed to write to I2C device");
        exit(1);
    }
}

// Read data from the I2C device
void readI2C(int file, unsigned char *data, int length)
{
    if (read(file, data, length) != length)
    {
        perror("Failed to read from I2C device");
        exit(1);
    }
}

// Stop periodic measurement
void stopPeriodicMeasurement(int file)
{
    unsigned char command[] = {CMD_STOP_PERIODIC_MEASUREMENT >> 8, CMD_STOP_PERIODIC_MEASUREMENT & 0xFF};
    writeI2C(file, command, sizeof(command));
}

// Start periodic measurement
void startPeriodicMeasurement(int file)
{
    unsigned char command[] = {CMD_START_PERIODIC_MEASUREMENT >> 8, CMD_START_PERIODIC_MEASUREMENT & 0xFF};
    writeI2C(file, command, sizeof(command));
}

// Read CO2, temperature, and humidity measurement
void readMeasurement(int file, float *co2, float *temperature, float *humidity)
{
    unsigned char command[] = {CMD_READ_MEASUREMENT >> 8, CMD_READ_MEASUREMENT & 0xFF};
    writeI2C(file, command, sizeof(command));
    usleep(DELAY_MEASUREMENT_COMPLETION);

    unsigned char response[6];
    readI2C(file, response, sizeof(response));

    *co2 = (response[0] << 8) | response[1];
    *temperature = (response[3] << 8) | response[4];
    *humidity = response[2];
}

int main()
{
    // Open the I2C bus
    int file = openI2C();

    // Set the I2C address for the SCD4x sensor
    setI2CAddress(file, SCD4X_ADDRESS);

    // Ensure SCD4x is in idle state
    usleep(DELAY_IDLE_STATE);

    // Make sure measurement is stopped
    stopPeriodicMeasurement(file);

    // Read the serial number
    unsigned char serialNumberCommand[] = {CMD_READ_SERIAL_NUMBER >> 8, CMD_READ_SERIAL_NUMBER & 0xFF};
    writeI2C(file, serialNumberCommand, sizeof(serialNumberCommand));

    unsigned char serialNumberResponse[9];
    readI2C(file, serialNumberResponse, sizeof(serialNumberResponse));
    printf("scd41 Serial Number: %s\n", serialNumberResponse);

    // Start periodic measurement
    startPeriodicMeasurement(file);

    // Measure every second for one minute
    for (int i = 0; i < 60; i++)
    {
        sleep(5);
        float co2, temperature, humidity;
        readMeasurement(file, &co2, &temperature, &humidity);
        printf("%f, %f, %f\n", co2, temperature, humidity);
    }

    // Close the I2C bus
    close(file);

    return 0;
}
