import smbus2
import time


bus = smbus2.SMBus(1)  # Use 0 for older Raspberry Pi boards

# Start a periodic measurement (0x21b1)
bus.write_byte(0x62, 0x21)
time.sleep(1)  # Wait for the measurement to complete (adjust as needed)

# Read measurement data (0xec05)
bus.write_byte(0x62, 0xec)
data = bus.read_i2c_block_data(0x62, 0x05, 6)  # Read 6 bytes of data
co2_concentration = (data[0] << 8) | data[1]
temperature_celsius = -45 + (175 * ((data[2] << 8) | data[3]) / 65536)
relative_humidity = 100 * ((data[4] << 8) | data[5]) / 65536

print(f"CO2 Concentration: {co2_concentration} ppm")
print(f"Temperature: {temperature_celsius} °C")
print(f"Relative Humidity: {relative_humidity} %")
