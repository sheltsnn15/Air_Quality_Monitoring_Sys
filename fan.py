import RPi.GPIO as GPIO
import time

# Set up GPIO
GPIO.setmode(GPIO.BCM)
fan_gpio_pin = 18  
GPIO.setup(fan_gpio_pin, GPIO.OUT)

# Turn off the fan initially
GPIO.output(fan_gpio_pin, GPIO.LOW)


def turn_fan_on():
    GPIO.output(fan_gpio_pin, GPIO.HIGH)


def turn_fan_off():
    GPIO.output(fan_gpio_pin, GPIO.LOW)


try:
    while True:
        turn_fan_on()
        time.sleep(5)  # Keep the fan on for 5 seconds
        turn_fan_off()
        time.sleep(5)  # Keep the fan off for 5 seconds

except KeyboardInterrupt:
    # Clean up when the program is terminated
    GPIO.cleanup()
