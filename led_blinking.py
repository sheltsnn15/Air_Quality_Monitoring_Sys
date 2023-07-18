import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
red_pin = 17
green_pin = 27
blue_pin = 22

GPIO.setup(red_pin, GPIO.OUT)
GPIO.setup(green_pin, GPIO.OUT)
GPIO.setup(blue_pin, GPIO.OUT)


def set_rgb_led(red, green, blue):
    GPIO.output(red_pin, red)
    GPIO.output(green_pin, green)
    GPIO.output(blue_pin, blue)


set_rgb_led(1, 0, 0)  # Turn on red
time.sleep(2)         # Wait for 2 seconds
set_rgb_led(0, 1, 0)  # Turn on green
time.sleep(2)         # Wait for 2 seconds
set_rgb_led(0, 0, 1)  # Turn on blue
time.sleep(2)         # Wait for 2 seconds
GPIO.cleanup()        # Clean up the GPIO pins
