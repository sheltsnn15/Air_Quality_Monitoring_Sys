import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
motion_pin = 17
GPIO.setup(motion_pin, GPIO.IN)

# Callback function for motion detection


def motion_detected(channel):
    print("Motion detected!")


# Add event listener for rising edge (motion detected)
GPIO.add_event_detect(motion_pin, GPIO.RISING, callback=motion_detected)

try:
    while True:
        time.sleep(1)  # Keep the script running
except KeyboardInterrupt:
    GPIO.cleanup()
