#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Adafruit_SSD1306
from PIL import Image, ImageDraw, ImageFont

# Initialize the OLED display
disp = Adafruit_SSD1306.SSD1306_128_64(rst=None, i2c_bus=1)

# Initialize with the display size
disp.begin()

# Function to clear the display
def clear_display():
    disp.clear()
    disp.display()

# Clear the display at the beginning
clear_display()

# Create an image buffer
width = disp.width
height = disp.height
image = Image.new("1", (width, height))

# Create a drawing object
draw = ImageDraw.Draw(image)

# Write text to the display
font = ImageFont.load_default()
draw.text((0, 0), "Hello, Raspberry Pi!", font=font, fill=255)

# Display the image
disp.image(image)
disp.display()

# To clear the display later in your script, call clear_display()
clear_display()

