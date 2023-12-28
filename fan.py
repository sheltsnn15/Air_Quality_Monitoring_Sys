import tkinter as tk
from time import sleep
import threading
import math

# Function to update the fan display in the Tkinter window
def update_fan_display(canvas, angle):
    canvas.delete("fan")
    for i in range(4):
        x0 = 50 + 30 * math.cos(math.radians(angle + 90 * i))
        y0 = 50 + 30 * math.sin(math.radians(angle + 90 * i))
        x1 = 50 + 30 * math.cos(math.radians(angle + 180 + 90 * i))
        y1 = 50 + 30 * math.sin(math.radians(angle + 180 + 90 * i))
        canvas.create_line(x0, y0, x1, y1, fill="black", tags="fan")
    root.update()

# Function to simulate rotating a fan
def rotate_fan():
    angle = 0
    while True:
        update_fan_display(canvas, angle)
        angle = (angle + 10) % 360
        sleep(0.1)

# Set up the Tkinter window
root = tk.Tk()
root.title("Fan Rotation Simulator")
canvas = tk.Canvas(root, height=100, width=100)
canvas.pack()

# Run the fan rotation function in a separate thread
thread = threading.Thread(target=rotate_fan)
thread.daemon = True  # Daemonize thread
thread.start()

# Start the Tkinter event loop
root.mainloop()

