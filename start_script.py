# start_script.py

import time
import subprocess

def is_container_running(container_name):
    try:
        result = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container_name], capture_output=True, text=True)
        return result.stdout.strip() == "true"
    except Exception as e:
        print(f"Error checking container status: {e}")
        return False

if __name__ == "__main__":
    container_name = "influxdb"

    while True:
        if is_container_running(container_name):
            print("Starting your Python script...")
            # Replace with the actual command to start your Python script
            subprocess.run(["python3", "/home/shelt/Documents/Code/Air_Quality_Monitoring_Sys/iaq_sensing.py"])
            break  # Exit loop once the script is run
        else:
            print(f"Waiting for the '{container_name}' container to start...")
            time.sleep(10)  # Adjust the polling interval as needed
