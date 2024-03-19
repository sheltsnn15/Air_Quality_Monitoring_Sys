# start_script.py

import time
import subprocess
import logging
logging.basicConfig(filename='/home/thepinalyser/.log/start_script.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def is_container_running(container_name):
    try:
        result = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container_name], capture_output=True, text=True)
        process = subprocess.run(["/home/thepinalyser/.pyenv/versions/3.9.2/bin/python", "/home/thepinalyser/Documents/Air_Quality_Monitoring_Sys/iaq_sensing.py"])
        if process.returncode != 0:
            logging.error(f"iaq_sensing.py exited with return code {process.returncode}")
        return result.stdout.strip() == "true"
    except Exception as e:
        logging.error(f"Error checking container status: {e}")
        return False

if __name__ == "__main__":
    container_name = "influx-influxdb-1"

    while True:
        if is_container_running(container_name):
            logging.info("Starting your Python script...")
            # Replace with the actual command to start your Python script
            subprocess.run(["python3", "/home/thepinalyser/Documents/Air_Quality_Monitoring_Sys/iaq_sensing.py"])
            break  # Exit loop once the script is run
        else:
            logging.info(f"Waiting for the '{container_name}' container to start...")
            time.sleep(10)  # Adjust the polling interval as needed
