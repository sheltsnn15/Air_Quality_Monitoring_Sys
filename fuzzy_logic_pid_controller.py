import numpy as np
from simple_pid import PID
import paho.mqtt.client as mqtt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from flc import setup_fuzzy_logic_system, get_fan_speed_setpoint
from pid_control import setup_pid_controller, get_adjusted_fan_speed

MQTT_BROKER = "localhost"  # Or your broker's IP address if different
MQTT_PORT = 1883  # Default MQTT port
MQTT_TOPIC = "zigbee2mqtt/silvercrest_zigbee_plug/set"  # Adjust based on your setup


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))


def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload))


def set_fan_speed(speed):
    # Create an MQTT client and attach our routines to it.
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    # Construct the message payload
    message_payload = (
        '{"fan_mode": "' + speed + '"}'
    )  # Adjust payload based on your device's requirements

    # Publishing the message
    client.publish(MQTT_TOPIC, message_payload)
    client.disconnect()


def setup_fuzzy_logic_system():
    # Define fuzzy variables and membership functions
    co2 = ctrl.Antecedent(np.arange(400, 5001, 1), "CO2")
    fan_speed = ctrl.Consequent(np.arange(0, 101, 1), "fan_speed")

    # Membership functions
    co2["normal"] = fuzz.trimf(co2.universe, [400, 400, 1000])
    co2["elevated"] = fuzz.trimf(co2.universe, [800, 1500, 2200])
    co2["high"] = fuzz.trimf(co2.universe, [1800, 3000, 5000])

    fan_speed["low"] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
    fan_speed["medium"] = fuzz.trimf(fan_speed.universe, [30, 50, 70])
    fan_speed["high"] = fuzz.trimf(fan_speed.universe, [60, 100, 100])

    # Fuzzy rules
    rule1 = ctrl.Rule(co2["normal"], fan_speed["low"])
    rule2 = ctrl.Rule(co2["elevated"], fan_speed["medium"])
    rule3 = ctrl.Rule(co2["high"], fan_speed["high"])

    fan_speed_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    return ctrl.ControlSystemSimulation(fan_speed_ctrl)


def get_fan_speed_setpoint(fan_speed_simulation, co2_level):
    fan_speed_simulation.input["CO2"] = co2_level
    fan_speed_simulation.compute()
    return fan_speed_simulation.output["fan_speed"]


def setup_pid_controller(setpoint):
    # Initialize and return a PID controller
    # Tune these PID values according to your system
    return PID(1.0, 0.1, 0.05, setpoint=setpoint)


def get_adjusted_fan_speed(pid, current_fan_speed):
    # Calculate and return the new fan speed
    return pid(current_fan_speed)


def main():
    # Example CO2 level
    co2_level = 1200
    current_fan_speed = 40  # Current fan speed in percentage

    # Setup FLC and PID
    fan_speed_simulation = setup_fuzzy_logic_system()
    fan_speed_setpoint = get_fan_speed_setpoint(fan_speed_simulation, co2_level)
    pid = setup_pid_controller(fan_speed_setpoint)

    # Adjust fan speed based on PID output
    new_fan_speed = get_adjusted_fan_speed(pid, current_fan_speed)
    print(f"New Fan Speed: {new_fan_speed}%")
    fan_mode = "medium"  # Placeholder, determine based on new_fan_speed
    set_fan_speed(fan_mode)

    print(f"Fan speed set to: {fan_mode}")


if __name__ == "__main__":
    main()
