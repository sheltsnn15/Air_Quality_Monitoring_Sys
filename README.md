# Air_Quality_Monitoring_Sys

This repository contains code for an Air Quality Monitoring System designed for different environments using Raspberry Pi devices.

## Branches

- **main**: Contains general information about the system setup, including usage instructions, sensor functionalities, systemctl services, and Docker setup.
- **classroompi**: Branch for deploying the system in a classroom environment.
- **lectureroompi**: Branch for deploying the system in a lecture room environment.

## Usage

To switch between different environments, follow these steps:

1. **Clone the Repository**: Clone this repository to your Raspberry Pi device.

   ```bash
   git clone <repository-url>
   ```

2. **Switch Branches**: Use the following commands to switch between different branches based on your environment.

   - For Classroom Environment:
     ```bash
     git checkout classroompi
     ```

   - For Lecture Room Environment:
     ```bash
     git checkout lectureroompi
     ```

3. **Follow Environment-Specific Instructions**: After switching to the desired branch, follow the instructions provided in the branch's README file for setting up the system in that specific environment.

## Sensor Guide

For detailed information on how to use the sensors and their functionalities, refer to the `sensorguide.md` file in the main branch.

## System Setup

The main branch includes information on:

- How to use the sensors and their functionalities.
- Systemctl services for managing the monitoring system.
- Start scripts for initializing the monitoring system.
- Docker setup with InfluxDB for data storage.

