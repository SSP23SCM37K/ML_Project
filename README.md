# ML_Project
# Driver Drowsiness Detection

Driver Drowsiness Detection is a project aimed at improving road safety by detecting driver drowsiness using computer vision and machine learning techniques. The system analyzes video input from a camera to monitor the driver's eyes and determine if they are open or closed. If the driver's eyes remain closed for an extended period of time, an alarm is triggered to alert the driver and prompt them to take necessary action.

## Team members:

## Aasritha Juvva (A20523774) ajuvva@hawk.iit.edu

## Sree Rama Murthy Kandala (A20505937) skandala3@hawk.iit.edu

## Pavan kumar Turpati (A20510960) pturpati@hawk.iit.edu

Kaagle dataset used: https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new?resource=download

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Usage](#usage)
5. [Results](#results)

## Introduction

Driver drowsiness is a significant cause of accidents on the road. This project aims to address this issue by utilizing computer vision techniques to detect signs of drowsiness in drivers. By monitoring the driver's eye movements and analyzing their state, the system can accurately determine if the driver is becoming drowsy or falling asleep.

## Features

- Real-time monitoring of driver's eye state
- Detection of closed eyes for extended periods
- Triggering of an alarm to alert the driver
- Integration with existing vehicle systems

## Requirements

To run this project, you will need the following:

- Python 3.x
- OpenCV
- NumPy

## Usage
Connect a camera to your computer or device.

Run the application
Copy code
python drowsiness_detection.py
The application will start monitoring the driver's eye state in real-time.

If the driver's eyes remain closed for an extended period, an alarm will be triggered.

## Results
The model used in this project achieved an accuracy of 96% in detecting the driver's eye state. Extensive testing and evaluation have been conducted to ensure reliable performance in various conditions and lighting environments. The system has shown promising results in detecting drowsiness and alerting drivers to prevent potential accidents.
