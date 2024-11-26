# Installation Guide: Emotion Detection App

## Overview
This is a Emotion Detection application created to detect one of your five emotions inlcuding Happy, Sad, Shock, Angry, and Neutral.

Follow the instructions below to clone the respository, set up your environment, and run the app.
-------------------------------
## Prerequisites

- **Python 3.10** (Ensure Python 3.10 is installed on your system). 
- **PIP** (Python package manager, which comes with Python).

## Steps to Run the Project

### 1. Install Python 3.10
In the files section below, [download Python 3.10.0](https://www.python.org/downloads/release/python-3100/)

### 2. Add Python 3.10 to your PATH Environment Variables

During installation, **make sure to select the option to add Python to your PATH**.

- Find the folder where python 3.10 is installed

1. For Mac
> which python3
2. For Windows
> where python

Add Python 3.10 to PATH [by following this tutorial](https://www.youtube.com/watch?v=3a3UOSBffUI)

To verify if Python 3.10 is installed, open your terminal/command prompt and type:
> python --version

### 3. Install Requirements
Paste the following in your terminal:

> cd EmotionDetectionApp

1. For MacOS
> python3 -m pip install -r requirements.txt
2. For Windows
> python -m pip install -r requirements.txt

### 4. Run the Application
Use the command in your terminal
> python app.py

### 5. Using the App

- Click the "Start Camera" button to open the webcam
- Click the "Stop Camera" button to close the webam
- Review the emotion count on the tbale on the right
- Refresh the webpage to reset the count to 0