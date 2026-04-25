#!/bin/bash

# needs `pip install nightskycam-images`
VENV="/path/to/nightskycam-venv"

# to monitor what is going on
LOG_FILE="/data1/nightskycam/backup-log"

# clear sky data destination
NIGHTSKYCAM_DIR="/data1/nightskycam/data"

# bad weather data destination
BAD_WEATHER_DIR="/data1/nightskycam/bad_weather"

# for saving of the server config/source code
WWW_DESTINATION="/data1/nightskycam/server/www"

# server IP, where the original data is
NIGHTSKYCAM_IP=142.144.47.184

# path to extra backup
NIGHTSKYCAM_NETSHARE="/is/projects/allsky/images"

# for classifying between clear sky and bad weather
CLOUDY_MODEL="/path/to/models/cloudy.pt"
RAINY_MODEL="/path/to/models/rainy.pt"
CLOUDY_THRESHOLD=0.94
RAINY_THRESHOLD=0.7

# duration after which original data is deleted
# from the server
RETENTION_MONTHS=1
TODAY=$(date +%s)
