#!/bin/bash

source "$(dirname "$0")/nightskycam_config.sh"
source "${VENV}/bin/activate"

get_nightskycam_size() {
    du -sh "$1" | cut -f1
}

echo "Backup started at $(date)" >> "$LOG_FILE"
echo "Initial size of $NIGHTSKYCAM_DIR: $(get_nightskycam_size "$NIGHTSKYCAM_DIR")" >> "$LOG_FILE"

# Sync www
rsync -avzP --links root@"${NIGHTSKYCAM_IP}":/www/ "${WWW_DESTINATION}"

# Sync images directly to NIGHTSKYCAM_DIR
rsync -avzP --links root@"${NIGHTSKYCAM_IP}":/data/nightskycam/* "${NIGHTSKYCAM_DIR}"

# Classify new images and move bad-weather ones to BAD_WEATHER_DIR
ns.backup.route "$NIGHTSKYCAM_DIR" \
    --bad-root "$BAD_WEATHER_DIR" \
    --model "cloudy:${CLOUDY_MODEL}:${CLOUDY_THRESHOLD}" \
    --model "rainy:${RAINY_MODEL}:${RAINY_THRESHOLD}"

# Sync good-weather images to network share (bad weather already moved out)
rsync -avzP --links "$NIGHTSKYCAM_DIR"/* "${NIGHTSKYCAM_NETSHARE}"

echo "Backup completed at $(date)" >> "$LOG_FILE"
echo "Final size of $NIGHTSKYCAM_DIR: $(get_nightskycam_size "$NIGHTSKYCAM_DIR")" >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"
