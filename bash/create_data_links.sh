#!/bin/bash

# Check if minimum required arguments provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <folder_path> <start_date> [<end_date>]"
    echo "Dates must be in YYYY_MM_DD format"
    echo "If end_date is omitted, includes all dates from start_date onwards"
    exit 1
fi

# Input validation
folder_path="$1"
start_date="$2"
end_date="$3"

# Validate date formats
validate_date() {
    local date_str="$1"
    if ! [[ "$date_str" =~ ^[0-9]{4}_[0-9]{2}_[0-9]{2}$ ]]; then
        return 1
    fi
    
    local year="${date_str:0:4}"
    local month="${date_str:5:2}"
    local day="${date_str:8:2}"
    
    if ! date -d "${year}-${month}-${day}" >/dev/null 2>&1; then
        return 1
    fi
    return 0
}

# Validate all inputs
if [ ! -d "$folder_path" ]; then
    echo "Error: Source folder '$folder_path' does not exist or is not accessible"
    exit 1
fi

if ! validate_date "$start_date"; then
    echo "Error: Invalid start date format. Use YYYY_MM_DD"
    exit 1
fi

# Convert start date to seconds since epoch
start_epoch=$(date -d "${start_date//_/}" +%s)

# If end date is provided, validate and convert it
if [ -n "$end_date" ]; then
    if ! validate_date "$end_date"; then
        echo "Error: Invalid end date format. Use YYYY_MM_DD"
        exit 1
    fi
    end_epoch=$(date -d "${end_date//_/}" +%s)
else
    # No end date means use distant future
    end_epoch=$((2**63-1))
fi

# Process each directory
for dir_name in "$folder_path"/*; do
    # Skip non-directories
    if [ ! -d "$dir_name" ]; then
        continue
    fi
    
    # Get base name of directory
    base_name="$(basename "$dir_name")"
    
    # Skip if not in YYYY_MM_DD format
    if ! validate_date "$base_name"; then
        continue
    fi
    
    # Convert directory name to epoch time
    dir_epoch=$(date -d "${base_name//_/}" +%s)
    
    # Create symlink if date is within range
    if [ $dir_epoch -ge $start_epoch ] && [ $dir_epoch -le $end_epoch ]; then
        if ln -s "$dir_name" "./$base_name"; then
            echo "Created symbolic link: $base_name -> $dir_name"
        else
            echo "Failed to create symbolic link for $base_name"
        fi
    fi
done