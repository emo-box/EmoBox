#!/bin/bash

DIRECTORY="./EmoBench/preprocess"


if [ ! -d "$DIRECTORY" ]; then
    echo "Directory $DIRECTORY does not exist."
    exit 1
fi


for script in "$DIRECTORY"/*.py; do
    if [ "$(basename "$script")" == "extract_features.py" ]; then
        echo "Skipping $script"
        continue
    fi
    
    if [ -f "$script" ]; then
        echo "Running $script..."
        python "$script"
        if [ $? -ne 0 ]; then
            echo "Error running $script"
        fi
    fi
done

echo "All scripts executed."
