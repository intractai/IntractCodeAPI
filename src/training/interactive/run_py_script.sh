#!/bin/bash

# Check if the correct number of arguments are provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <conda_environment_name> <path_to_python_script>"
  exit 1
fi

# Assign the arguments to variables
CONDA_ENV=$1
PYTHON_SCRIPT=$2

# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV 2>&1 >/dev/null

# Run the provided Python script and only display its output
if hash python 2>/dev/null; then
    python "$PYTHON_SCRIPT"
elif hash python3 2>/dev/null; then
    python3 "$PYTHON_SCRIPT"
else
    echo "Neither python nor python3 could be found"
    exit 1
fi