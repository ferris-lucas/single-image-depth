#!/bin/bash

# Create a data directory if it doesn't exist
mkdir -p data

# Define the URL for the data file
URL="http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

# Define the output file name
OUTPUT_FILE="data/nyu_depth_v2_labeled.mat"

# Download the data file directly to the data directory
wget -P data "$URL"

# Check if download was successful
if [ $? -eq 0 ]; then
  echo "Download successful."
else
  echo "Download failed. Please check the URL and try again."
fi
