#!/bin/bash

# Script to open HTML files in Chrome browser
if [ -z "$1" ]; then
    echo "Usage: ./open-browser.sh <html-file>"
    exit 1
fi

# Try different Chrome commands
if command -v google-chrome &> /dev/null; then
    google-chrome "$1" &
elif command -v chromium-browser &> /dev/null; then
    chromium-browser "$1" &
elif command -v chromium &> /dev/null; then
    chromium "$1" &
else
    echo "Chrome/Chromium not found. Please install Google Chrome."
    echo "On Ubuntu/Debian: sudo apt install google-chrome-stable"
    exit 1
fi