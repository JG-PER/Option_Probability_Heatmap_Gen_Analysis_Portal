#!/bin/bash

# Ensure the script runs from the directory where it is saved
cd "$(dirname "$0")"

echo "----------------------------------------------------------------"
echo "INITIATING OPTION PORTAL ENGINE (FLASK)"
echo "Target File: flask_portal_build.py"
echo "----------------------------------------------------------------"

# Check for required dependencies
# Note: flask-cors is required for the HTML file to talk to the server
if ! python3 -c "import flask, flask_cors, yfinance" &> /dev/null; then
    echo " > Missing dependencies detected."
    echo " > Installing: flask flask-cors yfinance pandas numpy scipy matplotlib seaborn"
    pip install flask flask-cors yfinance pandas numpy scipy matplotlib seaborn
fi

echo " > Engine Starting..."
echo " > Please open 'portal.html' in your web browser once the server is online."
echo "----------------------------------------------------------------"

# Run the Flask Server
python3 flask_portal_build.py