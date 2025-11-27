#!/bin/bash

# Ensure the script runs from the directory where it is saved
cd "$(dirname "$0")"

echo "----------------------------------------------------------------"
echo "INITIATING OPTION ANALYSIS DASHBOARD (STREAMLIT)"
echo "Target File: streamlit_dashboard_build.py"
echo "----------------------------------------------------------------"

# Check if virtual environment is preferred (optional, commented out)
# source venv/bin/activate

# Check for required dependencies
if ! python3 -c "import streamlit, yfinance, seaborn" &> /dev/null; then
    echo " > Missing dependencies detected."
    echo " > Installing: streamlit yfinance pandas numpy scipy matplotlib seaborn"
    pip install streamlit yfinance pandas numpy scipy matplotlib seaborn
fi

# Run the application
# We use 'python -m streamlit' to ensure the correct python path is used
python3 -m streamlit run streamlit_dashboard_build.py

echo "----------------------------------------------------------------"
echo "SESSION ENDED"
echo "----------------------------------------------------------------"