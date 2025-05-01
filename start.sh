#!/bin/bash

# Disable Streamlit watchdog warnings (correct for latest versions)
export STREAMLIT_DISABLE_WATCHDOG_WARNINGS=true

# Set Python path to access backend modules
export PYTHONPATH=backend

# Run the app
exec streamlit run frontend/GUI.py