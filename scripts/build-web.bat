@echo off
REM Build script for Flask WebUI

echo Starting Flask WebUI server...

cd /d "%~dp0.."

echo.
echo To run the Flask WebUI, use one of these commands:
echo   flask --app gui.flask_app run
echo   python -m gui.flask_app
echo   make run-flask
echo.
echo Or install dependencies with:
echo   uv pip install -e ".[web-flask]"
echo.
echo For Streamlit WebUI, use:
echo   streamlit run gui/streamlit_app.py
echo   uv pip install -e ".[web-streamlit]"
echo.
pause
