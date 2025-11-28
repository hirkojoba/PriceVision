@echo off
REM PriceVision - Quick Start Script for Windows

echo 🚀 PriceVision - AI Stock Trend Predictor
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python found
python --version
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment exists
)

echo.

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo.
echo 📥 Installing dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt

if errorlevel 1 (
    echo ❌ Error installing dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully

echo.
echo ==========================================
echo 🎉 Setup complete!
echo ==========================================
echo.
echo Starting Streamlit application...
echo.
echo The app will open in your browser at:
echo 👉 http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

REM Run the Streamlit app
streamlit run app\streamlit_app.py

pause
