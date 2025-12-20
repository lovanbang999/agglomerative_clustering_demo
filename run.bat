@echo off
REM Agglomerative Clustering Demo - Run Script for Windows

echo ==================================
echo Agglomerative Clustering Demo
echo ==================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Tao virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Kich hoat virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo 📥 Cai dat dependencies...
pip install -q -r requirements.txt

REM Run app
echo.
echo 🚀 Khoi chay ung dung...
echo ➡️  Mo browser tai: http://localhost:8501
echo.

streamlit run app.py
