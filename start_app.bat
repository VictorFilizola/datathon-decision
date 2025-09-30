@echo off
ECHO Starting the Recruitment MVP Application...

ECHO.
ECHO Step 1: Starting FastAPI backend server in a new window...
REM The API will run on http://127.0.0.1:8000
START "Backend API" cmd /k "uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

ECHO.
ECHO Waiting for 5 seconds for the backend to initialize...
timeout /t 5 /nobreak > NUL

ECHO.
ECHO Step 2: Starting Streamlit frontend in a new window...
REM Streamlit will automatically find an open port, usually 8501
START "Frontend UI" cmd /k "streamlit run frontend/app.py"

ECHO.
ECHO Both servers should be running in separate windows now.
ECHO.
ECHO Step 3: Automatically opening the frontend in your browser...
timeout /t 3 /nobreak > NUL
start http://localhost:8501

ECHO.
ECHO If the page did not open, please manually open your browser and go to http://localhost:8501
ECHO.
pause