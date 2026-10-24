@echo off
cd /d "%~dp0"
if exist ".venv-py314\Scripts\python.exe" (
    ".venv-py314\Scripts\python.exe" -m ml.manual_training_gui
) else (
    py -3.14 -m ml.manual_training_gui
)
if errorlevel 1 pause
