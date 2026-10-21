@echo off
cd /d "%~dp0"
if exist "dist\ASignX_DynamicStudio.exe" (
    start "" "dist\ASignX_DynamicStudio.exe"
) else if exist ".venv-py314\Scripts\python.exe" (
    ".venv-py314\Scripts\python.exe" -m ml.dynamic_studio_gui
    if errorlevel 1 pause
) else (
    python -m ml.dynamic_studio_gui
    if errorlevel 1 pause
)
