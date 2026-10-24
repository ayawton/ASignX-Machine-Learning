@echo off
setlocal
cd /d "%~dp0"
if exist "output\csv-merger\ASignX_CSVMerger\ASignX_CSVMerger.exe" (
    start "" "output\csv-merger\ASignX_CSVMerger\ASignX_CSVMerger.exe" %*
) else if exist ".venv-py314\Scripts\python.exe" (
    ".venv-py314\Scripts\python.exe" -m ml.csv_merger_gui %*
) else (
    py -3.14 -m ml.csv_merger_gui %*
)
endlocal
