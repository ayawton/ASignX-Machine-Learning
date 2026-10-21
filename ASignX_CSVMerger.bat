@echo off
setlocal
cd /d "%~dp0"
if exist "output\csv-merger\ASignX_CSVMerger\ASignX_CSVMerger.exe" (
    start "" "output\csv-merger\ASignX_CSVMerger\ASignX_CSVMerger.exe"
) else (
    py -3.11 -m ml.csv_merger_gui
)
endlocal
