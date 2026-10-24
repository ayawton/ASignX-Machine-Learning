@echo off
setlocal
cd /d "%~dp0"
if exist "output\extractor-reviewed\ASignX_DataExtractor\ASignX_DataExtractor.exe" (
    start "" "output\extractor-reviewed\ASignX_DataExtractor\ASignX_DataExtractor.exe" %*
) else if exist ".venv-py314\Scripts\python.exe" (
    ".venv-py314\Scripts\python.exe" -m ml.extractor_gui %*
) else (
    py -3.14 -m ml.extractor_gui %*
)
endlocal

