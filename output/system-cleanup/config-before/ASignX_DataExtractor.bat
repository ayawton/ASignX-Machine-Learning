@echo off
setlocal
cd /d "%~dp0"
if exist "output\extractor-reviewed\ASignX_DataExtractor\ASignX_DataExtractor.exe" (
    start "" "output\extractor-reviewed\ASignX_DataExtractor\ASignX_DataExtractor.exe" %*
) else if exist "output\extractor-large-datasets\ASignX_DataExtractor\ASignX_DataExtractor.exe" (
    start "" "output\extractor-large-datasets\ASignX_DataExtractor\ASignX_DataExtractor.exe" %*
) else if exist "output\extractor\ASignX_DataExtractor\ASignX_DataExtractor.exe" (
    start "" "output\extractor\ASignX_DataExtractor\ASignX_DataExtractor.exe" %*
) else (
    py -3.11 -m ml.extractor_gui %*
)
endlocal

