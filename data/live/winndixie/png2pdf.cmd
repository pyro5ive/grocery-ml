@echo off
setlocal enabledelayedexpansion

REM === Check if user provided a directory ===
if "%~1"=="" (
    echo Usage: %~nx0 "C:\path\to\inputpng"
    echo Example: %~nx0 "C:\scans"
    exit /b 1
)
if "%~2"=="" (
    echo Usage: %~nx0 "C:\path\to\outputPdf"
    echo Example: %~nx0 "C:\scans"
    exit /b 1
)


REM === Input directory from command-line argument ===
set "INPUT_DIR=%~1"

REM === Output directory for iles ===
set "OUTPUT_DIR=%~2"

REM === Make sure output directory exists ===
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM === Loop through all PNGs in the directory ===
for %%F in ("%INPUT_DIR%\*.png") do (
    echo Processing %%~nxF ...
    set "BASENAME=%%~nF"

    REM Run Tesseract with config file
    tesseract "%%F" "%OUTPUT_DIR%\!BASENAME!" -l eng --dpi 400 --psm 6 -c preserve_interword_spaces=1 pdf
)
