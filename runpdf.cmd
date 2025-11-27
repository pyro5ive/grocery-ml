@echo off
setlocal enabledelayedexpansion

REM === Check if user provided a directory ===
if "%~1"=="" (
    echo Usage: %~nx0 "C:\path\to\pdfs"
    echo Example: %~nx0 "C:\scans"
    exit /b 1
)

REM === Input directory from command-line argument ===
set "INPUT_DIR=%~1"

REM === Output directory for text files ===
set "OUTPUT_DIR=%INPUT_DIR%\text"

REM === Make sure output directory exists ===
if not exist "%OUTPUT_DIR%"  mkdir "%OUTPUT_DIR%"

REM === Loop through all PDFs in the directory ===
for %%F in ("%INPUT_DIR%\*.pdf") do (
    echo Processing %%~nxF ...
    set "BASENAME=%%~nF"


    pdftotext -layout -enc UTF-8 -r 400 "%%F" "%OUTPUT_DIR%\!BASENAME!.txt"
	REM pdftotext -r 400 -enc UTF-8 "%%F" "%OUTPUT_DIR%\!BASENAME!.txt"
)
