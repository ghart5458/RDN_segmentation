@echo off
REM Windows batch script to install pytorch_wavelets from source
REM This replaces the original Setup/pytorch_wavelets.bat

echo Installing pytorch_wavelets from GitHub source...

REM Create temporary directory
set TEMP_DIR=%TEMP%\pytorch_wavelets_install
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
mkdir "%TEMP_DIR%"

REM Clone repository
echo Cloning pytorch_wavelets repository...
git clone https://github.com/fbcotter/pytorch_wavelets.git "%TEMP_DIR%"
if errorlevel 1 (
    echo Failed to clone repository
    exit /b 1
)

REM Install the package
echo Installing pytorch_wavelets...
cd "%TEMP_DIR%"
python setup.py install
if errorlevel 1 (
    echo Failed to install pytorch_wavelets
    exit /b 1
)

REM Cleanup
cd /d "%~dp0"
rmdir /s /q "%TEMP_DIR%"

echo pytorch_wavelets installed successfully!
pause