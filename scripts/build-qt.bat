@echo off
REM Build Qt executable using PyInstaller

echo Building PyMLUI Qt executable...

cd /d "%~dp0.."

pyinstaller qt.spec --clean

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build successful!
    echo Executable located at: dist\pymlui-qt\pymlui-qt.exe
) else (
    echo.
    echo Build failed with error code %ERRORLEVEL%
)

pause
