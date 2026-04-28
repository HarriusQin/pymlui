@echo off
REM Build Tkinter executable using PyInstaller

echo Building PyMLUI Tkinter executable...

cd /d "%~dp0.."

pyinstaller mlui.spec --clean

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build successful!
    echo Executable located at: dist\pymlui\pymlui.exe
) else (
    echo.
    echo Build failed with error code %ERRORLEVEL%
)

pause
