@echo off
setlocal
title Visualizer server

rem  Double-click this instead of opening index.html directly.
rem
rem  Serving the folder over http is what makes the midi/aligned, energy/ and
rem  presets/ dropdowns work (they read the server's directory listing), and
rem  http://localhost:8000/ is also the URL an OBS Browser source needs.
rem
rem  THIS WINDOW IS THE SERVER. Leave it open while you work; closing it stops
rem  the server. Click the file again any time -- if it is already running it
rem  just reopens the browser instead of failing on a busy port.

set "PORT=8000"
set "URL=http://localhost:%PORT%/"

rem  Re-entry point: a second copy of this script is launched with --open, and
rem  waits for the port to actually answer before opening the browser. Without
rem  it the browser races the server and lands on "can't connect".
if "%~1"=="--open" goto waitopen

cd /d "%~dp0"

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (
  where py >nul 2>nul && set "PY=py"
)
if not defined PY (
  echo.
  echo   Python was not found on your PATH.
  echo   Install it from python.org and tick "Add python.exe to PATH".
  echo.
  pause
  exit /b 1
)

netstat -an | findstr "LISTENING" | findstr ":%PORT% " >nul
if not errorlevel 1 (
  echo.
  echo   Already serving on port %PORT% - opening the browser.
  echo   The server is the other window; close that one to stop it.
  echo.
  start "" "%URL%"
  rem  ping as a sleep: timeout.exe aborts outright if stdin isn't a console
  ping -n 4 127.0.0.1 >nul
  exit /b 0
)

start "" /b cmd /c ""%~f0" --open"

echo.
echo   Serving  %CD%
echo   Open at  %URL%
echo.
echo   Keep this window open. Closing it stops the server.
echo.

rem  Bound to localhost on purpose: nothing outside this machine needs to
rem  reach it, and it keeps Windows from asking about the firewall.
%PY% -m http.server %PORT% --bind 127.0.0.1

echo.
echo   Server stopped. If that was not on purpose, the reason is above --
echo   usually another program is already using port %PORT%.
echo.
pause
exit /b 0

:waitopen
set /a tries=0
:waitloop
set /a tries+=1
if %tries% gtr 40 exit /b 1
netstat -an | findstr "LISTENING" | findstr ":%PORT% " >nul
if errorlevel 1 (
  ping -n 2 127.0.0.1 >nul
  goto waitloop
)
start "" "%URL%"
exit /b 0
