@echo off
rem ============================================================
rem  Telltale: double-click to start everything on the ASUS and
rem  open the control panel in your browser. On the page you can
rem  run live or on a recording, record data, and train.
rem  Close this window (or press Ctrl+C in it) to stop everything,
rem  including the motor.
rem ============================================================

rem ---- edit this line once ----
set ASUS=asus@YOUR-ASUS-ADDRESS
set REPO=~/Downloads/Telltale/src

title Telltale
echo Starting Telltale on %ASUS% ...
rem Open the control panel a few seconds after the connection starts.
start "" cmd /c "timeout /t 6 /nobreak >nul && start http://localhost:8080"
ssh -t -L 8080:localhost:8080 %ASUS% "cd %REPO% && source .venv/bin/activate && python telltale.py"
echo.
echo Telltale has stopped. Press any key to close.
pause >nul
