@echo off
rem ============================================================
rem  Telltale: double-click to start the whole system on the ASUS
rem  and open the dashboard in your browser.
rem  Close this window (or press Ctrl+C in it) to stop everything,
rem  including the motor.
rem ============================================================

rem ---- edit these two lines once ----
set ASUS=asus@YOUR-ASUS-ADDRESS
set REPO=~/Downloads/Telltale/src

rem ---- what to run ----
set MODE=--replay data/loose_fastener_214501.csv

rem Replay mode: runs on a saved recording, no ESP32 needed.
rem Change the file name above to replay a different recording.

title Telltale
echo Starting Telltale on %ASUS% ...
rem Open the dashboard a few seconds after the connection starts.
start "" cmd /c "timeout /t 6 /nobreak >nul && start http://localhost:8080"
ssh -t -L 8080:localhost:8080 %ASUS% "cd %REPO% && source .venv/bin/activate && python telltale.py %MODE%"
echo.
echo Telltale has stopped. Press any key to close.
pause >nul
