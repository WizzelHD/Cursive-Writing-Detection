@echo off
:loop
echo [AUTO-START] Starte CRNN Training...

python main.py

if %errorlevel% neq 0 (
    echo [CRASH] Absturz bemerkt. Neustart in 10 Sekunden...
    timeout /t 10
    goto loop
)

echo [FERTIG] Das Training wurde regulaer beendet.
pause