#!/bin/bash

# Enable MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

while true
do
echo ""
echo "[AUTO-START] Starte CRNN Training... $(date)"
echo ""

python3 main.py
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "[CRASH] Absturz bemerkt. Neustart in 10 Sekunden... $(date)"
    sleep 10
else
    echo ""
    echo "[FERTIG] Das Training wurde regulär beendet. $(date)"
    break
fi

done

read -p "Drücke ENTER zum Beenden..."
