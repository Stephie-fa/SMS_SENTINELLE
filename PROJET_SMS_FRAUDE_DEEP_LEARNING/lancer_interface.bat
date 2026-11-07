@echo off
chcp 65001 > nul
cd /d "%~dp0"
python interface_graphique.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo L'interface n'a pas pu demarrer. Verifiez Python, PyTorch et les fichiers du modele.
    pause
)
