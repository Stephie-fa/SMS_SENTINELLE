@echo off
chcp 65001 > nul
echo =====================================================================
echo    PROJET 2 : DETECTION DE SMS FRAUDULEUX PAR DEEP LEARNING
echo =====================================================================
echo.

echo [1/3] Verification des tests d'integrite des donnees...
python tests_donnees.py
if %ERRORLEVEL% NEQ 0 (
    echo Erreur lors des tests.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Demonstration de detection sur 3 SMS types...
echo.
echo --- Test 1 : Arnaque aux gains ---
python tester_mon_sms.py "Vous avez gagne 1 000 000 FCFA ! Cliquez sur bit.ly/gain"
echo.
echo --- Test 2 : Arnaque bancaire / code OTP ---
python tester_mon_sms.py "Cher client, compte bloque. Envoyez le code 584930 au 07123456 pour debloquer"
echo.
echo --- Test 3 : Message legitime usuel ---
python tester_mon_sms.py "Salut, on mange ensemble ce midi a la cafeteria ?"

echo.
echo =====================================================================
echo    EXECUTION TERMINEE AVEC SUCCES !
echo =====================================================================
pause
