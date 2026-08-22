@echo off
setlocal
set PYTHON_EXE=E:\Anaconda\envs\edge-ecg\python.exe
set RUNNER=src\22_experiment2_6r_optimize.py
set LOG_DIR=results\experiment2_6r\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

call :run R1 2 || exit /b 1
call :run R1 3 || exit /b 1
call :run R1 4 || exit /b 1
call :run R1 5 || exit /b 1
call :run R2 1 || exit /b 1
call :run R2 2 || exit /b 1
call :run R2 3 || exit /b 1
call :run R2 4 || exit /b 1
call :run R2 5 || exit /b 1
call :run R3 1 || exit /b 1
call :run R3 2 || exit /b 1
call :run R3 3 || exit /b 1
call :run R3 4 || exit /b 1
call :run R3 5 || exit /b 1
exit /b 0

:run
%PYTHON_EXE% %RUNNER% --candidate %1 --outer-fold %2 --seed 20260803 >> "%LOG_DIR%\%1_fold%2_seed20260803.out.log" 2>> "%LOG_DIR%\%1_fold%2_seed20260803.err.log"
exit /b %errorlevel%
