@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "execute=--execute"
set "VICTORY_D3D12_GBV=0"
set "VICTORY_GPU_TIMING=0"
if "%~1"=="" goto setup
if /i "%~1"=="--check" (
    set "execute="
    goto setup
)
if /i "%~1"=="--deep" (
    set "VICTORY_D3D12_GBV=1"
    set "VICTORY_GPU_TIMING=1"
    goto setup
)
echo Usage: %~nx0 [--deep ^| --check]
exit /b 2

:setup
pushd "%~dp0" || exit /b 1
set "exe=%CD%\victory-rtx-world-fix.exe"
if not defined VICTORY_GAME_DIR (echo Set VICTORY_GAME_DIR to your game folder. & popd & exit /b 2)
set "title=%VICTORY_GAME_DIR%\game_scarlett_master.exe"
if not exist "%exe%" (
    echo Missing diagnostics build at "%exe%".
    popd
    exit /b 1
)
if not exist "%title%" (
    echo Missing game executable at "%title%".
    popd
    exit /b 1
)

set "stamp="
for /f %%I in ('powershell.exe -NoProfile -NonInteractive -Command "Get-Date -Format yyyyMMdd-HHmmss-fff"') do set "stamp=%%I"
if not defined stamp (
    echo Could not create a timestamp for the log folder.
    popd
    exit /b 1
)
if not defined VICTORY_LOG_ROOT set "VICTORY_LOG_ROOT=%CD%\VIctory-Logs"
set "log_dir=%VICTORY_LOG_ROOT%\%stamp%"
mkdir "%log_dir%" || (popd & exit /b 1)

set "NO_COLOR=1"
set "RUST_BACKTRACE=full"
set "VICTORY_LOG=trace"
if not defined VICTORY_SMALL_SHADOW_WINDOWS set "VICTORY_SMALL_SHADOW_WINDOWS=1"
set "VICTORY_D3D12_DEBUG=1"
set "VICTORY_D3D12_DRED=1"
set "VICTORY_CRASH_BREAKPOINTS=1"
set "VICTORY_CRASH_LOG=%log_dir%\crashes.log"
set "VICTORY_PIPELINE_LOG=%log_dir%\pipelines.log"

echo VIctory is writing logs to "%log_dir%".
> "%log_dir%\run.txt" echo Started at %DATE% %TIME%
>> "%log_dir%\run.txt" echo Executable: "%exe%"
>> "%log_dir%\run.txt" echo Game: "%title%"
>> "%log_dir%\run.txt" echo Options: trace, D3D12 debug layer, DRED, first-chance breakpoints
if /i "%~1"=="--check" >> "%log_dir%\run.txt" echo Mode: loader check only
if /i "%~1"=="--deep" >> "%log_dir%\run.txt" echo Mode: GPU validation and per-command timings enabled
if exist "%SystemRoot%\System32\nvidia-smi.exe" "%SystemRoot%\System32\nvidia-smi.exe" --query-gpu=name,driver_version,memory.total --format=csv,noheader >> "%log_dir%\run.txt" 2>&1

"%exe%" config > "%log_dir%\console.log" 2>&1
if errorlevel 1 goto finished
"%exe%" run "%title%" %execute% >> "%log_dir%\console.log" 2>&1

:finished
set "result=%ERRORLEVEL%"
>> "%log_dir%\run.txt" echo Finished at %DATE% %TIME% with exit code %result%
echo VIctory exited with code %result%. Logs are in "%log_dir%".
popd
exit /b %result%
