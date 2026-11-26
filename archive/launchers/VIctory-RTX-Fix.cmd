@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0" || exit /b 1

set "execute=--execute"
if /i "%~1"=="--check" set "execute="
if not "%~1"=="" if /i not "%~1"=="--check" (
    echo Usage: %~nx0 [--check]
    popd
    exit /b 2
)

set "exe=%CD%\victory-rtx-world-fix.exe"
if not defined VICTORY_GAME_DIR (echo Set VICTORY_GAME_DIR to your game folder. & popd & exit /b 2)
set "title=%VICTORY_GAME_DIR%\game_scarlett_master.exe"
if not exist "%exe%" (
    echo Missing executable at "%exe%".
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
set "log_dir=%VICTORY_LOG_ROOT%\%stamp%-rtx"
mkdir "%log_dir%" || (popd & exit /b 1)

set "NO_COLOR=1"
set "VICTORY_LOG=info"
if not defined VICTORY_SMALL_SHADOW_WINDOWS set "VICTORY_SMALL_SHADOW_WINDOWS=1"
set "VICTORY_D3D12_DEBUG=0"
set "VICTORY_D3D12_GBV=0"
set "VICTORY_GPU_TIMING=0"
set "VICTORY_PIPELINE_LOG="
set "VICTORY_CRASH_LOG=%log_dir%\crashes.log"
echo VIctory is writing logs to "%log_dir%".
"%exe%" run "%title%" %execute% > "%log_dir%\console.log" 2>&1
set "result=%ERRORLEVEL%"
echo VIctory exited with code %result%. Logs are in "%log_dir%".
popd
exit /b %result%
