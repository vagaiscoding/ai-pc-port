@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0" || exit /b 1

set "execute=--execute"
set "mode=dlss-45"
set "NO_COLOR=1"
if not defined VICTORY_CACHE_DIR set "VICTORY_CACHE_DIR=%CD%\local\victory-rtx\cache"
if not defined VICTORY_DATA_DIR set "VICTORY_DATA_DIR=%CD%\local\victory-rtx\data"
set "VICTORY_LOG=info"
set "VICTORY_D3D12_DEBUG=0"
set "VICTORY_D3D12_DRED=0"
set "VICTORY_D3D12_GBV=0"
set "VICTORY_GPU_TIMING=0"
set "VICTORY_GPU_BREADCRUMBS=0"
set "VICTORY_CRASH_BREAKPOINTS=0"
set "VICTORY_PIPELINE_LOG="
if not defined VICTORY_DLSS_DIR set "VICTORY_DLSS_DIR=%CD%\Streamline-2.14.16\bin\x64"
set "VICTORY_DLSS_PRESET_M=1"
if not defined VICTORY_SMALL_SHADOW_WINDOWS set "VICTORY_SMALL_SHADOW_WINDOWS=1"
if not defined VICTORY_DISABLE_BVH set "VICTORY_DISABLE_BVH=1"

if "%~1"=="" goto launch
if not "%~2"=="" goto usage
if /i "%~1"=="--check" (
    set "execute="
    set "mode=dlss-45-check"
    goto launch
)
if /i "%~1"=="--gpu-log" (
    set "VICTORY_GPU_BREADCRUMBS=1"
    set "mode=dlss-45-gpu-log"
    goto launch
)
if /i "%~1"=="--full-log" (
    set "VICTORY_LOG=trace"
    set "VICTORY_D3D12_DEBUG=1"
    set "VICTORY_D3D12_DRED=1"
    set "VICTORY_GPU_BREADCRUMBS=1"
    set "VICTORY_CRASH_BREAKPOINTS=1"
    set "mode=dlss-45-full-log"
    goto launch
)

:usage
echo Usage: %~nx0 [--check ^| --gpu-log ^| --full-log]
popd
exit /b 2

:launch
if not defined VICTORY_RTX_EXE set "VICTORY_RTX_EXE=%CD%\runtime\victory-rtx\target\diag\victory.exe"
set "exe=%VICTORY_RTX_EXE%"
if not defined VICTORY_GAME_DIR (echo Set VICTORY_GAME_DIR to your game folder. & popd & exit /b 2)
set "title=%VICTORY_GAME_DIR%\game_scarlett_master.exe"
for %%F in ("%exe%" "%title%" "%VICTORY_DLSS_DIR%\sl.interposer.dll" "%VICTORY_DLSS_DIR%\sl.common.dll" "%VICTORY_DLSS_DIR%\sl.dlss.dll" "%VICTORY_DLSS_DIR%\nvngx_dlss.dll") do (
    if not exist "%%~F" (
        echo Missing required file "%%~F".
        popd
        exit /b 1
    )
)

set "stamp="
for /f %%I in ('powershell.exe -NoProfile -NonInteractive -Command "Get-Date -Format yyyyMMdd-HHmmss-fff"') do set "stamp=%%I"
if not defined stamp (
    echo Could not create a timestamp for the log folder.
    popd
    exit /b 1
)
if not defined VICTORY_LOG_ROOT set "VICTORY_LOG_ROOT=%CD%\VIctory-Logs"
set "log_dir=%VICTORY_LOG_ROOT%\%stamp%-%mode%"
mkdir "%log_dir%" || (popd & exit /b 1)
set "VICTORY_CRASH_LOG=%log_dir%\crashes.log"
if "%mode%"=="dlss-45-full-log" set "VICTORY_PIPELINE_LOG=%log_dir%\pipelines.log"

> "%log_dir%\run.txt" echo Started at %DATE% %TIME%
>> "%log_dir%\run.txt" echo Executable: "%exe%"
>> "%log_dir%\run.txt" echo Mode: %mode%
>> "%log_dir%\run.txt" echo Console: Series S
>> "%log_dir%\run.txt" echo DLSS preset: M
>> "%log_dir%\run.txt" echo Streamline runtime: "%VICTORY_DLSS_DIR%"
>> "%log_dir%\run.txt" echo Disable ray-tracing BVH: %VICTORY_DISABLE_BVH%
echo VIctory is writing logs to "%log_dir%".
if defined execute (
    call "%~dp0VIctory-Run-Game.cmd" "%exe%" "%title%" series-s > "%log_dir%\console.log" 2>&1
) else (
    "%exe%" run "%title%" --console series-s > "%log_dir%\console.log" 2>&1
)
set "result=%ERRORLEVEL%"
if "%result%"=="73" echo VIctory is already running. Close its game window before starting another session.
>> "%log_dir%\run.txt" echo Finished at %DATE% %TIME% with exit code %result%
echo VIctory exited with code %result%. Logs are in "%log_dir%".
popd
exit /b %result%
