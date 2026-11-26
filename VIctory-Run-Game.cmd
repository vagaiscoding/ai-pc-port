@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "VICTORY_LAUNCH_EXE=%~1"
set "VICTORY_LAUNCH_TITLE=%~2"
set "VICTORY_LAUNCH_CONSOLE=%~3"
set "VICTORY_LAUNCH_ROOT=%~dp0"

powershell.exe -NoProfile -NonInteractive -Command ^
    "$lockPath = Join-Path $env:VICTORY_LAUNCH_ROOT 'victory-game.lock';" ^
    "$busy = 'VIctory is already running. Close its game window before starting another session.';" ^
    "try {" ^
    "    $playLock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None');" ^
    "} catch {" ^
    "    $errorCode = $_.Exception.GetBaseException().HResult -band 0xffff;" ^
    "    if ($errorCode -in 32, 33) { [Console]::Error.WriteLine($busy); exit 73 };" ^
    "    [Console]::Error.WriteLine($_.Exception.Message); exit 1;" ^
    "};" ^
    "try {" ^
    "    $installDir = $env:VICTORY_LAUNCH_ROOT.TrimEnd([IO.Path]::DirectorySeparatorChar);" ^
    "    $running = Get-Process -Name 'victory-rtx-*' -ErrorAction SilentlyContinue | Where-Object { $_.Path -and [IO.Path]::GetDirectoryName($_.Path) -eq $installDir } | Select-Object -First 1;" ^
    "    if ($running) { [Console]::Error.WriteLine($busy); exit 73 };" ^
    "    $runArgs = @('run', $env:VICTORY_LAUNCH_TITLE, '--execute');" ^
    "    if ($env:VICTORY_LAUNCH_CONSOLE) { $runArgs += @('--console', $env:VICTORY_LAUNCH_CONSOLE) };" ^
    "    $ErrorActionPreference = 'Stop';" ^
    "    & $env:VICTORY_LAUNCH_EXE @runArgs; exit $LASTEXITCODE;" ^
    "} finally { $playLock.Dispose() }"
exit /b %ERRORLEVEL%
