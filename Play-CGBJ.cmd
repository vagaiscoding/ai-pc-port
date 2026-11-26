@echo off
setlocal EnableExtensions DisableDelayedExpansion

if not defined VICTORY_GAME_DIR (echo Set VICTORY_GAME_DIR to your game folder. & exit /b 2)
if not defined VICTORY_CACHE_DIR set "VICTORY_CACHE_DIR=%~dp0local\cgbj\cache"
if not defined VICTORY_DATA_DIR set "VICTORY_DATA_DIR=%~dp0local\cgbj\data"
if not defined VICTORY_WINMM_PAD set "VICTORY_WINMM_PAD=0"

pushd "%~dp0runtime\cgbj" || exit /b 1
call Play-VIctory.cmd pareto-residency
set "result=%ERRORLEVEL%"
popd
exit /b %result%
