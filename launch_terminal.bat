@echo off
setlocal
REM 1) go to the folder where this .bat lives
cd /d "%~dp0"

REM 2) activate conda env (edit path/env name if different)
call "C:\Miniconda3\condabin\conda.bat" activate cdf

REM 3) handy aliases for the session
doskey python3=python $*
doskey pip3=pip $*

REM 4) stay in an interactive shell
cmd /K