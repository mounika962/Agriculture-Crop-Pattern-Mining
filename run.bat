@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
python crop_pattern_mining.py %*
pause
