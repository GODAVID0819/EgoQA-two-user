@echo off
where py >nul 2>&1
if %errorlevel%==0 (py "%~dp0serve_rlhf_labeling.py") else (python "%~dp0serve_rlhf_labeling.py")
pause
