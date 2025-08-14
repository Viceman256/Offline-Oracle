@echo off
title Offline Oracle Server
echo =================================
echo  Offline Oracle - RAG Server
echo =================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting the RAG server...
echo Check 'config.ini' to see which LLM provider is active.
echo.
python server.py

echo.
echo =================================
echo  Server has been stopped.
echo =================================
pause