@echo off
chcp 936 >nul 2>&1
title PA Agent - AI K线分析助手
cd /d "%~dp0"

REM === TradingView 代理设置（如需代理取消下方注释） ===
REM set HTTP_PROXY=http://127.0.0.1:7897
REM set HTTPS_PROXY=http://127.0.0.1:7897
REM set NO_PROXY=xiaomimimo.com

python run.py
if errorlevel 1 (
    echo.
    echo [错误] 程序异常退出，请查看上方错误信息。
    pause
)