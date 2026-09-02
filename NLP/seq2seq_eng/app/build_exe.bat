@echo off
chcp 65001 >nul
rem 切到项目根目录（app 的上一级）
cd /d "%~dp0.."

rem 用 en2fr_app 环境（CPU 版 torch + PyInstaller）打包成单文件 exe
rem --windowed 表示不带控制台窗口；--add-data 把数据文件和模型权重一起打进 exe（Windows 下用分号分隔 源;目标）
"D:\Anaconda3\envs\en2fr_app\python.exe" -m PyInstaller --onefile --windowed --name en2fr ^
    --add-data "eng-fra-v2.txt;." ^
    --add-data "model;model" ^
    app\app.py

echo.
echo 打包完成，exe 在 dist 目录下：dist\en2fr.exe
pause
