# Jetson Orin 程式實作專案 - (1)

本練習聚焦四項操作：環境準備（檢查並安裝 gcc/g++、Eigen、OpenBLAS）、程式編譯與執行（在本目錄下以 g++ 編譯並連結上述庫）、效能監測（使用 tegrastats 觀察 CPU/GPU/記憶體），以及延伸練習題。
---

## 1. 環境準備

先更新套件清單：
```bash
sudo apt update
```
再裝基本工具：
```bash
sudo apt install -y build-essential libeigen3-dev libopenblas-dev
```
確認一下版本：
```bash
gcc --version
g++ --version
```
看一下 Eigen 路徑：
```
ls /usr/include/eigen3
```
檢查 OpenBLAS：
```
ls /usr/lib/aarch64-linux-gnu | grep openblas
```
最後跑一下 tegrastats：
```
sudo tegrastats
```
會看到 CPU/GPU/記憶體的即時狀況，按 Ctrl+C 結束。

2. 編譯程式
3. 效能監測與執行

延伸練習題
