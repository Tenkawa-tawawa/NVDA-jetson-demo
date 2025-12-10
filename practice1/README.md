# Jetson Orin 程式實作專案 - (1)

在這個學習資源中，使用者將學到如何使用 `gcc/g++` 進行程式編譯，並與 `Eigen` 與 `OpenBLAS` 函式庫結合以完成矩陣與數值運算；同時透過 `tegrastats` 工具觀察 CPU、GPU 與記憶體的效能表現。

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
