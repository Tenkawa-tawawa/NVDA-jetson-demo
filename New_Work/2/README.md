# 進階練習 (9)：即時多感測訊號的狀態相似度搜尋  
Real-time Multi-Sensor State Similarity Search on NVIDIA Jetson Orin

## 專案簡介
本專案模擬工業物聯網（IIoT）場景中的 **異常狀態檢測 / 相似度搜尋** 任務。  
目標是在 **NVIDIA Jetson Orin** 平台上，針對多通道感測器訊號（震動、聲學、電流），  
實現 **毫秒級（< 2 ms）的歷史狀態檢索**。

本系統採用 **CPU-GPU 異質協同運算（Heterogeneous Computing）** 架構：
- **GPU（cuBLAS）**：負責高吞吐量的矩陣乘法
- **CPU（OpenMP + STL）**：負責低延遲的 Top-K 排序搜尋

---

## 1. 數學原理

### 相似度計算
即時查詢訊號 `Q` 與歷史資料庫 `H` 的相似度定義為：

\[
S = Q \times H^T
\]

其中：

- \( Q \in \mathbb{R}^{B \times d} \)：Batch Query  
  - \( B = 128 \)（同時處理 128 筆查詢）
  - \( d = 64 \)（特徵維度）
- \( H \in \mathbb{R}^{M \times d} \)：History Database  
  - \( M = 4096 \)
- \( S \in \mathbb{R}^{B \times M} \)：Similarity Score Matrix

---

### Top-K 搜尋
對於相似度矩陣 `S` 的每一列（每筆 Query），找出 **數值最大的 K 筆歷史狀態索引**。

---

## 2. 系統架構與優化策略

### CPU–GPU 異質協同運算
| 模組 | 運算內容 | 技術 | 原因 |
|----|----|----|----|
| GPU | 矩陣乘法 `Q × Hᵀ` | cuBLAS | 高並行、O(N³) 密集計算 |
| CPU | Top-K 搜尋 | OpenMP + STL | 避免 GPU Kernel 啟動與排序開銷 |

### 設計理念
- 在 `M = 4096` 的規模下，**排序屬於低計算密度任務**
- GPU 排序的 Kernel Launch Overhead 反而拖慢延遲
- 將 Top-K Offload 給多核心 CPU，可獲得更低的 **End-to-End Latency**

---

## 3. 環境準備（Prerequisites）

### 硬體
- NVIDIA Jetson Orin（Nano / NX / AGX）

### 軟體
- Ubuntu (JetPack)
- CUDA Toolkit
- cuBLAS
- OpenMP（g++）

### 檢查 CUDA
```bash
nvcc --version



4. 編譯方式（Build）

本專案需連結 cuBLAS 並啟用 OpenMP。

nvcc similarity_search.cpp -o similarity_app \
-O3 \
-lcublas \
-Xcompiler -fopenmp

編譯參數說明

-O3：最高等級編譯最佳化

-lcublas：連結 NVIDIA cuBLAS 函式庫

-Xcompiler -fopenmp：啟用 CPU 多執行緒（OpenMP）


5. 執行方式（Execution）
./similarity_app

6. 預期輸出（Sample Output）
Initializing Data (M=4096, B=128, d=64)...
Starting Performance Test...
Processing Complete!

Total Execution Time: 184.44 ms
Average Latency per Query: 1.44094 ms   <-- 關鍵效能指標

[Query 0] Top-5 Results:
Rank 1: History[1236] Score = 9.28487
Rank 2: History[ 982] Score = 9.17321
...


✔ 平均延遲 < 2 ms
✔ 符合即時異常偵測與線上相似度搜尋需
