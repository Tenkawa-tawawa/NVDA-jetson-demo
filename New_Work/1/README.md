實作範例 (5): 旋轉機械振動評估 - 矩陣運算優化本練習模擬工業振動分析中的數學模型運算。目標是透過 演算法優化 (結合律) 與 硬體加速 (SIMD/Multi-threading)，在 NVIDIA Jetson Orin 平台上將 $4096 \times 4096$ 的矩陣運算延遲壓低至 10ms 以下。1. 數學原理題目要求計算公式：$$y = (A \times B) \times x + \lambda x$$其中：$A, B \in \mathbb{R}^{N \times N}$ ($N=4096$)$x, y \in \mathbb{R}^{N \times 1}$$\lambda \in \mathbb{R}$ (Scalar)優化關鍵：矩陣結合律直覺解法 (Naive)：先算 $(A \times B)$，得到一個 $4096 \times 4096$ 的暫存矩陣，運算量為 $O(N^3)$。這在嵌入式系統上會跑非常久（數秒鐘）。優化解法 (Optimized)：利用結合律改寫為 $A \times (B \times x)$。先算 $v = B \times x$ (矩陣乘向量)，運算量 $O(N^2)$。再算 $y = A \times v$ (矩陣乘向量)，運算量 $O(N^2)$。總運算量大幅下降，是本題能達成 < 10ms 的理論基礎。2. 環境準備 (Prerequisites)在開始之前，請確保您的 Jetson Orin 環境已安裝必要的數學函式庫與設定硬體效能。2.1 安裝函式庫Bashsudo apt-get update
# 注意：在 Ubuntu 上套件名稱是 libeigen3-dev (有個 3)
sudo apt-get install -y libeigen3-dev libopenblas-dev
2.2 釋放硬體效能 (極重要)Jetson 預設處於省電模式，記憶體頻寬會被鎖住。執行本練習前必須執行以下指令：Bash# 切換到最高效能模式 (MAXN)
sudo nvpmodel -m 0

# 強制鎖定 CPU 與記憶體 (EMC) 到最高頻率
sudo jetson_clocks
3. 編譯程式碼 (Build)本題比較 Eigen (C++ Template Library) 與 OpenBLAS (C Library) 的效能。為了讓 Eigen 發揮最大效能，我們需要開啟 OpenMP 並將底層運算轉包給 BLAS。請使用以下指令進行編譯：Bashg++ vibration_analysis.cpp -o vibration_app \
    -O3 \
    -march=native \
    -fopenmp \
    -DEIGEN_USE_BLAS \
    -I/usr/include/eigen3 \
    -lopenblas
參數說明：-O3 -march=native: 啟用最高級別優化，並針對 Orin (Cortex-A78AE) 自動使用 NEON 指令集。-fopenmp: 啟用多執行緒支援。-DEIGEN_USE_BLAS: 關鍵！ 強制 Eigen 使用 OpenBLAS 作為運算後端，否則 Eigen 在矩陣向量乘法 (GEMV) 上可能會堅持用單核心慢跑。4. 執行與效能測試 (Run)⚠️ 執行前注意事項 (TA Note)本題是 記憶體頻寬密集 (Memory Bound) 的應用。由於 Jetson 的 CPU 與 GPU 共用記憶體頻寬，開啟瀏覽器 (Chrome/Firefox)、VS Code 或 Telegram 會嚴重干擾測試結果。請務必關閉所有背景視窗，僅保留終端機。建議執行指令根據實測，在 $4096$ 維度下，使用 2~4 個執行緒 效果最佳。過多的執行緒反而會造成記憶體頻寬塞車 (Contention)。Bash# 設定執行緒數量 (建議 4)
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

# 執行程式
./vibration_app
5. 預期結果 (Expected Output)若環境設定正確，您應該會看到如下輸出：PlaintextInitializing data for N=4096...
Testing with 100 iterations (Average Latency).
Data ready.
------------------------------------------------
[Eigen] Average Time: 9.8523 ms
>> [Eigen] SUCCESS!
------------------------------------------------
[OpenBLAS] Average Time: 9.7104 ms
>> [OpenBLAS] SUCCESS!
常見問題排除 (Troubleshooting)問題狀況可能原因解決方法編譯錯誤 fatal error: Eigen/Dense找不到標頭檔確認安裝 libeigen3-dev，且編譯參數包含 -I/usr/include/eigen3。Eigen 跑很慢 (~30ms)Eigen 使用了預設的單核運算確認編譯時有加上 -DEIGEN_USE_BLAS。兩者都卡在 14ms 左右記憶體頻寬被背景程式吃掉關閉瀏覽器與 VS Code，確認 sudo jetson_clocks 已執行。兩者都卡在 14ms 左右 (已關程式)執行緒過多導致互搶嘗試將 OPENBLAS_NUM_THREADS 改為 2 試試。6. 進階思考為什麼矩陣乘向量 (GEMV) 的效能瓶頸通常在記憶體 (Memory Bound) 而不是算力 (Compute Bound)？試著使用 taskset 綁定核心，為什麼在 Orin 的 Cluster 架構下，綁定同一組 Cluster (如 4,5,6,7) 反而可能變慢？