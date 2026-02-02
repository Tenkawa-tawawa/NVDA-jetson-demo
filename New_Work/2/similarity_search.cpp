#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h> // CPU 平行化排序用

// 定義題目參數
const int M = 4096; // 歷史資料筆數 (History)
const int B = 128;  // 即時資料批次 (Batch Query)
const int d = 64;   // 特徵維度 (Dimensions)
const int K = 5;    // Top-K

// 錯誤檢查巨集
#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        return -1; \
    } \
}
#define CHECK_CUBLAS(func) { \
    cublasStatus_t status = (func); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error Code: " << status << std::endl; \
        return -1; \
    } \
}

int main() {
    // 1. 資料準備 (Host)
    std::cout << "Initializing Data (M=" << M << ", B=" << B << ", d=" << d << ")..." << std::endl;
    
    std::vector<float> H(M * d); // 歷史矩陣 (4096 x 64)
    std::vector<float> Q(B * d); // 查詢矩陣 (128 x 64)
    std::vector<float> S(B * M); // 相似度結果矩陣 (128 x 4096)

    // 亂數生成
    std::mt19937 gen(1234); // 固定種子以重現結果
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    auto rand_func = [&]() { return dis(gen); };
    
    std::generate(H.begin(), H.end(), rand_func);
    std::generate(Q.begin(), Q.end(), rand_func);

    // 2. GPU 記憶體配置
    float *d_H, *d_Q, *d_S;
    CHECK_CUDA(cudaMalloc(&d_H, M * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q, B * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_S, B * M * sizeof(float)));

    // 3. 資料搬移 (H2D)
    CHECK_CUDA(cudaMemcpy(d_H, H.data(), M * d * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Q, Q.data(), B * d * sizeof(float), cudaMemcpyHostToDevice));

    // 建立 cuBLAS Handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::cout << "Starting Performance Test..." << std::endl;
    auto start_total = std::chrono::high_resolution_clock::now();

    // =====================================================
    // Step 1: 矩陣乘法 (Matrix Multiplication) - cuBLAS
    // =====================================================
    // 目標: S = Q * H^T
    // Q: (128 x 64), H: (4096 x 64), S: (128 x 4096)
    // 
    // ⚠️ cuBLAS 是 Column-Major (行優先)。
    // 在 C++ (Row-Major) 看來，我們想要 S (128列 x 4096行)。
    // 對 cuBLAS 來說，這等於要算 S^T (4096列 x 128行)。
    // 公式變換: S^T = (Q * H^T)^T = H * Q^T
    //
    // cuBLAS 參數映射:
    // C = alpha * op(A) * op(B) + beta * C
    // 我們要算: S_transposed = H * Q_transposed
    // Matrix A = H (在記憶體中是 4096x64 RowMajor -> 視為 64x4096 ColMajor) -> 需要轉置 -> OP_T
    // Matrix B = Q (在記憶體中是 128x64 RowMajor -> 視為 64x128 ColMajor) -> 不需要轉置 -> OP_N
    // m = 4096 (A的列數), n = 128 (B的行數), k = 64 (共用維度)
    
    float alpha = 1.0f;
    float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N, // A轉置(H), B不轉(Q)
        M, B, d,                  // m=4096, n=128, k=64
        &alpha,
        d_H, d,                   // LDA = 64 (因為原始記憶體stride是64)
        d_Q, d,                   // LDB = 64
        &beta,
        d_S, M                    // LDC = 4096 (結果矩陣的 leading dimension)
    ));

    // 等待 GPU 計算完成
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4. 資料搬回 (D2H)
    // 這裡拿回來的 d_S 在 C++ Row-Major 視角下，剛好就是 128x4096 的連續記憶體
    // 每 4096 個 float 代表一個 Query 對所有 History 的相似度
    CHECK_CUDA(cudaMemcpy(S.data(), d_S, B * M * sizeof(float), cudaMemcpyDeviceToHost));

    // =====================================================
    // Step 2: Top-K 搜尋 (CPU Optimization)
    // =====================================================
    // 使用 OpenMP 平行化處理 128 個 Queries
    
    std::vector<std::vector<int>> results(B, std::vector<int>(K));

    #pragma omp parallel for
    for (int i = 0; i < B; ++i) {
        // 取得第 i 筆 Query 的相似度列首位址
        float* row_ptr = &S[i * M];

        // 建立索引列表 [0, 1, 2, ..., 4095]
        std::vector<int> indices(M);
        for(int j=0; j<M; ++j) indices[j] = j;

        // 關鍵優化：使用 std::partial_sort 取代 std::sort
        // 複雜度從 O(M log M) 降為 O(M log K) 或 O(M)
        std::partial_sort(
            indices.begin(), 
            indices.begin() + K, 
            indices.end(),
            [&](int a, int b) { return row_ptr[a] > row_ptr[b]; } // 比較分數大小
        );

        // 存下前 K 個索引
        for(int k=0; k<K; ++k) {
            results[i][k] = indices[k];
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_total - start_total;

    std::cout << "Processing Complete!" << std::endl;
    std::cout << "Total Execution Time: " << duration.count() << " ms" << std::endl;
    std::cout << "Average Latency per Query: " << duration.count() / B << " ms" << std::endl;

    // 驗證輸出 (印出第一筆 Query 的 Top-5)
    std::cout << "\n[Query 0] Top-5 Results:" << std::endl;
    for(int k=0; k<K; ++k) {
        int idx = results[0][k];
        std::cout << "Rank " << k+1 << ": History[" << idx << "] Score = " << S[idx] << std::endl;
    }

    // 釋放資源
    cudaFree(d_H);
    cudaFree(d_Q);
    cudaFree(d_S);
    cublasDestroy(handle);

    return 0;
}