#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <omp.h>          // 引入 OpenMP
#include <Eigen/Dense>
#include <cblas.h>

// 定義常數
const int N = 4096;
const float LAMBDA = 0.1f;
const int ITERATIONS = 100; // 測試 100 次取平均

int main() {
    // 設定 Eigen 使用多執行緒 (與 OpenBLAS 一致)
    // 如果編譯沒加 -fopenmp，這行可能無效，但不會報錯
    Eigen::setNbThreads(4);

    std::cout << "Initializing data for N=" << N << "..." << std::endl;
    std::cout << "Testing with " << ITERATIONS << " iterations (Average Latency)." << std::endl;
    
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> x(N);
    std::vector<float> y(N);

    // 亂數產生
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    auto rand_func = [&]() { return dis(gen); };

    std::generate(A.begin(), A.end(), rand_func);
    std::generate(B.begin(), B.end(), rand_func);
    std::generate(x.begin(), x.end(), rand_func);

    std::cout << "Data ready." << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // ==========================================
    // 方法 1: 使用 Eigen
    // ==========================================
    {
        Eigen::Map<Eigen::MatrixXf> matA(A.data(), N, N);
        Eigen::Map<Eigen::MatrixXf> matB(B.data(), N, N);
        Eigen::Map<Eigen::VectorXf> vecX(x.data(), N);
        Eigen::Map<Eigen::VectorXf> vecY(y.data(), N);

        // Warmup (暖身，不計時) - 讓 CPU Cache 準備好
        vecY = matA * (matB * vecX) + LAMBDA * vecX;

        auto start = std::chrono::high_resolution_clock::now();

        for(int i=0; i<ITERATIONS; ++i) {
            // 核心運算
            vecY = matA * (matB * vecX) + LAMBDA * vecX;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        double avg_time = duration.count() / ITERATIONS;

        std::cout << "[Eigen] Average Time: " << avg_time << " ms" << std::endl;
        
        if(avg_time < 10.0) {
            std::cout << ">> [Eigen] SUCCESS!" << std::endl;
        } else {
            std::cout << ">> [Eigen] WARNING: Still too slow (" << avg_time << "ms). Did you add -fopenmp?" << std::endl;
        }
    }

    std::cout << "------------------------------------------------" << std::endl;

    // ==========================================
    // 方法 2: 使用 OpenBLAS
    // ==========================================
    {
        std::vector<float> tmp(N, 0.0f);
        
        // Warmup
        cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, 1.0f, B.data(), N, x.data(), 1, 0.0f, tmp.data(), 1);
        cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, 1.0f, A.data(), N, tmp.data(), 1, 0.0f, y.data(), 1);
        cblas_saxpy(N, LAMBDA, x.data(), 1, y.data(), 1);

        auto start = std::chrono::high_resolution_clock::now();

        for(int i=0; i<ITERATIONS; ++i) {
            // Step 1: tmp = B * x
            cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, 
                        1.0f, B.data(), N, x.data(), 1, 0.0f, tmp.data(), 1);

            // Step 2: y = A * tmp
            cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, 
                        1.0f, A.data(), N, tmp.data(), 1, 0.0f, y.data(), 1);

            // Step 3: y = y + lambda * x
            cblas_saxpy(N, LAMBDA, x.data(), 1, y.data(), 1);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        double avg_time = duration.count() / ITERATIONS;

        std::cout << "[OpenBLAS] Average Time: " << avg_time << " ms" << std::endl;
         
        if(avg_time < 10.0) {
            std::cout << ">> [OpenBLAS] SUCCESS!" << std::endl;
        }
    }

    return 0;
}