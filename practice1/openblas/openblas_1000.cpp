#include <cblas.h>
#include <vector>
#include <chrono>
#include <iostream>

int main() {
    const int N = 1000;
    std::vector<double> A(N*N), B(N*N), C(N*N, 0.0);

    // TODO: 初始化 A, B 的內容
    // TODO: 使用 cblas_dgemm 做矩陣乘法
    // TODO: 計時並輸出執行時間
    // TODO: 簡單檢查結果 (例如輸出 C[0])

    return 0;
}
