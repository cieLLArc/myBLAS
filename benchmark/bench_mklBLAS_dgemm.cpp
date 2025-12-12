#include "utils.h"
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>
#include <mkl_cblas.h> // 引入 MKL CBLAS 接口

// 声明 MKL 函数（实际不需要 extern，因为已包含头文件）
// void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
//                  const int M, const int N, const int K,
//                  const double alpha, const double *A, const int lda,
//                  const double *B, const int ldb,
//                  const double beta, double *C, const int ldc);

void warmup_run(double *A, double *B, double *C, int N)
{
    for (int i = 0; i < 10; ++i)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N,
                    1.0, A, N, B, N,
                    0.0, C, N);
    }
}

std::vector<uint64_t> benchmark_run(double *A, double *B, double *C, int N, int trials = 1000)
{
    std::vector<uint64_t> times_ns;
    times_ns.reserve(trials);
    for (int i = 0; i < trials; ++i)
    {
        uint64_t start = get_timestamp_ns();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N,
                    1.0, A, N, B, N,
                    0.0, C, N);
        uint64_t end = get_timestamp_ns();
        uint64_t elapsed = end - start;
        times_ns.push_back(elapsed);
    }
    return times_ns;
}

int main()
{
    calibrate_tsc();

    std::vector<int> sizes = {128, 512, 1024, 2048, 4096};

    for (int N : sizes)
    {
        std::cout << "\nTesting matrix size: " << N << "x" << N << std::endl;

        // 分配内存（对齐 32-byte，适合 AVX/YMM）
        double *A = (double *)aligned_alloc(32, sizeof(double) * N * N);
        double *B = (double *)aligned_alloc(32, sizeof(double) * N * N);
        double *C = (double *)aligned_alloc(32, sizeof(double) * N * N);

        if (!A || !B || !C)
        {
            std::cerr << "Memory allocation failed for size " << N << std::endl;
            continue;
        }

        // 初始化矩阵
        generate_random_matrix(A, N * N);
        generate_random_matrix(B, N * N);

        // 预热
        warmup_run(A, B, C, N);

        // 正式测试
        std::vector<uint64_t> times_ns = benchmark_run(A, B, C, N);
        double avg_gflops = calculate_avg_gflops(times_ns, N);

        std::cout << "GFLOPS (MKL): " << std::fixed << std::setprecision(2) << avg_gflops << std::endl;

        // 写入 CSV：每行一个 N
        write_to_csv("matrix_gflops.csv", "MKL", N, avg_gflops);

        // 释放内存
        free(A);
        free(B);
        free(C);
    }

    std::cout << "All MKL results saved to matrix_gflops.csv" << std::endl;
    return 0;
}
