#include "utils.h"
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>

// 声明你的内核函数
extern void compute_fma_ymm_alignment_blocking_openMP_optimized_fast(double *A, double *B, double *C, int m, int n, int k);

void warmup_run(double *A, double *B, double *C, int N)
{
    for (int i = 0; i < 3; ++i)
    {
        compute_fma_ymm_alignment_blocking_openMP_optimized_fast(A, B, C, N, N, N);
    }
}

std::vector<uint64_t> benchmark_run(double *A, double *B, double *C, int N, int trials = 10)
{
    std::vector<uint64_t> times_ns;
    times_ns.reserve(trials);
    for (int i = 0; i < trials; ++i)
    {
        uint64_t start = get_timestamp_ns();
        compute_fma_ymm_alignment_blocking_openMP_optimized_fast(A, B, C, N, N, N);
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
        // 分配内存
        double *A = (double *)aligned_alloc(32, sizeof(double) * N * N);
        double *B = (double *)aligned_alloc(32, sizeof(double) * N * N);
        double *C = (double *)aligned_alloc(32, sizeof(double) * N * N);
        if (!A || !B || !C)
        {
            std::cerr << "Memory allocation failed for size " << N << std::endl;
            continue;
        }
        generate_random_matrix(A, N * N);
        generate_random_matrix(B, N * N);
        warmup_run(A, B, C, N);
        std::vector<uint64_t> times_ns = benchmark_run(A, B, C, N);
        double avg_gflops = calculate_avg_gflops(times_ns, N);
        std::cout << "GFLOPS: " << std::fixed << std::setprecision(2) << avg_gflops << std::endl;
        // 写入 CSV：每行一个 N
        write_to_csv("matrix_gflops.csv", "CustomKernel", N, avg_gflops);
        free(A);
        free(B);
        free(C);
    }
    std::cout << "All results saved to matrix_gflops.csv" << std::endl;
    return 0;
}
