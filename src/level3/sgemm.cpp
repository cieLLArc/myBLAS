#include <cstdio>
#include <cmath>
#include <immintrin.h>

#include "myBLAS_level3.h"

void compute_fma_ymm_alignment_blocking_openMP(double *A, double *B, double *C, int m, int n, int k)
{
#pragma omp parallel for
    for (int i = 0; i < m; i += 4)
    {
        for (int j = 0; j < n; j += 4)
        {
            __m256d c_0_vreg = _mm256_setzero_pd();
            __m256d c_1_vreg = _mm256_setzero_pd();
            __m256d c_2_vreg = _mm256_setzero_pd();
            __m256d c_3_vreg = _mm256_setzero_pd();

            for (int p = 0; p < k; p++)
            {
                __m256d a_0_extended_vreg = _mm256_set1_pd(A[(i + 0) * k + p]);
                __m256d a_1_extended_vreg = _mm256_set1_pd(A[(i + 1) * k + p]);
                __m256d a_2_extended_vreg = _mm256_set1_pd(A[(i + 2) * k + p]);
                __m256d a_3_extended_vreg = _mm256_set1_pd(A[(i + 3) * k + p]);

                __m256d b_vreg = _mm256_load_pd(&B[p * n + j]);

                c_0_vreg = _mm256_fmadd_pd(a_0_extended_vreg, b_vreg, c_0_vreg);
                c_1_vreg = _mm256_fmadd_pd(a_1_extended_vreg, b_vreg, c_1_vreg);
                c_2_vreg = _mm256_fmadd_pd(a_2_extended_vreg, b_vreg, c_2_vreg);
                c_3_vreg = _mm256_fmadd_pd(a_3_extended_vreg, b_vreg, c_3_vreg);
            }
            _mm256_store_pd(&C[(i + 0) * n + j], c_0_vreg);
            _mm256_store_pd(&C[(i + 1) * n + j], c_1_vreg);
            _mm256_store_pd(&C[(i + 2) * n + j], c_2_vreg);
            _mm256_store_pd(&C[(i + 3) * n + j], c_3_vreg);
        }
    }
}

void compute_fma_ymm_alignment_blocking_openMP_optimized_fast(double *A, double *B, double *C, int m, int n, int k)
{
    constexpr int MC = 256;  // L1 cache blocking for A
    constexpr int NC = 4096; // L2 cache blocking for B
    constexpr int KC = 256;  // L1 cache blocking for k
    constexpr int MR = 4;    // Micro-kernel M
    constexpr int NR = 4;    // Micro-kernel N

#pragma omp parallel for collapse(2)
    for (int jc = 0; jc < n; jc += NC)
    {
        for (int pc = 0; pc < k; pc += KC)
        {
            int pb = std::min(KC, k - pc);
            int nb = std::min(NC, n - jc);

            for (int ic = 0; ic < m; ic += MC)
            {
                int mb = std::min(MC, m - ic);

                for (int jr = 0; jr < nb; jr += NR)
                {
                    for (int ir = 0; ir < mb; ir += MR)
                    {
                        int i = ic + ir;
                        int j = jc + jr;

                        // Prefetch next block
                        _mm_prefetch((char *)&B[(pc + 16) * n + j], _MM_HINT_T0);

                        __m256d c_0_vreg = _mm256_load_pd(&C[(i + 0) * n + j]);
                        __m256d c_1_vreg = _mm256_load_pd(&C[(i + 1) * n + j]);
                        __m256d c_2_vreg = _mm256_load_pd(&C[(i + 2) * n + j]);
                        __m256d c_3_vreg = _mm256_load_pd(&C[(i + 3) * n + j]);

                        for (int p = pc; p < pc + pb; p++)
                        {
                            // Prefetch A elements
                            _mm_prefetch((char *)&A[(i + 0) * k + p + 8], _MM_HINT_T0);

                            __m256d b_vreg = _mm256_load_pd(&B[p * n + j]);

                            __m256d a_0 = _mm256_broadcast_sd(&A[(i + 0) * k + p]);
                            __m256d a_1 = _mm256_broadcast_sd(&A[(i + 1) * k + p]);
                            __m256d a_2 = _mm256_broadcast_sd(&A[(i + 2) * k + p]);
                            __m256d a_3 = _mm256_broadcast_sd(&A[(i + 3) * k + p]);

                            c_0_vreg = _mm256_fmadd_pd(a_0, b_vreg, c_0_vreg);
                            c_1_vreg = _mm256_fmadd_pd(a_1, b_vreg, c_1_vreg);
                            c_2_vreg = _mm256_fmadd_pd(a_2, b_vreg, c_2_vreg);
                            c_3_vreg = _mm256_fmadd_pd(a_3, b_vreg, c_3_vreg);
                        }

                        _mm256_store_pd(&C[(i + 0) * n + j], c_0_vreg);
                        _mm256_store_pd(&C[(i + 1) * n + j], c_1_vreg);
                        _mm256_store_pd(&C[(i + 2) * n + j], c_2_vreg);
                        _mm256_store_pd(&C[(i + 3) * n + j], c_3_vreg);
                    }
                }
            }
        }
    }
}
