#include <cstdio>
#include <immintrin.h>

#include "blas_level3.h"

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
        printf("row %d done\n", i);
    }
}