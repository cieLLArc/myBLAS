#include <cstdio>
#include <cmath>
#include <immintrin.h>

#include "myBLAS_level3.h"

void naive_col_major_sgemm(char transa, char transb, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;

    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float acc = 0.f;
            const float *a_ptr = A + m * a_stride_m;
            const float *b_ptr = B + n * b_stride_n;

            for (int k = 0; k < K; k++)
            {
                acc += a_ptr[0] * b_ptr[0];
                a_ptr += a_stride_k;
                b_ptr += b_stride_k;
            }

            C[m + n * ldc] = alpha * acc + beta * C[m + n * ldc];
        }
    }
}

void naive_col_major_dgemm(char transa, char transb, int M, int N, int K, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc)
{
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;

    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            double acc = 0.f;
            const double *a_ptr = A + m * a_stride_m;
            const double *b_ptr = B + n * b_stride_n;

            for (int k = 0; k < K; k++)
            {
                acc += a_ptr[0] * b_ptr[0];
                a_ptr += a_stride_k;
                b_ptr += b_stride_k;
            }

            C[m + n * ldc] = alpha * acc + beta * C[m + n * ldc];
        }
    }
}

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
    constexpr int MC = 256;
    constexpr int NC = 4096;
    constexpr int KC = 256;
    constexpr int MR = 4;
    constexpr int NR = 4;

// 设置线程亲和性
#pragma omp parallel proc_bind(close)
    {
#pragma omp for schedule(static) collapse(2) nowait
        for (int jc = 0; jc < n; jc += NC)
        {
            for (int ic = 0; ic < m; ic += MC)
            {
                int mb = std::min(MC, m - ic);
                int nb = std::min(NC, n - jc);

                // 按 K 维度分块（最内层）
                for (int pc = 0; pc < k; pc += KC)
                {
                    int pb = std::min(KC, k - pc);

                    for (int jr = 0; jr < nb; jr += NR)
                    {
                        for (int ir = 0; ir < mb; ir += MR)
                        {
                            int i = ic + ir;
                            int j = jc + jr;

                            // 第一次迭代：初始化为0
                            __m256d c_0, c_1, c_2, c_3;
                            if (pc == 0)
                            {
                                c_0 = _mm256_setzero_pd();
                                c_1 = _mm256_setzero_pd();
                                c_2 = _mm256_setzero_pd();
                                c_3 = _mm256_setzero_pd();
                            }
                            else
                            {
                                // 后续迭代：加载已有结果
                                c_0 = _mm256_load_pd(&C[(i + 0) * n + j]);
                                c_1 = _mm256_load_pd(&C[(i + 1) * n + j]);
                                c_2 = _mm256_load_pd(&C[(i + 2) * n + j]);
                                c_3 = _mm256_load_pd(&C[(i + 3) * n + j]);
                            }

                            // 内核计算
                            for (int p = pc; p < pc + pb; p++)
                            {
                                // 提前 prefetch
                                if (p + 8 < pc + pb)
                                {
                                    _mm_prefetch((char *)&B[(p + 8) * n + j], _MM_HINT_T0);
                                }

                                __m256d b_vreg = _mm256_load_pd(&B[p * n + j]);

                                __m256d a_0 = _mm256_broadcast_sd(&A[(i + 0) * k + p]);
                                __m256d a_1 = _mm256_broadcast_sd(&A[(i + 1) * k + p]);
                                __m256d a_2 = _mm256_broadcast_sd(&A[(i + 2) * k + p]);
                                __m256d a_3 = _mm256_broadcast_sd(&A[(i + 3) * k + p]);

                                c_0 = _mm256_fmadd_pd(a_0, b_vreg, c_0);
                                c_1 = _mm256_fmadd_pd(a_1, b_vreg, c_1);
                                c_2 = _mm256_fmadd_pd(a_2, b_vreg, c_2);
                                c_3 = _mm256_fmadd_pd(a_3, b_vreg, c_3);
                            }

                            // 写回结果
                            _mm256_store_pd(&C[(i + 0) * n + j], c_0);
                            _mm256_store_pd(&C[(i + 1) * n + j], c_1);
                            _mm256_store_pd(&C[(i + 2) * n + j], c_2);
                            _mm256_store_pd(&C[(i + 3) * n + j], c_3);
                        }
                    }
                }
            }
        }
    }
}
