#pragma once

void naive_col_major_sgemm(char transa, char transb, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);
void naive_col_major_dgemm(char transa, char transb, int M, int N, int K, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);
void compute_fma_ymm_alignment_blocking_openMP(double *A, double *B, double *C, int m, int n, int k);
void compute_fma_ymm_alignment_blocking_openMP_optimized_fast(double *A, double *B, double *C, int m, int n, int k);