#pragma once

void compute_fma_ymm_alignment_blocking_openMP(double *A, double *B, double *C, int m, int n, int k);
void compute_fma_ymm_alignment_blocking_openMP_optimized_fast(double *A, double *B, double *C, int m, int n, int k);