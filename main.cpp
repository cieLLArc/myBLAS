#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <iostream>

#include <immintrin.h>
#include <pthread.h>
#include <sys/mman.h>

#include "myBLAS_level3.h"

#include "utils.h"

int main(int argc, char *argv[])
{

    // size_t nBytesA = m * k * sizeof(float);
    // size_t nBytesB = k * n * sizeof(float);
    // size_t nBytesC = m * n * sizeof(float);

    // float *A = ;

    // return 0;
    // int m = 8;
    // int n = 8;
    // int k = 8;

    int m = 5000;
    int n = 5000;
    int k = 5000;

    double *A = (double *)aligned_alloc(32, sizeof(double) * m * k);
    double *B = (double *)aligned_alloc(32, sizeof(double) * k * n);
    double *C = (double *)aligned_alloc(32, sizeof(double) * m * n);

    if (!A || !B || !C)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        free(A);
        free(B);
        free(C);
        return 1;
    }

    memset(A, 0, sizeof(double) * m * k);
    memset(B, 0, sizeof(double) * k * n);
    memset(C, 0, sizeof(double) * m * n);

    srand(0); // 为了验证优化后的结果是否正确，故使用相同的一组随机数

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            A[i * k + j] = rand_double();
        }
    }

    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            B[i * n + j] = rand_double();
        }
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i * n + j] = rand_double();
        }
    }

    compute_fma_ymm_alignment_blocking_openMP(A, B, C, m, n, k);

    free(A);
    free(B);
    free(C);

    return 0;
}