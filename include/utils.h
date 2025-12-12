#pragma once

#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <ctime>

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>

extern double g_tsc_freq;

void *page_alloc_aligned(size_t size);

void generate_random_matrix(double *mat, int size);

//
double calculate_avg_gflops(const std::vector<uint64_t> &times_ns, int N);

void write_to_csv(const std::string &filename,
                  const std::string &label,
                  int N,
                  double gflops);

void sgemm_mkl(double *A, double *B, double *C, int m, int n, int k);
void gemm_mkl(double *A, double *B, double *C, int m, int n, int k);

inline uint64_t get_cycles()
{
    uint32_t lo, hi;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi) : : "rcx", "memory");
    return ((uint64_t)hi << 32) | lo;
}

inline double get_timestamp_seconds()
{
    return get_cycles() / g_tsc_freq;
}

inline uint64_t get_timestamp_ns()
{
    return static_cast<uint64_t>(get_cycles() * 1e9 / g_tsc_freq);
}

inline double rand_double()
{
    return rand() / (RAND_MAX + 1.0);
}
