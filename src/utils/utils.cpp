#include "utils.h"

double g_tsc_freq = 0.0;

void calibrate_tsc()
{
    if (g_tsc_freq != 0.0)
        return;

    const int iterations = 5;
    double frequencies[iterations];

    for (int i = 0; i < iterations; ++i)
    {
        struct timespec start, end;
        uint64_t tsc_start, tsc_end;

        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        tsc_start = get_cycles();

        struct timespec sleep_time = {1, 0};
        nanosleep(&sleep_time, NULL);

        tsc_end = get_cycles();
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        double wall_time = (end.tv_sec - start.tv_sec) +
                           (end.tv_nsec - start.tv_nsec) / 1e9;

        frequencies[i] = (tsc_end - tsc_start) / wall_time;
    }

    // 排序取中位数
    for (int i = 0; i < iterations - 1; ++i)
        for (int j = i + 1; j < iterations; ++j)
            if (frequencies[i] > frequencies[j])
                std::swap(frequencies[i], frequencies[j]);

    g_tsc_freq = frequencies[iterations / 2];
}

void generate_random_matrix(double *mat, int size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < size; ++i)
    {
        mat[i] = dis(gen);
    }
}

double calculate_avg_gflops(const std::vector<uint64_t> &times_ns, int N)
{
    double total_gflops = 0.0;
    for (uint64_t t : times_ns)
    {
        double flops = 2.0 * N * N * N;
        double time_s = static_cast<double>(t) / 1e9;
        total_gflops += flops / time_s / 1e9;
    }
    return total_gflops / times_ns.size();
}

void write_to_csv(const std::string &filename,
                  const std::string &label,
                  int N,
                  double gflops)
{
    std::ofstream csv(filename, std::ios::app);

    // 如果文件为空，写表头
    if (csv.tellp() == 0)
    {
        csv << "Label,N,GFLOPS\n";
    }

    // 写数据行
    csv << label << "," << N << "," << std::fixed << std::setprecision(6) << gflops << "\n";
}

// static void *page_alloc_aligned(size_t size)
// {

//     return NULL;
// }
