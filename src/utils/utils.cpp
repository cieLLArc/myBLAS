#include "utils.h"

double g_tsc_freq = 2419200000.0;


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
