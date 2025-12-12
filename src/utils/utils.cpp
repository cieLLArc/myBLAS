#include "utils.h"

double tsc_freq = 0.0;

void calibrate_tsc()
{
    uint32_t eax, ebx, ecx, edx;
    uint64_t crystal_hz, tsc_freq_hz;

    // 使用CPUID指令，叶功能0x15（时间戳计数器与核心晶体时钟信息）
    // 输入：EAX = 0x15
    // 输出：EAX = TSC/“核心晶体时钟”分频比 (分母)
    //       EBX = TSC/“核心晶体时钟”倍频比 (分子)
    //       ECX = “核心晶体时钟”频率 (Hz)。对某些平台可能为0。
    asm volatile("cpuid"
                 : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                 : "a"(0x15), "c"(0));

    // 检查返回的分母(EAX)和分子(EBX)是否有效
    if (eax == 0 || ebx == 0)
    {
        std::cerr << "[WARNING] CPUID.15H not supported. Falling back to calibration.\n";
    }

    // 计算TSC频率：TSC (Hz) = (晶体频率 * EBX) / EAX
    if (ecx != 0)
    {
        // 情况1：ECX寄存器直接提供了晶体频率（Hz）
        crystal_hz = (uint64_t)ecx;
    }
    else
    {
        // 情况2：ECX为0，需要使用已知的默认值。
        // 对于包括i7-14650HX在内的大多数现代Intel客户端平台，该值为24 MHz。
        // 注：某些服务器平台可能使用25 MHz。
        crystal_hz = 24000000; // 24,000,000 Hz
    }

    tsc_freq = (crystal_hz * ebx) / eax;
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
