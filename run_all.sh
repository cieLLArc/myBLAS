echo "Building myBLAS..."
rm -rf build
mkdir -p build
cd build
cmake .. || { echo "CMake failed"; exit 1; }
make -j$(nproc) || { echo "Make failed"; exit 1; }
echo "Build completed."
echo ""
# -----------------------------------------------------------------------------
# 运行基准测试
# -----------------------------------------------------------------------------
echo "== Running myBLAS DGEMM benchmark =="
./benchmark/bench_myBLAS_dgemm || { echo "Benchmark failed"; exit 1; }
echo "Benchmark completed. Results saved to results/matrix_gflops.csv"
echo ""

echo "== Running myBLAS DGEMM1 benchmark =="
./benchmark/bench_myBLAS_dgemm1 || { echo "Benchmark failed"; exit 1; }
echo "Benchmark completed. Results saved to results/matrix_gflops.csv"
echo ""

echo "== Running mklBLAS DGEMM benchmark =="
./benchmark/bench_mklBLAS_dgemm || { echo "Benchmark failed"; exit 1; }
echo "Benchmark completed. Results saved to results/matrix_gflops.csv"
echo ""
# -----------------------------------------------------------------------------
# 绘图分析
# -----------------------------------------------------------------------------
echo "Generating performance plot..."
python ../scripts/plot_gflops.py || { echo "Plotting failed"; exit 1; }
echo "Plot saved to results/matrix_gflops.pdf"
echo ""
# -----------------------------------------------------------------------------
# 提示
# -----------------------------------------------------------------------------
echo "  All done! Your results are ready:"
echo "   - CSV data : benchmark/results/matrix_gflops.csv"
echo "   - PDF plot : benchmark/results/matrix_gflops.pdf"
echo "   - PNG image: benchmark/results/matrix_gflops.png (if generated)"
echo ""
echo "Pipeline completed successfully!"