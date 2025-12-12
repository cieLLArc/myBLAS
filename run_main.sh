echo "Building myBLAS..."
rm -rf build
mkdir -p build
cd build
cmake .. || { echo "CMake failed"; exit 1; }
make -j$(nproc) || { echo "Make failed"; exit 1; }
echo "Build completed."
echo ""
