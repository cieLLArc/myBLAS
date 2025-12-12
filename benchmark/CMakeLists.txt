cmake_minimum_required(VERSION 3.12)

set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
set(MKL_DIR ${MKL_ROOT}/lib/cmake/mkl)
find_package(MKL CONFIG REQUIRED)

# 通用配置
set(BENCH_TARGETS bench_myBLAS_dgemm bench_mklBLAS_dgemm)
set(COMPILE_OPTS -O3 -Wall -Wextra)

foreach(target ${BENCH_TARGETS})
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} PRIVATE myblas)
    target_compile_options(${target} PRIVATE ${COMPILE_OPTS})
    target_compile_features(${target} PRIVATE cxx_std_11)
endforeach()

# MKL benchmark 额外链接 MKL
target_link_libraries(bench_mklBLAS_dgemm PRIVATE MKL::MKL)

message(STATUS "=== [BENCHMARK] Built: ${BENCH_TARGETS} ===")


message(STATUS "")
message(STATUS "=== [BENCHMARK] ===")
message(STATUS "[BENCH] Build Type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "[BENCH] Compiler    : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "[BENCH] C++ Standard: cxx_std_${CMAKE_CXX_STANDARD}")
message(STATUS "[BENCH] Executable  : ${CMAKE_CURRENT_BINARY_DIR}/bench_myBLAS_dgemm")
message(STATUS "[BENCH] Executable  : ${CMAKE_CURRENT_BINARY_DIR}/bench_mklBLAS_dgemm")
message(STATUS "")


