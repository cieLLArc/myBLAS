cmake_minimum_required(VERSION 3.12)

add_executable(test_myBLAS_dgemm test_myBLAS_dgemm.cpp)
target_link_libraries(test_myBLAS_dgemm PRIVATE myblas)
target_compile_options(test_myBLAS_dgemm PRIVATE
    -g 
    -O0 
    -Wall 
    -Wextra
)
target_compile_features(test_myBLAS_dgemm PRIVATE cxx_std_11)



# set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
# set(MKL_DIR ${MKL_ROOT}/lib/cmake/mkl)
# find_package(MKL CONFIG REQUIRED)

# add_executable(generate_trace generate_trace.cpp)
# target_link_libraries(generate_trace PRIVATE MKL::MKL)
# target_compile_options(generate_trace PRIVATE
#     -O3 
#     -Wall 
#     -Wextra
# )
# target_compile_features(generate_trace PRIVATE cxx_std_11)


message(STATUS "=== [TESTBENCH] Built: test_myBLAS_dgemm, generate_trace ===")
message(STATUS "")
message(STATUS "=== [TESTBENCH] ===")
message(STATUS "[TEST] Build Type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "[TEST] Compiler    : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "[TEST] C++ Standard: cxx_std_${CMAKE_CXX_STANDARD}")
message(STATUS "[TEST] Executable  : ${CMAKE_CURRENT_BINARY_DIR}/test_myBLAS_dgemm")
message(STATUS "")

