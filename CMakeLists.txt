cmake_minimum_required(VERSION 3.12)
project(MyBLAS VERSION 1.0.0 LANGUAGES CXX)

# 设置默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/level3/sgemm.cpp

    ${PROJECT_SOURCE_DIR}/src/utils/utils.cpp
)

if(SOURCES)
    add_library(myblas ${SOURCES})
    target_include_directories(myblas PUBLIC ${PROJECT_SOURCE_DIR}/include)

    find_package(OpenMP REQUIRED)
    if(OpenMP_FOUND)
        target_link_libraries(myblas PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(myblas PRIVATE
            -O0
            -g
            -march=native
            -funroll-loops 
            -ffast-math
            -Wall
            -Wextra
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(myblas PRIVATE
            -O3
            -march=native
            -funroll-loops 
            -ffast-math
            -Wall
            -Wextra
        )
    endif()
    target_compile_features(myblas PRIVATE cxx_std_11)
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/main.cpp")
    add_executable(myBLASmain main.cpp)

    if(TARGET myblas)
        target_link_libraries(myBLASmain PRIVATE myblas)
    endif()

    target_include_directories(myBLASmain PRIVATE ${PROJECT_SOURCE_DIR}/include)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(myBLASmain PRIVATE
            -O0
            -g
            -Wall
            -Wextra
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(myBLASmain PRIVATE
            -O3
            -Wall
            -Wextra
        )
    endif()
    target_compile_features(myblas PRIVATE cxx_std_11)
endif()

option(BUILD_BENCHMARKS "Build benchmarks" ON)

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

# ========================
# Project Info
# ========================
message(STATUS "")
message(STATUS "=== [MAIN PROJECT] ===")
message(STATUS "Project         : ${PROJECT_NAME}")
message(STATUS "Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler        : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard    : cxx_std_${CMAKE_CXX_STANDARD}")


# ========================
# Build Targets
# ========================
message(STATUS "Built targets")
foreach(TARGET myBLASmain bench_sgemm)
    message(STATUS "  • ${TARGET}")
endforeach()

# ========================
# Warnings & Notices
# ========================
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING "Non-Release build: performance may be suboptimal.")
endif()
message(STATUS "")
