cmake_minimum_required(VERSION 3.10)
project(MyBLAS VERSION 1.0.0 LANGUAGES CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native -Wall -O3")

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 收集源文件（根据你的实际结构调整）
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/*.c"
)

# 如果有源文件，创建库
if(SOURCES)
    add_library(myblas ${SOURCES})
endif()

# 如果有 main.cpp，创建可执行文件
if(EXISTS "${PROJECT_SOURCE_DIR}/main.cpp")
    add_executable(myBLASmain main.cpp)
    if(TARGET myblas)
        target_link_libraries(myBLASmain myblas)
    endif()
endif()

# 打印信息
message(STATUS "Project source dir: ${PROJECT_SOURCE_DIR}")
message(STATUS "Found sources: ${SOURCES}")
