cmake_minimum_required(VERSION 3.10)
project(micrograd_cpp)

set(CMAKE_CXX_STANDARD 17)

# Add all source files
add_executable(micrograd_cpp
    test.cpp
    value.cpp
    ops.cpp
)