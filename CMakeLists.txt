cmake_minimum_required(VERSION 3.16)
project(micrograd_cpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(micrograd_cpp
    test.cpp
    value.cpp
    ops.cpp
)