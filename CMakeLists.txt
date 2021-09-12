cmake_minimum_required(VERSION 3.20)
project(KDTree)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(googletest)

include_directories(.)

add_executable(KDTree
        kd-tree.cpp
        kd-tree.h
        main.cpp)

# Unit tests
add_executable( runUnitTests test/kd-tree-test.cpp)
target_link_libraries(runUnitTests gtest gtest_main)
add_test( runUnitTests runUnitTests )