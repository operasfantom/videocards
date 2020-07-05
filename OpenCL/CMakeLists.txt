cmake_minimum_required(VERSION 3.1)
project(OpenCL)

message(STATUS "OpenCL CMake loading...")

add_executable(lab1
        lab1.c
        matrix_multiplication.c matrix_multiplication.h
        matrix_util.c matrix_util.h
        )
add_executable(lab2 lab2.c prefix_sum_util.c prefix_sum_util.h)

add_library(common STATIC
        cl_util.c cl_util.h
        constants.h constants.c lab2.c prefix_sum.c prefix_sum.h)
target_link_libraries(lab1 common)
target_link_libraries(lab2 common)

set_property(TARGET lab1 common PROPERTY C_STANDARD 11)
set_property(TARGET lab2 common PROPERTY C_STANDARD 11)

message(STATUS "Checking OpenCL...")

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

if (OPENCL_FOUND)
    message(STATUS "Found OpenCL!")
    # add OpenCL to project
    target_include_directories(lab1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(lab1 ${OpenCL_LIBRARY})
    target_link_libraries(lab2 ${OpenCL_LIBRARY})
else ()
    message(STATUS "Missed OpenCL!")
endif ()