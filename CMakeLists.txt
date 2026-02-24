cmake_minimum_required(VERSION 3.10)
project(cuda_testbed LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Gather all .cu files
file(GLOB CUDA_SOURCES "${CMAKE_SOURCE_DIR}/*.cu")

set(CUBLAS_SAMPLE "CuBlas")

# Create an executable for each .cu file
foreach(CUDA_FILE ${CUDA_SOURCES})
    get_filename_component(EXE_NAME ${CUDA_FILE} NAME_WE) # extract file name without extension.
    add_executable(${EXE_NAME} ${CUDA_FILE})
    # set_target_properties(${EXE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${EXE_NAME} PROPERTIES CUDA_ARCHITECTURES native)
    if (${EXE_NAME} STREQUAL ${CUBLAS_SAMPLE})
        message(STATUS "Linking CuBlas against ${CUDA_CUBLAS_LIBRARIES}")
        target_link_libraries(${EXE_NAME} ${CUDA_CUBLAS_LIBRARIES})
    endif()
endforeach()
