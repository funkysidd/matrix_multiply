cmake_minimum_required(VERSION 3.10)
project(cuda_testbed LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Gather all .cu files
file(GLOB CUDA_SOURCES "${CMAKE_SOURCE_DIR}/*.cu")

# Create an executable for each .cu file
foreach(CUDA_FILE ${CUDA_SOURCES})
    get_filename_component(EXE_NAME ${CUDA_FILE} NAME_WE) # extract file name without extension.
    add_executable(${EXE_NAME} ${CUDA_FILE})
    # set_target_properties(${EXE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${EXE_NAME} PROPERTIES CUDA_ARCHITECTURES native)
endforeach()
