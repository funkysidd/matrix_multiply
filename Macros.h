#pragma once

#define CUDA_CHECK(expr_to_check) do {                 \
    cudaError_t result  = expr_to_check;               \
    if(result != cudaSuccess)                          \
    {                                                  \
        fprintf(stderr,                                \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                              \
                __LINE__,                              \
                result,                                \
                cudaGetErrorString(result));           \
    }                                                  \
} while(0)

#define MATRIX_DIMS(numBlocks, blockDims) { numBlocks.x * blockDims.x, numBlocks.y * blockDims.y }

#define GET_SIZE(dims) (dims.x * dims.y)

#define ASSIGN_MATRIX(data, dims, val) do {    \
    for (int row = 0; row < dims.y; ++row)     \
    {                                          \
        for (int col = 0; col < dims.x; ++col) \
        {                                      \
            data[row * dims.x + col] = val;    \
        }                                      \
    }                                          \
} while (0)

#define VERIFY_RESULT(data, dims, expVal, maxError)  do {                           \
    for (int row = 0; row < dims.y; ++row)                                          \
    {                                                                               \
        for (int col = 0; col < dims.x; ++col)                                      \
        {                                                                           \
            float curVal = data[row * dims.x + col];                                \
            maxError = cuda::std::fmax(maxError, cuda::std::fabs(curVal - expVal)); \
        }                                                                           \
    }                                                                               \
} while (0)
