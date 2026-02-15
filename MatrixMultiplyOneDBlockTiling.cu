#include "Macros.h"

#include <cuda/cmath>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdio>

// Matrix dimensions
#define M 4096 // rows
#define N 4096 // cols
#define K 4096 // common dimension

// Matrix A : M x K
// Matrix B : K x N

// Block dimensions
#define BM 64 // rows
#define BN 64 // cols
#define BK 8  // common dimension
#define TM 8  // rows/thread

// Block size in A : BM x BK
// Block size in B : BK x BN

struct dim2
{
    uint x;
    uint y;
};

__global__ void matMul(float *matA, dim3 dimsA, dim3 blocksA,
                       ///
                       float *matB, dim3 dimsB, dim3 blocksB,
                       ///
                       float *matC, // c = a x b
                       dim3 dimsC)
{
    /// @todo: All references to `M`, `N`, `K` need to be relaced with the actual dims of `matA` and `matB`

    __shared__ float sharedBlockA[BM][BK];
    __shared__ float sharedBlockB[BK][BN];

    float threadData[TM] = { 0.0f };

    dim2 blockCellOffsetMatA = { 0, blockIdx.y * BM };
    dim2 blockCellOffsetMatB = { blockIdx.x * BN, 0 };
    dim2 blockCellOffsetMatC = { blockIdx.x * BN, blockIdx.y * BM };

    dim2 cellIdxBlockA = { threadIdx.x % BK, threadIdx.x / BK }; // local index within a block in A
    dim2 cellIdxBlockB = { threadIdx.x % BN, threadIdx.x / BN }; // local index within a block in B

    for (int blockIt = 0; blockIt < K; blockIt += BK)
    {
        dim2 cellIdxMatA = { blockCellOffsetMatA.x + cellIdxBlockA.x, blockCellOffsetMatA.y + cellIdxBlockA.y };
        dim2 cellIdxMatB = { blockCellOffsetMatB.x + cellIdxBlockB.x, blockCellOffsetMatB.y + cellIdxBlockB.y };

        sharedBlockA[cellIdxBlockA.y][cellIdxBlockA.x] = matA[cellIdxMatA.y * K + cellIdxMatA.x];
        sharedBlockB[cellIdxBlockB.y][cellIdxBlockB.x] = matB[cellIdxMatB.y * N + cellIdxMatB.x];

        __syncthreads();

        // Both variants below result in similar performance
#if 0
        for (int idx = 0; idx < TM; ++idx)
        {
            dim2 cellIdxBlockC = { cellIdxBlockB.x, cellIdxBlockB.y * BK + idx };
            for (int in = 0; in < BK; ++in)
            {
                threadData[idx] += sharedBlockA[cellIdxBlockC.y][in] * sharedBlockB[in][cellIdxBlockC.x];
            }
        }
#else
        for (int idx = 0; idx < BK; ++idx)
        {
            // For a given value in a column in B, computes all the thread data
            float tmpB = sharedBlockB[idx][cellIdxBlockB.x];
            for (int in = 0; in < TM; ++in)
            {
                threadData[idx] += sharedBlockA[cellIdxBlockB.y * BK + in][idx] * tmpB;
            }
        }
#endif

        __syncthreads();

        blockCellOffsetMatA.x += BK;
        blockCellOffsetMatB.y += BK;
    }

    for (int idx = 0; idx < BK; ++idx)
    {
        dim2 cellIdxBlockC = { cellIdxBlockB.x, cellIdxBlockB.y * BK + idx };
        dim2 cellIdxMatC = { blockCellOffsetMatC.y + cellIdxBlockC.y, blockCellOffsetMatC.x + cellIdxBlockC.x };
        matC[cellIdxMatC.y * N + cellIdxMatC.x] = threadData[idx];
    }
}

int main()
{
    int cudaDeviceCount = -1;
    CUDA_CHECK(cudaGetDeviceCount(&cudaDeviceCount));

    if (cudaDeviceCount < 1)
    {
        fprintf(stderr, "No CUDA device found, exiting!\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    constexpr dim3 dimsMatA{ K, M }; // x (cols), y (rows)
    constexpr dim3 dimsMatB{ N, K }; // x (cols), y (rows)

    assert(dimsMatA.x == dimsMatB.y);

    constexpr dim3 blockDims{ BN, BM };
    constexpr dim3 threadDims{ (blockDims.x * blockDims.y) / TM };
    constexpr dim3 blocksMatA{ cuda::ceil_div(dimsMatA.x, blockDims.x), cuda::ceil_div(dimsMatA.y, blockDims.y) };
    constexpr dim3 blocksMatB{ cuda::ceil_div(dimsMatB.x, blockDims.x), cuda::ceil_div(dimsMatB.y, blockDims.y) };

    assert(blocksMatA.x == blocksMatB.y);

    constexpr dim3 blocksMatC{ blocksMatB.x, blocksMatA.y };

    constexpr dim3 dimsMatAMod = MATRIX_DIMS(blocksMatA, blockDims);
    constexpr dim3 dimsMatBMod = MATRIX_DIMS(blocksMatB, blockDims);
    constexpr dim3 dimsMatCMod = MATRIX_DIMS(blocksMatC, blockDims);

    float *matADevice = nullptr;
    float *matBDevice = nullptr;
    float *matCDevice = nullptr;

    float *matAHost = nullptr;
    float *matBHost = nullptr;
    float *matCHost = nullptr;

    CUDA_CHECK(cudaMallocHost(&matAHost, GET_SIZE(dimsMatAMod) * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&matBHost, GET_SIZE(dimsMatBMod) * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&matCHost, GET_SIZE(dimsMatCMod) * sizeof(float)));

    ASSIGN_MATRIX(matAHost, dimsMatAMod, 1.0f);
    ASSIGN_MATRIX(matBHost, dimsMatBMod, 1.0f);

    CUDA_CHECK(cudaMalloc(&matADevice, GET_SIZE(dimsMatAMod) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matBDevice, GET_SIZE(dimsMatBMod) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matCDevice, GET_SIZE(dimsMatCMod) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(matADevice, matAHost, GET_SIZE(dimsMatAMod) * sizeof(float),
                          cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matBDevice, matBHost, GET_SIZE(dimsMatBMod) * sizeof(float),
                          cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(matCDevice, 0, GET_SIZE(dimsMatCMod) * sizeof(float)));

    constexpr dim3 gridDims{ blocksMatC.x, blocksMatC.y };

    cudaEvent_t startEvent;
    cudaEvent_t endEvent;

    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&endEvent));

    CUDA_CHECK(cudaEventRecord(startEvent, 0));
    matMul<<<gridDims, threadDims>>>(matADevice, dimsMatAMod, blocksMatA, matBDevice, dimsMatBMod, blocksMatC,
                                     matCDevice, dimsMatCMod);
    CUDA_CHECK(cudaEventRecord(endEvent, 0));
    CUDA_CHECK(cudaEventSynchronize(endEvent));

    float elapsedTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTimeMs, startEvent, endEvent));
    fprintf(stdout, "Duration: %0.4f (milliseconds)\n", elapsedTimeMs);

    constexpr float totalOps = 2.0f * dimsMatCMod.x * dimsMatCMod.y * dimsMatAMod.x;
    float flops = totalOps / (elapsedTimeMs / 1000.f);
    fprintf(stdout, "Performance (GFLOPS): %0.4f\n", flops / 1.e9); // A giga flop is 10^9 flops

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventDestroy(endEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));

    CUDA_CHECK(cudaMemcpy(matCHost, matCDevice, GET_SIZE(dimsMatCMod) * sizeof(float),
                          cudaMemcpyKind::cudaMemcpyDeviceToHost));

    float maxError = 0.0f;
    VERIFY_RESULT(matCHost, dimsMatCMod, 4096.0f, maxError);
    fprintf(stdout, "Max error: %0.4f\n", maxError);

    CUDA_CHECK(cudaFree(matCDevice));
    CUDA_CHECK(cudaFree(matBDevice));
    CUDA_CHECK(cudaFree(matADevice));

    CUDA_CHECK(cudaFreeHost(matCHost));
    CUDA_CHECK(cudaFreeHost(matBHost));
    CUDA_CHECK(cudaFreeHost(matAHost));

    return 0;
}