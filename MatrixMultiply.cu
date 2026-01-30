#include "Macros.h"

#include <cuda_runtime_api.h>
#include <cuda/cmath>

#include <cstdio>
#include <chrono>

#define BLOCKSIZE 32

__global__ void matMul(
    float* matA,
    dim3 dimsA,
    dim3 blocksA,
    ///
    float* matB,
    dim3 dimsB,
    dim3 blocksB,
    ///
    float *matC, // c = a x b
    dim3 dimsC
)
{
    __shared__ float blockA[BLOCKSIZE][BLOCKSIZE];
    __shared__ float blockB[BLOCKSIZE][BLOCKSIZE+1];
    __shared__ float blockC[BLOCKSIZE][BLOCKSIZE];

    dim3 mappedBlockA = { blockIdx.y % blocksA.x, blockIdx.y / blocksA.x };
    dim3 mappedBlockB = { blockIdx.x, mappedBlockA.x };
    
    dim3 blockOffsetA = { mappedBlockA.x * BLOCKSIZE, mappedBlockA.y * BLOCKSIZE };
    dim3 threadOffsetA = { blockOffsetA.x + threadIdx.x, blockOffsetA.y + threadIdx.y };
    dim3 blockOffsetB = { mappedBlockB.x * BLOCKSIZE, mappedBlockB.y * BLOCKSIZE };
    dim3 threadOffsetB = { blockOffsetB.x + threadIdx.x, blockOffsetB.y + threadIdx.y };

    blockA[threadIdx.y][threadIdx.x] = matA[threadOffsetA.y * dimsA.x + threadOffsetA.x];
    blockB[threadIdx.y][threadIdx.x] = matB[threadOffsetB.y * dimsB.x + threadOffsetB.x];

    __syncthreads();

    blockC[threadIdx.y][threadIdx.x] = 0;
    for (int idx = 0; idx < BLOCKSIZE; ++idx)
    {
        blockC[threadIdx.y][threadIdx.x] += blockA[threadIdx.y][idx] * blockB[idx][threadIdx.x];
    }

    dim3 mappedBlockC = { blockIdx.x, blockIdx.y / blocksB.y };
    dim3 blockOffsetC = { mappedBlockC.x * BLOCKSIZE, mappedBlockC.y * BLOCKSIZE };
    dim3 threadOffsetC = { blockOffsetC.x + threadIdx.x, blockOffsetC.y + threadIdx.y, 0 };

    atomicAdd(&matC[threadOffsetC.y * dimsC.x + threadOffsetC.x], blockC[threadIdx.y][threadIdx.x]);
}

int main()
{
    int cudaDeviceCount = -1;
    CUDA_CHECK(cudaGetDeviceCount(&cudaDeviceCount));

    if (cudaDeviceCount < 1)
    {
        fprintf(stderr, "No CUDA device found, exiting!\n");
        return 0;
    }

    CUDA_CHECK(cudaSetDevice(0));

    constexpr dim3 dimsMatA{ 4096, 4096 }; // x (cols), y (rows)
    constexpr dim3 dimsMatB{ 4096, 4096 }; // x (cols), y (rows)

    assert(dimsMatA.x == dimsMatB.y);

    // Compute number of 32x32 blocks in each of the matrices
    constexpr dim3 blockDims{ BLOCKSIZE, BLOCKSIZE };
    constexpr dim3 blocksMatA{ cuda::ceil_div(dimsMatA.x, blockDims.x), cuda::ceil_div(dimsMatA.y, blockDims.y) };
    constexpr dim3 blocksMatB{ cuda::ceil_div(dimsMatB.x, blockDims.x), cuda::ceil_div(dimsMatB.y, blockDims.y) };

    assert(blocksMatA.x == blocksMatB.y);

    constexpr dim3 blocksMatC{ blocksMatB.x, blocksMatA.y };

    constexpr dim3 dimsMatAMod = MATRIX_DIMS(blocksMatA, blockDims);
    constexpr dim3 dimsMatBMod = MATRIX_DIMS(blocksMatB, blockDims);
    constexpr dim3 dimsMatCMod = MATRIX_DIMS(blocksMatC, blockDims);

    float* matADevice = nullptr;
    float* matBDevice = nullptr;
    float* matCDevice = nullptr;

    float* matAHost = nullptr;
    float* matBHost = nullptr;
    float* matCHost = nullptr;

    CUDA_CHECK(cudaMallocHost(&matAHost, GET_SIZE(dimsMatAMod) * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&matBHost, GET_SIZE(dimsMatBMod) * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&matCHost, GET_SIZE(dimsMatCMod) * sizeof(float)));

    ASSIGN_MATRIX(matAHost, dimsMatAMod, 1.0f);
    ASSIGN_MATRIX(matBHost, dimsMatBMod, 1.0f);

    CUDA_CHECK(cudaMalloc(&matADevice, GET_SIZE(dimsMatAMod) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matBDevice, GET_SIZE(dimsMatBMod) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matCDevice, GET_SIZE(dimsMatCMod) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(matADevice, matAHost, GET_SIZE(dimsMatAMod) * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matBDevice, matBHost, GET_SIZE(dimsMatBMod) * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(matCDevice, 0, GET_SIZE(dimsMatCMod) * sizeof(float)));
    
    constexpr dim3 gridDims{ blocksMatC.x, blocksMatC.y * blocksMatB.y };

    cudaEvent_t startEvent;
    cudaEvent_t endEvent;

    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&endEvent));

    CUDA_CHECK(cudaEventRecord(startEvent, 0));
    matMul<<<gridDims, blockDims>>>(matADevice, dimsMatAMod, blocksMatA, matBDevice, dimsMatBMod, blocksMatC, matCDevice, dimsMatCMod);
    CUDA_CHECK(cudaEventRecord(endEvent, 0));
    CUDA_CHECK(cudaEventSynchronize(endEvent));

    float elapsedTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, startEvent, endEvent));
    fprintf(stdout, "Duration: %0.4f (milliseconds)\n", elapsedTime);

    constexpr float totalOps = 2.0f * dimsMatCMod.x * dimsMatCMod.y * dimsMatAMod.x;
    float flops = totalOps / (elapsedTime / 1000.f);
    fprintf(stdout, "Performance (GFLOPS): %0.4f\n", flops / 1.e9); // A giga flop is 10^9 flops
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventDestroy(endEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));

    CUDA_CHECK(cudaMemcpy(matCHost, matCDevice, GET_SIZE(dimsMatCMod) * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));

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