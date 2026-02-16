#include "DataTypes.h"
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

// Block parameters
#define BM 128 // rows
#define BN 128 // cols
#define BK 8   // common dimension
#define TM 8   // rows/thread
#define TN 8   // cols/thread
static constexpr int THREADS_BLOCKS_IN_Y = BM / TM;
static constexpr int THREADS_BLOCKS_IN_X = BN / TN;
static constexpr int NUM_THREADS_PER_BLOCK = THREADS_BLOCKS_IN_Y * THREADS_BLOCKS_IN_X;

// Block size in A : BM x BK
// Block size in B : BK x BN

// Macros enabling optimizations
#define OPT_SHARED_BLOCK_UPDATE 1
#define OPT_THREAD_DATA_UPDATE 1

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

    float threadData[TM][TN] = { 0.0f };
    float rowA[TM];
    float colB[TN];

    dim2 blockCellOffsetMatA = { 0, blockIdx.y * BM };
    dim2 blockCellOffsetMatB = { blockIdx.x * BN, 0 };
    dim2 blockCellOffsetMatC = { blockIdx.x * BN, blockIdx.y * BM };
    dim2 threadBlockOffset = { (threadIdx.x % THREADS_BLOCKS_IN_X) * TN, (threadIdx.x / THREADS_BLOCKS_IN_X) * TM };

#if OPT_SHARED_BLOCK_UPDATE
    dim2 baseCellIdxBlockA = { threadIdx.x % BK, threadIdx.x / BK };
    dim2 baseCellIdxBlockB = { threadIdx.x % BN, threadIdx.x / BN };
    uint32_t strideBlockA = NUM_THREADS_PER_BLOCK / BK;
    uint32_t strideBlockB = NUM_THREADS_PER_BLOCK / BN;
#endif

    for (int blockIt = 0; blockIt < K; blockIt += BK)
    {
#if OPT_SHARED_BLOCK_UPDATE
        for (uint i = 0; i < BM; i += strideBlockA)
        {
            dim2 cellIdxMatA = { blockCellOffsetMatA.x + baseCellIdxBlockA.x,
                                 blockCellOffsetMatA.y + baseCellIdxBlockA.y + i };
            sharedBlockA[baseCellIdxBlockA.y + i][baseCellIdxBlockA.x] = matA[cellIdxMatA.y * K + cellIdxMatA.x];
        }

        for (uint i = 0; i < BK; i += strideBlockB)
        {
            dim2 cellIdxMatB = { blockCellOffsetMatB.x + baseCellIdxBlockB.x,
                                 blockCellOffsetMatB.y + baseCellIdxBlockB.y + i };
            sharedBlockB[baseCellIdxBlockB.y + i][baseCellIdxBlockB.x] = matB[cellIdxMatB.y * N + cellIdxMatB.x];
        }
#else
        for (uint i = threadIdx.x; i < (BM * BK); i += NUM_THREADS_PER_BLOCK)
        {
            dim2 cellIdxBlockA = { i % BK, i / BK }; // local index within a block in A
            dim2 cellIdxMatA = { blockCellOffsetMatA.x + cellIdxBlockA.x, blockCellOffsetMatA.y + cellIdxBlockA.y };
            sharedBlockA[cellIdxBlockA.y][cellIdxBlockA.x] = matA[cellIdxMatA.y * K + cellIdxMatA.x];
        }

        for (uint i = threadIdx.x; i < (BK * BN); i += NUM_THREADS_PER_BLOCK)
        {
            dim2 cellIdxBlockB = { i % BN, i / BN }; // local index within a block in B
            dim2 cellIdxMatB = { blockCellOffsetMatB.x + cellIdxBlockB.x, blockCellOffsetMatB.y + cellIdxBlockB.y };
            sharedBlockB[cellIdxBlockB.y][cellIdxBlockB.x] = matB[cellIdxMatB.y * N + cellIdxMatB.x];
        }
#endif

        __syncthreads();

        // Every thread computes the associated threadData
#if OPT_THREAD_DATA_UPDATE
        for (int i = 0; i < BK; ++i)
        {
            for (int row = 0; row < TM; ++row)
            {
                rowA[row] = sharedBlockA[threadBlockOffset.y + row][i];
            }

            for (int col = 0; col < TN; ++col)
            {
                colB[col] = sharedBlockB[i][threadBlockOffset.x + col];
            }

            for (int row = 0; row < TM; ++row)
            {
                for (int col = 0; col < TN; ++col)
                {
                    threadData[row][col] += rowA[row] * colB[col];
                }
            }
        }
#else
        for (int row = 0; row < TM; ++row)
        {
            for (int col = 0; col < TN; ++col)
            {
                for (int i = 0; i < BK; ++i)
                {
                    threadData[row][col] +=
                        sharedBlockA[threadBlockOffset.y + row][i] * sharedBlockB[i][threadBlockOffset.x + col];
                }
            }
        }
#endif

        __syncthreads();

        blockCellOffsetMatA.x += BK;
        blockCellOffsetMatB.y += BK;
    }

    for (int row = 0; row < TM; ++row)
    {
        for (int col = 0; col < TN; ++col)
        {
            dim2 cellIdxMatC = { blockCellOffsetMatC.x + threadBlockOffset.x + col,
                                 blockCellOffsetMatC.y + threadBlockOffset.y + row };
            matC[cellIdxMatC.y * N + cellIdxMatC.x] = threadData[row][col];
        }
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
    constexpr dim3 threadDims{ (blockDims.x * blockDims.y) / (TM * TN) };
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