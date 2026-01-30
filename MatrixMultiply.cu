#include <cuda_runtime_api.h>
#include <cuda/cmath>

#include <cstdio>
#include <chrono>

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

struct Matrix
{
    float* data;
    dim3 dims;
    dim3 blocks;
};

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
    __shared__ float blockA[32][32];
    __shared__ float blockB[32][33];
    __shared__ float blockC[32][32];

    dim3 mappedBlockA = { blockIdx.y % blocksA.x, blockIdx.y / blocksA.x };
    dim3 mappedBlockB = { blockIdx.x, mappedBlockA.x };
    
    dim3 blockOffsetA = { mappedBlockA.x * 32, mappedBlockA.y * 32 };
    dim3 threadOffsetA = { blockOffsetA.x + threadIdx.x, blockOffsetA.y + threadIdx.y };
    dim3 blockOffsetB = { mappedBlockB.x * 32, mappedBlockB.y * 32 };
    dim3 threadOffsetB = { blockOffsetB.x + threadIdx.x, blockOffsetB.y + threadIdx.y };

    blockA[threadIdx.y][threadIdx.x] = matA[threadOffsetA.y * dimsA.x + threadOffsetA.x];
    blockB[threadIdx.y][threadIdx.x] = matB[threadOffsetB.y * dimsB.x + threadOffsetB.x];

    __syncthreads();

    blockC[threadIdx.y][threadIdx.x] = 0;
    for (int idx = 0; idx < 32; ++idx)
    {
        blockC[threadIdx.y][threadIdx.x] += blockA[threadIdx.y][idx] * blockB[idx][threadIdx.x];
    }

    dim3 mappedBlockC = { blockIdx.x, blockIdx.y / blocksB.y };
    dim3 blockOffsetC = { mappedBlockC.x * 32, mappedBlockC.y * 32 };
    dim3 threadOffsetC = { blockOffsetC.x + threadIdx.x, blockOffsetC.y + threadIdx.y, 0 };

    atomicAdd(&matC[threadOffsetC.y * dimsC.x + threadOffsetC.x], blockC[threadIdx.y][threadIdx.x]);
}

void computeMappings(
    const Matrix* a,
    const Matrix* b,
    int *aMapping,
    int *bMapping
)
{
    int cIdx = 0;
    for (int row = 0; row < a->blocks.y; ++row)
    {
        for (int col = 0; col < a->blocks.x; ++col)
        {
            int aIdx = row * a->blocks.x + col;

            // The incoming `col` becomes the row that we look at in in matrix `b`.
            int bBase= col * b->blocks.x;
            for (int i = 0; i < b->blocks.x; ++i)
            {
                int bIdx = bBase + i;
                aMapping[cIdx] = aIdx;
                bMapping[cIdx] = bIdx;
                cIdx++;
            }
        }
    }
}

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
    constexpr dim3 blockDims{ 32, 32 };
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