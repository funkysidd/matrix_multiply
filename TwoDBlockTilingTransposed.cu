#include "MatrixMultiplyRunner.h"

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

__global__ void matMul(float *matA, dim3 dimsA, dim3 blocksA,
                       ///
                       float *matB, dim3 dimsB, dim3 blocksB,
                       ///
                       float *matC, // c = a x b
                       dim3 dimsC)
{
    /// @todo: All references to `M`, `N`, `K` need to be relaced with the actual dims of `matA` and `matB`

    __shared__ float sharedBlockA_T[BK][BM]; // Transposed, i.e., as opposed to BM x BK
    __shared__ float sharedBlockB[BK][BN];

    float threadData[TM][TN] = { 0.0f };
    float rowA[TM];
    float colB[TN];

    dim2 blockCellOffsetMatA = { 0, blockIdx.y * BM };
    dim2 blockCellOffsetMatB = { blockIdx.x * BN, 0 };
    dim2 blockCellOffsetMatC = { blockIdx.x * BN, blockIdx.y * BM };
    dim2 threadBlockOffset = { (threadIdx.x % THREADS_BLOCKS_IN_X) * TN, (threadIdx.x / THREADS_BLOCKS_IN_X) * TM };

    dim2 baseCellIdxBlockA = { threadIdx.x % BK, threadIdx.x / BK };
    dim2 baseCellIdxBlockB = { threadIdx.x % BN, threadIdx.x / BN };
    uint32_t strideBlockA = NUM_THREADS_PER_BLOCK / BK;
    uint32_t strideBlockB = NUM_THREADS_PER_BLOCK / BN;

    for (int blockIt = 0; blockIt < K; blockIt += BK)
    {
        for (uint i = 0; i < BM; i += strideBlockA)
        {
            dim2 cellIdxMatA = { blockCellOffsetMatA.x + baseCellIdxBlockA.x,
                                 blockCellOffsetMatA.y + baseCellIdxBlockA.y + i };
            sharedBlockA_T[baseCellIdxBlockA.x][baseCellIdxBlockA.y + i] = matA[cellIdxMatA.y * K + cellIdxMatA.x];
        }

        for (uint i = 0; i < BK; i += strideBlockB)
        {
            dim2 cellIdxMatB = { blockCellOffsetMatB.x + baseCellIdxBlockB.x,
                                 blockCellOffsetMatB.y + baseCellIdxBlockB.y + i };
            sharedBlockB[baseCellIdxBlockB.y + i][baseCellIdxBlockB.x] = matB[cellIdxMatB.y * N + cellIdxMatB.x];
        }

        __syncthreads();

        // Every thread computes the associated threadData
        for (int i = 0; i < BK; ++i)
        {
            for (int row = 0; row < TM; ++row)
            {
                rowA[row] = sharedBlockA_T[i][threadBlockOffset.y + row];
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

class Runner : public MatrixMultiplyRunner
{
  public:
    Runner() : MatrixMultiplyRunner() {}

  protected:
    void InitThreadBlockDimensions() override { m_threadBlockDims = { BN, BM }; }

    void InitNumberOfThreads() override { m_threads = { (m_threadBlockDims.x * m_threadBlockDims.y) / (TM * TN) }; }

    void LaunchKernel()
    {
        matMul<<<m_gridDims, m_threads>>>(m_matADevice, m_matADimsPadded, m_blocksMatA, m_matBDevice, m_matBDimsPadded,
                                          m_blocksMatC, m_matCDevice, m_matCDimsPadded);
    }
};

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

    Runner runner;
    runner.Init(M, N, K);
    runner.Run();
    runner.Destroy();

    return 0;
}