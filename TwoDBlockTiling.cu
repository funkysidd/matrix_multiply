#include "ArgsParser.h"
#include "MatrixMultiplyRunner.h"

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

__global__ void matMul(float *matA, float *matB, float *matC, const uint M, const uint N, const uint K)
{
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

class Runner : public MatrixMultiplyRunner
{
  public:
    Runner() : MatrixMultiplyRunner() {}

  protected:
    void InitThreadBlockDimensions() override { m_threadBlockDims = { BN, BM }; }

    void InitNumberOfThreads() override { m_threads = { (m_threadBlockDims.x * m_threadBlockDims.y) / (TM * TN) }; }

    void LaunchKernel() { matMul<<<m_gridDims, m_threads>>>(m_matADevice, m_matBDevice, m_matCDevice, M(), N(), K()); }
};

int main(int argc, char *argv[])
{
    COMMON_APP_SETUP()

    const std::array<uint, 3> &dims = parser.GetDimenions();

    float perfGFlops = 0.0f;
    float maxError = 0.0f;

    Runner runner;
    runner.Init(dims[0], dims[1], dims[2], 1.0f, 1.0f);
    for (int i = 0; i < parser.GetIterations(); ++i)
    {
        runner.Run(parser.GetExpectedResult());
        perfGFlops = std::max(perfGFlops, runner.GetPerfInGigaFlops());
        maxError = std::max(maxError, runner.GetMaxError());
        fprintf(stdout, "Iteration %0.3d, performance (GLOPS): %0.4f\n", i, runner.GetPerfInGigaFlops());
    }
    runner.Destroy();

    fprintf(stdout, "Best performance (GFLOPS): %0.4f, Max error: %0.4f\n", perfGFlops, maxError);

    return 0;
}