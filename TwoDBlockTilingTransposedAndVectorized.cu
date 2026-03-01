#include "ArgsParser.h"
#include "MatrixMultiplyRunner.h"

#include <thread>
#include <chrono>

// Matrix A : M x K
// Matrix B : K x N

// Block parameters
#define BM 128 // rows
#define BN 128 // cols
#define BK 8   // common dimension
#define TM 8   // rows/thread
#define TN 8   // cols/thread

// static constexpr uint THREADS_BLOCKS_IN_Y = BM / TM;
static constexpr uint THREADS_BLOCKS_IN_X = BN / TN;
// static constexpr uint NUM_THREADS_PER_BLOCK = THREADS_BLOCKS_IN_Y * THREADS_BLOCKS_IN_X;

// Block size in A : BM x BK
// Block size in B : BK x BN

#define CELLS_PER_COLUMN_LOADED_BY_THREAD 4 // Cells/column loaded by a thread
static constexpr uint THREAD_LOADS_PER_ROW_BLOCK_A = BK / CELLS_PER_COLUMN_LOADED_BY_THREAD;
static constexpr uint THREAD_LOADS_PER_ROW_BLOCK_B = BN / CELLS_PER_COLUMN_LOADED_BY_THREAD;

#define OPT_WRITE_MAT_C_VECTORIZED 1 // Optimization enabling vectorized writes to matrix C

__global__ void matMul(float *matA, float *matB, float *matC, const uint M, const uint N, const uint K)
{
    __shared__ float sharedBlockA_T[BK][BM]; // Transposed, i.e., as opposed to BM x BK
    __shared__ float sharedBlockB[BK][BN];

    float threadData[TM][TN] = { 0.0f };
    float rowA[TM];
    float colB[TN];

    dim2 blockCellOffsetMatA = { 0, blockIdx.y * BM };
    dim2 blockCellOffsetMatB = { blockIdx.x * BN, 0 };
    dim2 blockCellOffsetMatC = { blockIdx.x * BN, blockIdx.y * BM };
    dim2 threadBlockOffset = { (threadIdx.x % THREADS_BLOCKS_IN_X) * TN, (threadIdx.x / THREADS_BLOCKS_IN_X) * TM };

    dim2 baseCellIdxBlockA = { (threadIdx.x % THREAD_LOADS_PER_ROW_BLOCK_A) * CELLS_PER_COLUMN_LOADED_BY_THREAD,
                               threadIdx.x / THREAD_LOADS_PER_ROW_BLOCK_A };
    dim2 baseCellIdxBlockB = { (threadIdx.x % THREAD_LOADS_PER_ROW_BLOCK_B) * CELLS_PER_COLUMN_LOADED_BY_THREAD,
                               threadIdx.x / THREAD_LOADS_PER_ROW_BLOCK_B };

    for (uint blockIt = 0; blockIt < K; blockIt += BK)
    {
        // Load `CELLS_PER_COLUMN_LOADED_BY_THREAD` (i.e., float4)  cells at baseCellIdxBlockA
        dim2 cellIdxMatA = { blockCellOffsetMatA.x + baseCellIdxBlockA.x, blockCellOffsetMatA.y + baseCellIdxBlockA.y };
        float4 dataMatA = reinterpret_cast<float4 *>(&matA[cellIdxMatA.y * K + cellIdxMatA.x])[0];
        // Note: Given we store transposed results, we need the next four lines.
        sharedBlockA_T[baseCellIdxBlockA.x + 0][baseCellIdxBlockA.y] = dataMatA.x;
        sharedBlockA_T[baseCellIdxBlockA.x + 1][baseCellIdxBlockA.y] = dataMatA.y;
        sharedBlockA_T[baseCellIdxBlockA.x + 2][baseCellIdxBlockA.y] = dataMatA.z;
        sharedBlockA_T[baseCellIdxBlockA.x + 3][baseCellIdxBlockA.y] = dataMatA.w;

        // Load `CELLS_PER_COLUMN_LOADED_BY_THREAD` (i.e., float4) cells at baseCellIdxBlockB
        dim2 cellIdxMatB = { blockCellOffsetMatB.x + baseCellIdxBlockB.x, blockCellOffsetMatB.y + baseCellIdxBlockB.y };
        reinterpret_cast<float4 *>(&sharedBlockB[baseCellIdxBlockB.y][baseCellIdxBlockB.x])[0] =
            reinterpret_cast<float4 *>(&matB[cellIdxMatB.y * N + cellIdxMatB.x])[0];

        __syncthreads();

        // Every thread computes the associated threadData
        for (uint i = 0; i < BK; ++i)
        {
            for (uint row = 0; row < TM; ++row)
            {
                rowA[row] = sharedBlockA_T[i][threadBlockOffset.y + row];
            }

            for (uint col = 0; col < TN; ++col)
            {
                colB[col] = sharedBlockB[i][threadBlockOffset.x + col];
            }

            for (uint row = 0; row < TM; ++row)
            {
                for (uint col = 0; col < TN; ++col)
                {
                    threadData[row][col] += rowA[row] * colB[col];
                }
            }
        }

        __syncthreads();

        blockCellOffsetMatA.x += BK;
        blockCellOffsetMatB.y += BK;
    }

#if OPT_WRITE_MAT_C_VECTORIZED
    for (uint row = 0; row < TM; ++row)
    {
        uint tmpY = blockCellOffsetMatC.y + threadBlockOffset.y + row;
        for (uint col = 0; col < TN; col += CELLS_PER_COLUMN_LOADED_BY_THREAD)
        {
            dim2 cellIdxMatC = { blockCellOffsetMatC.x + threadBlockOffset.x + col, tmpY };
            reinterpret_cast<float4 *>(&matC[cellIdxMatC.y * N + cellIdxMatC.x])[0] =
                reinterpret_cast<float4 *>(&threadData[row][col])[0];
        }
    }
#else
    for (uint row = 0; row < TM; ++row)
    {
        for (uint col = 0; col < TN; ++col)
        {
            dim2 cellIdxMatC = { blockCellOffsetMatC.x + threadBlockOffset.x + col,
                                 blockCellOffsetMatC.y + threadBlockOffset.y + row };
            matC[cellIdxMatC.y * N + cellIdxMatC.x] = threadData[row][col];
        }
    }
#endif
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
    static_assert(CELLS_PER_COLUMN_LOADED_BY_THREAD == sizeof(float4) / sizeof(float));

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