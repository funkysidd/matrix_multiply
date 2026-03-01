#include "ArgsParser.h"
#include "MatrixMultiplyRunner.h"

// Matrix A : M x K
// Matrix B : K x N

// Block dimensions
#define BM 64 // rows
#define BN 64 // cols
#define BK 8  // common dimension
#define TM 8  // rows/thread

// Block size in A : BM x BK
// Block size in B : BK x BN

__global__ void matMul(float *matA, float *matB, float *matC, const uint M, const uint N, const uint K)
{
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

class Runner : public MatrixMultiplyRunner
{
  public:
    Runner() : MatrixMultiplyRunner() {}

  protected:
    void InitThreadBlockDimensions() override { m_threadBlockDims = { BN, BM }; }

    void InitNumberOfThreads() override { m_threads = { (m_threadBlockDims.x * m_threadBlockDims.y) / TM }; }

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
