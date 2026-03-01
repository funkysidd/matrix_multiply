#include "ArgsParser.h"
#include "MatrixMultiplyRunner.h"

#define BLOCKSIZE 32

__global__ void matMul(float *matA, float *matB, float *matC, const uint M, const uint N, const uint K)
{
    __shared__ float sharedBlockA[BLOCKSIZE][BLOCKSIZE];
    __shared__ float sharedBlockB[BLOCKSIZE][BLOCKSIZE];

    float threadResult = { 0.0f };

    dim2 blockCellOffsetMatA = { 0, blockIdx.y * BLOCKSIZE };
    dim2 blockCellOffsetMatB = { blockIdx.x * BLOCKSIZE, 0 };
    dim2 blockCellOffsetMatC = { blockIdx.x * BLOCKSIZE, blockIdx.y * BLOCKSIZE };

    for (int blockIt = 0; blockIt < K; blockIt += BLOCKSIZE)
    {
        dim2 cellIdxMatA = { blockCellOffsetMatA.x + threadIdx.x, blockCellOffsetMatA.y + threadIdx.y };
        dim2 cellIdxMatB = { blockCellOffsetMatB.x + threadIdx.x, blockCellOffsetMatB.y + threadIdx.y };

        sharedBlockA[threadIdx.y][threadIdx.x] = matA[cellIdxMatA.y * K + cellIdxMatA.x];
        sharedBlockB[threadIdx.y][threadIdx.x] = matB[cellIdxMatB.y * N + cellIdxMatB.x];

        __syncthreads();

        for (int idx = 0; idx < BLOCKSIZE; ++idx)
        {
            threadResult += sharedBlockA[threadIdx.y][idx] * sharedBlockB[idx][threadIdx.x];
        }

        __syncthreads();

        blockCellOffsetMatA.x += BLOCKSIZE;
        blockCellOffsetMatB.y += BLOCKSIZE;
    }

    dim2 cellIdxMatC = { blockCellOffsetMatC.x + threadIdx.x, blockCellOffsetMatC.y + threadIdx.y };
    matC[cellIdxMatC.y * N + cellIdxMatC.x] = threadResult;
}

class Runner : public MatrixMultiplyRunner
{
  public:
    Runner() : MatrixMultiplyRunner() {}

  protected:
    void InitThreadBlockDimensions() override { m_threadBlockDims = { BLOCKSIZE, BLOCKSIZE }; }

    void InitNumberOfThreads() override { m_threads = { BLOCKSIZE, BLOCKSIZE }; }

    void LaunchKernel()
    {
        matMul<<<m_gridDims, m_threads>>>(m_matADevice, m_matBDevice, m_matCDevice, M(), N(), K());
    }
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