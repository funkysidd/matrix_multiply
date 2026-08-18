#include "ArgsParser.h"
#include "MatrixMultiplyRunner.h"

#define BLOCKSIZE 32

__global__ void matMul(float *matA, float *matB, float *matC, const uint M, const uint N, const uint K)
{
#if 0
    const int x = blockIdx.x * BLOCKSIZE + threadIdx.x;
    const int y = blockIdx.y * BLOCKSIZE + threadIdx.y;

    float tmp = 0.0;
    for (int i = 0; i < K; ++i)
    {
        tmp += matA[y * K + i] * matB[i * N + x];
    }
    matC[y * N + x] = tmp;
#else
    // NOTE: Even though thread access patterns within a block are similar, swapping threadIdx.x/y
    // results in 1.34x better performance.
    // Claude based analysis indicates the long scoreboard (i.e., memory operations that went out 
    // to L1/L2/DRAM, as opposed to shared memoory) stalls (measured in cycles) are higher in the 
    // inactive code snippet.
    const int x = blockIdx.x * BLOCKSIZE + threadIdx.y;
    const int y = blockIdx.y * BLOCKSIZE + threadIdx.x;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += matA[x * K + i] * matB[i * N + y];
        }
        matC[x * N + y] = tmp;
    }
#endif
}

class Runner : public MatrixMultiplyRunner
{
  public:
    Runner() : MatrixMultiplyRunner() {}

  protected:
    void InitThreadBlockDimensions() override { m_threadBlockDims = { BLOCKSIZE, BLOCKSIZE }; }

    void InitNumberOfThreads() override { m_threads = { BLOCKSIZE, BLOCKSIZE }; }

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