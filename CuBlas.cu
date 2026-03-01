#include "ArgsParser.h"
#include "MatrixMultiplyRunner.h"

#include <cublas_v2.h>

// Matrix A : M x K
// Matrix B : K x N

// Block parameters
#define BM 128 // rows
#define BN 128 // cols
#define BK 8   // common dimension
#define TM 8   // rows/thread
#define TN 8   // cols/thread

// Block size in A : BM x BK
// Block size in B : BK x BN

class Runner : public MatrixMultiplyRunner
{
  public:
    Runner(cublasHandle_t handle) : MatrixMultiplyRunner(), m_handle(handle) {}

  protected:
    void InitThreadBlockDimensions() override { m_threadBlockDims = { BN, BM }; }

    void InitNumberOfThreads() override { m_threads = { (m_threadBlockDims.x * m_threadBlockDims.y) / (TM * TN) }; }

    void LaunchKernel() override
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        cublasSgemm(m_handle, CUBLAS_OP_N, CUBLAS_OP_N, N(), M(), K(), &alpha, m_matBDevice, N(), m_matADevice, K(),
                    &beta, m_matCDevice, N());
    }

  private:
    cublasHandle_t m_handle;
};

int main(int argc, char *argv[])
{
    COMMON_APP_SETUP()

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "Failed to create CUBLAS handle!");
        return 1;
    }

    const std::array<uint, 3> &dims = parser.GetDimenions();

    float perfGFlops = 0.0f;
    float maxError = 0.0f;

    Runner runner(handle);
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

    cublasDestroy(handle);

    return 0;
}