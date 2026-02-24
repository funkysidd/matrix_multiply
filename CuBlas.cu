#include "MatrixMultiplyRunner.h"
#include "ArgsParser.h"

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
    Runner(cublasHandle_t handle)
    : MatrixMultiplyRunner() 
    , m_handle(handle)
    {}

  protected:
    void InitThreadBlockDimensions() override { m_threadBlockDims = { BN, BM }; }

    void InitNumberOfThreads() override { m_threads = { (m_threadBlockDims.x * m_threadBlockDims.y) / (TM * TN) }; }

    void LaunchKernel() override
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        uint m = m_matADimsPadded.y;
        uint n = m_matBDimsPadded.x;
        uint k = m_matADimsPadded.x;

        cublasSgemm(m_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, m_matBDevice, n, m_matADevice, k, &beta, m_matCDevice, n);
    }

private:
    cublasHandle_t m_handle;
};

int main(int argc, char *argv[])
{
    ArgsParser parser(argc, argv);

    int cudaDeviceCount = -1;
    CUDA_CHECK(cudaGetDeviceCount(&cudaDeviceCount));

    if (cudaDeviceCount < 1)
    {
        fprintf(stderr, "No CUDA device found, exiting!\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "Failed to create CUBLAS handle!");
        return 1;
    } 

    const std::array<uint, 3>& dims = parser.GetDimenions();

    Runner runner(handle);
    runner.Init(dims[0], dims[1], dims[2], 1.0f, 2.0f);
    for (int i = 0; i < parser.GetIterations(); ++i)
    {
        runner.Run(256.0f);
    }
    runner.Destroy();

    cublasDestroy(handle);

    return 0;
}