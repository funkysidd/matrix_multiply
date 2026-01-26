#include <cuda_runtime_api.h>
#include <cuda/cmath>

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

// Enabling this causes the application to use `cudaEventSynchronize()`, which causes the main thread to block instead
// of busy wait. Using this path requires that the event be created with `cudaEventCreateWithFlags()`.
#define EVENT_SYNCHRONIZE 0

#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)

// Kernel function to add the elements of two arrays
__global__ void add(float* x, float* y)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    y[idx] = x[idx] + y[idx];
}

__global__ void scale(float* y, float scale)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    y[idx] *= scale;
}

void copyDataWhenReady(float* t_host, float* y_device, int numElements, cudaEvent_t event, cudaStream_t stream)
{
#if !EVENT_SYNCHRONIZE
    while (true)
    {
        if (cudaEventQuery(event) == cudaSuccess)
        {
            // nvtx3::scoped_range r{std::string("MaxError")};
            cudaMemcpyAsync(t_host, y_device, numElements * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
            break;
        }
    }
#else
    cudaEventSynchronize(event);
    cudaMemcpyAsync(t_host, y_device, numElements * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
#endif
}

int main()
{
    int cudaDeviceCount = -1;
    CUDA_CHECK(cudaGetDeviceCount(&cudaDeviceCount));

    if (cudaDeviceCount < 1)
    {
        std::cout << "No CUDA device found, exiting!" << std::endl;
        return 0;
    }

    CUDA_CHECK(cudaSetDevice(0));

    constexpr int numElements = 1 << 24; // i.e, equivalent of a 4096x4096 matrix
    constexpr int blockSize = 32 * 32;
    constexpr int gridSize = cuda::ceil_div(numElements, blockSize);

    float* x_device = nullptr;
    float* y_device = nullptr;
    float* t_host = nullptr;

    CUDA_CHECK(cudaMalloc(&x_device, numElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_device, numElements * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&t_host, numElements * sizeof(float)));

    {
        std::vector<float> x_host_tmp(numElements, 1.0f);
        std::vector<float> y_host_tmp(numElements, 2.0f);
        cudaMemcpy(x_device, x_host_tmp.data(), numElements * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(y_device, y_host_tmp.data(), numElements * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
    }

    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaEvent_t event;

    // Create primitives
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
#if !EVENT_SYNCHRONIZE
    CUDA_CHECK(cudaEventCreate(&event));
#else
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventBlockingSync));
#endif

    add<<<gridSize, blockSize, 0, stream1>>>(x_device, y_device);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(event, stream1));

    // `scale` and `copyDataWhenReady` should ideally overlap. In this specific example, the choice of `numElements`
    // dictates the overlap. Smaller value for `numElements` doesn't results in the expected overlap.
    scale<<<gridSize, blockSize, 0, stream1>>>(x_device, 0.0);
    CUDA_CHECK(cudaGetLastError());
    copyDataWhenReady(t_host, y_device, numElements, event, stream2);

    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Detroy primitives
    CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    // Check for errors (all values should be 3.0f)
    float maxError = -1.0f;
    std::for_each(t_host, t_host + numElements, [&](const float& val) {
        maxError = cuda::std::fmax(maxError, cuda::std::fabs(val - 3.0f));
        }
    );

    std::cout << "Max error: " << maxError << std::endl;

    CUDA_CHECK(cudaFree(x_device));
    CUDA_CHECK(cudaFree(y_device));
    CUDA_CHECK(cudaFreeHost(t_host));
}