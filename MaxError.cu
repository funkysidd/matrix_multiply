#include "Macros.h"

#include <cuda_runtime_api.h>
#include <cuda/cmath>

#include <algorithm>
#include <iostream>
 
// Kernel function to add the elements of two arrays
__global__ void add(float *x, float *y, int numOffsets)
{
  extern __shared__ float offsets[];

  if (threadIdx.x == 0)
  {
    for (int i = 0; i < numOffsets; ++i)
    {
      offsets[i] = i;
    }
  }

  __syncthreads();

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  y[idx] = x[idx] + y[idx] + offsets[idx % numOffsets];
}

void printPoperties(const cudaDeviceProp& props)
{
  std::cout << "Name:                " << props.name << std::endl;
  std::cout << "Threads/block:       " << props.maxThreadsPerBlock << std::endl;
  std::cout << "Threads/SM:          " << props.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "Shared Memory/block: " << props.sharedMemPerBlock << std::endl;
  std::cout << "Registers/SM:        " << props.regsPerMultiprocessor << std::endl;
  std::cout << "Registers/block:     " << props.regsPerBlock << std::endl;
}
 
int main()
{
  int cudaDeviceCount = -1;
  CUDA_CHECK(cudaGetDeviceCount(&cudaDeviceCount));

  if (cudaDeviceCount < 1)
  {
    std::cout << "No CUDA device found, exiting!" << std::endl;
    return 1;
  }

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);

  printPoperties(props);

  CUDA_CHECK(cudaSetDevice(0));

  float *x, *y;
  constexpr int numElements = 1 << 20;
  const int numOffsets = 12000;
  constexpr int blockSize = 256;
  constexpr int gridSize = cuda::ceil_div(numElements, 256);

  std::cout << "Grid size: " << gridSize << std::endl;
  
  // Allocate Unified Memory – accessible from CPU or GPU
  CUDA_CHECK(cudaMallocManaged(&x, numElements*sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&y, numElements*sizeof(float)));
  
  // initialize x and y arrays on the host
  for (int i = 0; i < numElements; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  
  // Run kernel on 1M elements on the GPU
  add<<<gridSize, blockSize, numOffsets*sizeof(float)>>>(x, y, numOffsets);

  // Check for error state after kernel launch
  CUDA_CHECK(cudaGetLastError());
  
  // Wait for GPU to finish before accessing on host
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Check for errors (all values should be 3.0f)
  float maxError = -1.0f;
  std::for_each(y, y + numElements, [&](const float& val) {
      maxError = cuda::std::fmax(maxError, cuda::std::fabs(val - 3.0f));
    }
  );

  std::cout << "Max error: " << maxError << std::endl;
  
  // Free memory
  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));

  return 0;
}

