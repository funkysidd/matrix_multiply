#include <cooperative_groups.h>

#include <cstdio>
#include <iostream>

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

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input,
                                   size_t array_size)
{
  // NOTE: Every thred block still has its own shared memory block
  extern __shared__ int smem[]; // NOTE: Should be sized bins_per_block
  int tid = cooperative_groups::this_grid().thread_rank(); // NOTE: This is the global thread ID in the block.
  // Is `tid` same as below?
  // int tid = blockId.x * blockDim.x + threadId.x;

  // Cluster initialization, size and calculating local bin offsets.
  cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank(); // NOTE: i.e., the rank within the current cluster
  int cluster_size = cluster.dim_blocks().x; // NOTE: i.e., the the number of thread blocks

  // `threads_per_block` could potentially be greater than 
  // `bins_per_block`. The `i < bins_per_block` check ensures 
  // that only selective threads update the `smem` array.
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    smem[i] = 0; // Initialize shared memory histogram to zeros
  }

  // Cluster synchronization ensures that shared memory is 
  // initialized to zero in all thread blocks in the cluster. 
  // It also ensures that all thread blocks have started 
  // executing and they exist concurrently.
  cluster.sync();

  for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
  {
    int data = input[i];

    // Find the right histogram bin.
    int bin_id = data;
    if (data < 0)
    {
      bin_id = 0;
    }
    else if (data >= nbins)
    {
      bin_id = nbins - 1;
    }
      
    // NOTE: The total number of bins are now distributed across
    // the cluster. So every `binid` should be found across the 
    // thread blocks in the cluster.

    // Find destination block rank and offset for computing
    // distributed shared memory histogram
    int dst_block_rank = (int)(bin_id / bins_per_block);
    int dst_offset = bin_id % bins_per_block;

    // Pointer to target block shared memory
    int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    // Perform atomic update of the histogram bin
    atomicAdd(dst_smem + dst_offset, 1);
  }

  // Cluster synchronization is required to ensure all 
  // distributed shared memory operations are completed and no 
  // thread block exits while other thread blocks are still 
  // accessing distributed shared memory
  cluster.sync();

  // Perform global memory histogram, using the local distributed memory histogram
  // NOTE: This essentially copies local (cluster specific)
  // histogram into the global histogram into the global
  // histogram.
  int *lbins = bins + cluster.block_rank() * bins_per_block;
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    atomicAdd(&lbins[i], smem[i]);
  }
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

    constexpr int threads_per_block = 32;
    constexpr int array_size = 1 << 12;
    constexpr int num_bins = array_size; // Every array element is in its own bin
    
    // cluster_size depends on the histogram size. // ( cluster_size == 1 ) implies no distributed shared memory, just
    // thread block local shared memory
    constexpr int cluster_size = 2; // size 2 is an example here
    constexpr int bins_per_block = num_bins / cluster_size;

    int* bins = nullptr;
    int* input = nullptr;
    CUDA_CHECK(cudaMallocManaged(&bins, array_size*sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&input, array_size*sizeof(int)));

    // Initialize arrays
    for (int i = 0; i < array_size; ++i)
    {
        bins[i]= 0;
        input[i] = i;
    }

    // Launch via extensible launch
    {
        cudaLaunchConfig_t config = {0};
        config.gridDim = array_size / threads_per_block;
        config.blockDim = threads_per_block;
        // Dynamic shared memory size is per block.
        // Distributed shared memory size =  cluster_size * bins_per_block * sizeof(int)
        config.dynamicSmemBytes = bins_per_block * sizeof(int);

        CUDA_CHECK(cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = cluster_size;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;

        CUDA_CHECK(cudaLaunchKernelEx(&config, clusterHist_kernel, bins, num_bins, bins_per_block, input, array_size));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Check if every bin count is 1
    bool success = true;
    for (int i = 0; i < array_size; ++i)
    {
        if (bins[i] != 1)
        {
            std::cout << bins[i] << " != " << i << std::endl;        
            success = false;
            break;
        }
    }

    std::cout << "Test succeeded: " << success << std::endl;

    CUDA_CHECK(cudaFree(bins));
    CUDA_CHECK(cudaFree(input));
}