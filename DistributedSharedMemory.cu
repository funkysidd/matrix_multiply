#include "Macros.h"

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <iostream>

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(int *bins, const int nbins, const int binsPerBlock, const int *__restrict__ input,
                                   const size_t arraySize)
{
    // NOTE: Every thred block still has its own shared memory block
    extern __shared__ int smem[];                            // NOTE: Should be sized binsPerBlock
    int tid = cooperative_groups::this_grid().thread_rank(); // NOTE: This is the global thread ID in the block.
    // Is `tid` same as below? int tid = blockId.x * blockDim.x + threadId.x;

    // Cluster initialization, size and calculating local bin offsets.
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank(); // NOTE: i.e., the rank within the current cluster
    int clusterSize = cluster.dim_blocks().x;             // NOTE: i.e., the the number of thread blocks

    // `threadsPerBlock` could potentially be greater than `binsPerBlock`. The `i < binsPerBlock` check ensures that
    // only selective threads update the `smem` array.
    for (int i = threadIdx.x; i < binsPerBlock; i += blockDim.x)
    {
        smem[i] = 0; // Initialize shared memory histogram to zeros
    }

    // Cluster synchronization ensures that shared memory is initialized to zero in all thread blocks in the cluster. It
    // also ensures that all thread blocks have started executing and they exist concurrently.
    cluster.sync();

    for (int i = tid; i < arraySize; i += blockDim.x * gridDim.x)
    {
        int data = input[i];

        // Find the right histogram bin.
        int binId = data;
        if (data < 0)
        {
            binId = 0;
        }
        else if (data >= nbins)
        {
            binId = nbins - 1;
        }

        // NOTE: The total number of bins are now distributed across the cluster. So every `binId` should be found
        // across the thread blocks in the cluster.

        // Find destination block rank and offset for computing distributed shared memory histogram.
        int dstBlockRank = (int)(binId / binsPerBlock);
        int dstOffset = binId % binsPerBlock;

        // Pointer to target block shared memory
        int *dstSmem = cluster.map_shared_rank(smem, dstBlockRank);

        // Perform atomic update of the histogram bin
        atomicAdd(dstSmem + dstOffset, 1);
    }

    // Cluster synchronization is required to ensure all distributed shared memory operations are completed and no
    // thread block exits while other thread blocks are still accessing distributed shared memory.
    cluster.sync();

    // Perform global memory histogram, using the local distributed memory histogram NOTE: This essentially copies local
    // (cluster specific) histogram into the global histogram into the global histogram.
    int *lbins = bins + cluster.block_rank() * binsPerBlock;
    for (int i = threadIdx.x; i < binsPerBlock; i += blockDim.x)
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
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    constexpr int threadsPerBlock = 32;
    constexpr int arraySize = 1 << 12;
    constexpr int numBins = arraySize; // Every array element is in its own bin

    // Cluster size depends on the histogram size. `clusterSize == 1` implies no distributed shared memory, just
    // thread block local shared memory
    constexpr int clusterSize = 2;
    constexpr int binsPerBlock = numBins / clusterSize;

    int *bins = nullptr;
    int *input = nullptr;
    CUDA_CHECK(cudaMallocManaged(&bins, arraySize * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&input, arraySize * sizeof(int)));

    // Initialize arrays
    for (int i = 0; i < arraySize; ++i)
    {
        bins[i] = 0;
        input[i] = i;
    }

    // Launch via extensible launch
    {
        cudaLaunchConfig_t config = { 0 };
        config.gridDim = arraySize / threadsPerBlock;
        config.blockDim = threadsPerBlock;
        config.dynamicSmemBytes = binsPerBlock * sizeof(int); // Dynamic shared memory size is per block. Distributed
                                                              // shared memory size =  clusterSize * binsPerBlock *
                                                              // sizeof(int).

        CUDA_CHECK(cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        config.dynamicSmemBytes));

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = clusterSize;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;

        CUDA_CHECK(cudaLaunchKernelEx(&config, clusterHist_kernel, bins, numBins, binsPerBlock, input, arraySize));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Check if every bin count is 1
    bool success = true;
    for (int i = 0; i < arraySize; ++i)
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