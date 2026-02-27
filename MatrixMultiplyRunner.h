#pragma once

#include "DataTypes.h"
#include "Macros.h"

#include <cuda/cmath>
#include <cuda_runtime_api.h>

#include <cstdio>

class MatrixMultiplyRunner
{
  public:
    MatrixMultiplyRunner() = default;
    virtual ~MatrixMultiplyRunner() = default;

    void Init(uint m, uint n, uint k, float valA = 1.0f, float valB = 1.0f)
    {
        m_matADims = { k, m };
        m_matBDims = { n, k };

        InitThreadBlockDimensions();
        InitNumberOfThreads();

        m_blocksMatA = { cuda::ceil_div(m_matADims.x, m_threadBlockDims.x),
                         cuda::ceil_div(m_matADims.y, m_threadBlockDims.y) };
        m_blocksMatB = { cuda::ceil_div(m_matBDims.x, m_threadBlockDims.x),
                         cuda::ceil_div(m_matBDims.y, m_threadBlockDims.y) };
        m_blocksMatC = { m_blocksMatB.x, m_blocksMatA.y };

        m_gridDims = { m_blocksMatC.x, m_blocksMatC.y };

        m_matADimsPadded = MATRIX_DIMS(m_blocksMatA, m_threadBlockDims);
        m_matBDimsPadded = MATRIX_DIMS(m_blocksMatB, m_threadBlockDims);
        m_matCDimsPadded = MATRIX_DIMS(m_blocksMatC, m_threadBlockDims);

        CUDA_CHECK(cudaMallocHost(&m_matAHost, GET_SIZE(m_matADimsPadded) * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&m_matBHost, GET_SIZE(m_matBDimsPadded) * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&m_matCHost, GET_SIZE(m_matCDimsPadded) * sizeof(float)));

        ASSIGN_MATRIX(m_matAHost, m_matADimsPadded, valA);
        ASSIGN_MATRIX(m_matBHost, m_matBDimsPadded, valB);

        CUDA_CHECK(cudaMalloc(&m_matADevice, GET_SIZE(m_matADimsPadded) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_matBDevice, GET_SIZE(m_matBDimsPadded) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_matCDevice, GET_SIZE(m_matCDimsPadded) * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(m_matADevice, m_matAHost, GET_SIZE(m_matADimsPadded) * sizeof(float),
                              cudaMemcpyKind::cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_matBDevice, m_matBHost, GET_SIZE(m_matBDimsPadded) * sizeof(float),
                              cudaMemcpyKind::cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(m_matCDevice, 0, GET_SIZE(m_matCDimsPadded) * sizeof(float)));

        CUDA_CHECK(cudaEventCreate(&m_startEvent));
        CUDA_CHECK(cudaEventCreate(&m_endEvent));
    }

    void Run(float expectedResult = 4096.0f)
    {
        CUDA_CHECK(cudaEventRecord(m_startEvent, 0));
        LaunchKernel();
        CUDA_CHECK(cudaEventRecord(m_endEvent, 0));
        CUDA_CHECK(cudaEventSynchronize(m_endEvent));

        float elapsedTimeMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTimeMs, m_startEvent, m_endEvent));

        float totalOps = 2.0f * m_matCDimsPadded.x * m_matCDimsPadded.y * m_matADimsPadded.x;
        float flops = totalOps / (elapsedTimeMs / 1000.f); // A giga flop is 10^9 flops

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(m_matCHost, m_matCDevice, GET_SIZE(m_matCDimsPadded) * sizeof(float),
                              cudaMemcpyKind::cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(m_matCDevice, 0, GET_SIZE(m_matCDimsPadded) * sizeof(float)));

        float maxError = 0.0f;
        VERIFY_RESULT(m_matCHost, m_matCDimsPadded, expectedResult, maxError);
        fprintf(stdout, "Duration (milliseconds): %0.4f, Performance (GFLOPS): %0.4f, Max error: %0.4f\n", elapsedTimeMs, flops / 1.e9, maxError);
    }

    void Destroy()
    {
        CUDA_CHECK(cudaEventDestroy(m_endEvent));
        CUDA_CHECK(cudaEventDestroy(m_startEvent));

        CUDA_CHECK(cudaFree(m_matCDevice));
        CUDA_CHECK(cudaFree(m_matBDevice));
        CUDA_CHECK(cudaFree(m_matADevice));

        CUDA_CHECK(cudaFreeHost(m_matCHost));
        CUDA_CHECK(cudaFreeHost(m_matBHost));
        CUDA_CHECK(cudaFreeHost(m_matAHost));
    }

  protected:
    uint M() const { return m_matADimsPadded.y; }
    uint N() const { return m_matBDimsPadded.x; }
    uint K() const { return m_matADimsPadded.x; }

    virtual void InitThreadBlockDimensions() = 0;
    virtual void InitNumberOfThreads() = 0;
    virtual void LaunchKernel() = 0;

  protected:
    cudaEvent_t m_startEvent;
    cudaEvent_t m_endEvent;

    float *m_matADevice = nullptr;
    float *m_matBDevice = nullptr;
    float *m_matCDevice = nullptr;

    float *m_matAHost = nullptr;
    float *m_matBHost = nullptr;
    float *m_matCHost = nullptr;

    dim3 m_threadBlockDims; // Thread block dimensions
    dim3 m_gridDims;        // Thread blocks in the grid
    dim3 m_threads;         // Number of threads in a thread block

    dim3 m_blocksMatA; // Blocks in matrix A
    dim3 m_blocksMatB; // Blocks in matrix B
    dim3 m_blocksMatC; // Blocks in matrix C

    dim2 m_matADims; // Initial matrix A dimensions
    dim2 m_matBDims; // Initial matrix B dimensions

    dim3 m_matADimsPadded; // Padded matrix A dimensions
    dim3 m_matBDimsPadded; // Padded matrix B dimensions
    dim3 m_matCDimsPadded; // Padded matrix C dimensions
};