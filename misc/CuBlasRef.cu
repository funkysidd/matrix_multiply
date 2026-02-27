#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
 * A stand-alone script to invoke & benchmark standard cuBLAS SGEMM performance
 */

#define CUDA_CHECK(expr_to_check)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t result = expr_to_check;                                                                            \
        if (result != cudaSuccess)                                                                                     \
        {                                                                                                              \
            fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__, __LINE__, result,                         \
                    cudaGetErrorString(result));                                                                       \
        }                                                                                                              \
    } while (0)

int main(int argc, char *argv[])
{
    int m = 4096;
    int k = 4096;
    int n = 4096;
    int print = 0;
    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context

    int i, j;

    float *a, *b, *c;

    // malloc for a,b,c...
    a = (float *)malloc(m * k * sizeof(float));
    b = (float *)malloc(k * n * sizeof(float));
    c = (float *)malloc(m * n * sizeof(float));

    int ind = 11;
    for (j = 0; j < m * k; j++)
    {
        a[j] = (float)ind++;
    }

    ind = 11;
    for (j = 0; j < k * n; j++)
    {
        b[j] = (float)ind++;
    }

    ind = 11;
    for (j = 0; j < m * n; j++)
    {
        c[j] = (float)ind++;
    }

    // DEVICE
    float *d_a, *d_b, *d_c;

    // cudaMalloc for d_a, d_b, d_c...
    cudaMalloc((void **)&d_a, m * k * sizeof(float));
    cudaMalloc((void **)&d_b, k * n * sizeof(float));
    cudaMalloc((void **)&d_c, m * n * sizeof(float));

    stat = cublasCreate(&handle); // initialize CUBLAS context

    cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, m * n * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.5f;

    if (print == 1)
    {
        printf("alpha = %4.0f, beta = %4.0f\n", alpha, beta);
        printf("A = (mxk: %d x %d)\n", m, k);
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < k; j++)
            {
                printf("%4.1f ", a[i * m + j]);
            }
            printf("\n");
        }
        printf("B = (kxn: %d x %d)\n", k, n);
        for (i = 0; i < k; i++)
        {
            for (j = 0; j < n; j++)
            {
                printf("%4.1f ", b[i * n + j]);
            }
            printf("\n");
        }
        printf("C = (mxn: %d x %d)\n", m, n);
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                printf("%4.1f ", c[i * n + j]);
            }
            printf("\n");
        }
    }

    for (int i = 0; i < 50; ++i)
    {
        cudaEvent_t startEvent;
        cudaEvent_t endEvent;

        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&endEvent));

        CUDA_CHECK(cudaEventRecord(startEvent));
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_b, n, d_a, k, &beta, d_c, n);
        CUDA_CHECK(cudaEventRecord(endEvent));
        CUDA_CHECK(cudaEventSynchronize(endEvent));

        float elapsedTimeMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTimeMs, startEvent, endEvent));

        float totalOps = 2.0f * m * n * k;
        float flops = totalOps / (elapsedTimeMs / 1000.f);
        fprintf(stdout, "Duration (ms): %0.4f, perf (GFLOPS): %0.4f\n", elapsedTimeMs, flops / 1.e9);

        cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

        CUDA_CHECK(cudaEventDestroy(endEvent));
        CUDA_CHECK(cudaEventDestroy(startEvent));
    }

    if (print == 1)
    {
        printf("\nC after SGEMM = \n");
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                printf("%4.1f ", c[i * n + j]);
            }
            printf("\n");
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle); // destroy CUBLAS context
    free(a);
    free(b);
    free(c);

    return EXIT_SUCCESS;
}