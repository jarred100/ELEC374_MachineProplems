#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

void initMatrix(float* A, int n)
{
    for (int i = 0; i < n * n; i++)
    {
        A[i] = (float)rand() / RAND_MAX;
    }
}

void cpuMatMul(float* P, const float* M, const float* N, int n)
{
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
            {
                sum += M[row * n + k] * N[k * n + col];
            }
            P[row * n + col] = sum;
        }
    }
}

float measureHostToDeviceTime(const float* M, const float* N, int n)
{
    float* d_M;
    float* d_N;

    size_t bytes = n * n * sizeof(float);

    cudaMalloc((void**)&d_M, bytes);
    cudaMalloc((void**)&d_N, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemcpy(d_M, M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_M);
    cudaFree(d_N);

    return ms;
}

float measureDeviceToHostTime(float* M, float* N, int n)
{
    float* d_M;
    float* d_N;

    size_t bytes = n * n * sizeof(float);

    cudaMalloc((void**)&d_M, bytes);
    cudaMalloc((void**)&d_N, bytes);

    cudaMemcpy(d_M, M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemcpy(M, d_M, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(N, d_N, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_M);
    cudaFree(d_N);

    return ms;
}

__global__ void gpuMatMulKernel(float* P, const float* M, const float* N, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
        {
            sum += M[row * n + k] * N[k * n + col];
        }
        P[row * n + col] = sum;
    }
}

void gpuMatMul(float* P, const float* M, const float* N, int n)
{
    float* d_M;
    float* d_N;
    float* d_P;

    size_t bytes = n * n * sizeof(float);

    cudaMalloc((void**)&d_M, bytes);
    cudaMalloc((void**)&d_N, bytes);
    cudaMalloc((void**)&d_P, bytes);

    cudaMemcpy(d_M, M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);

    gpuMatMulKernel << <grid, block >> > (d_P, d_M, d_N, n);
    cudaDeviceSynchronize();

    cudaMemcpy(P, d_P, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

bool compareMatrices(const float* A, const float* B, int n, float tol)
{
    for (int i = 0; i < n * n; i++)
    {
        if (fabs(A[i] - B[i]) > tol)
        {
            return false;
        }
    }
    return true;
}

int main()
{
    int n = 300;
    size_t bytes = n * n * sizeof(float);

    float* M = (float*)malloc(bytes);
    float* N = (float*)malloc(bytes);
    float* P_cpu = (float*)malloc(bytes);
    float* P_gpu = (float*)malloc(bytes);

    srand(0);

    initMatrix(M, n);
    initMatrix(N, n);

    cpuMatMul(P_cpu, M, N, n);
    gpuMatMul(P_gpu, M, N, n);

    if (compareMatrices(P_cpu, P_gpu, n, 1e-3f))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    free(M);
    free(N);
    free(P_cpu);
    free(P_gpu);

    int sizes[5] = { 300, 750, 1500, 3000, 4500 };

    printf("\nPart (a): Data Transfer Timing\n");
    printf("MatrixSize,HostToDevice_ms,DeviceToHost_ms\n");

    for (int s = 0; s < 5; s++)
    {
        int size = sizes[s];
        size_t bytes = size * size * sizeof(float);

        float* A = (float*)malloc(bytes);
        float* B = (float*)malloc(bytes);

        initMatrix(A, size);
        initMatrix(B, size);

        float h2dTime = measureHostToDeviceTime(A, B, size);
        float d2hTime = measureDeviceToHostTime(A, B, size);

        printf("%d,%.6f,%.6f\n", size, h2dTime, d2hTime);

        free(A);
        free(B);
    }

    return 0;
}