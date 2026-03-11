#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>

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

__global__ void gpuMatMulSingleThreadKernel(float* P, const float* M, const float* N, int n)
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

double measureCpuTime(const float* M, const float* N, float* P, int n)
{
    auto start = std::chrono::high_resolution_clock::now();

    cpuMatMul(P, M, N, n);

    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = stop - start;

    return elapsed.count();
}

float measureGpuKernelTime(const float* M, const float* N, int n, int blockWidth, bool singleThread)
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

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    if (singleThread)
    {
        gpuMatMulSingleThreadKernel << <1, 1 >> > (d_P, d_M, d_N, n);
    }
    else
    {
        dim3 block(blockWidth, blockWidth);

        dim3 grid(
            (n + blockWidth - 1) / blockWidth,
            (n + blockWidth - 1) / blockWidth
        );

        gpuMatMulKernel << <grid, block >> > (d_P, d_M, d_N, n);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;

    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return ms;
}

int main()
{
    srand(0);

    int n = 300;
    size_t bytes = n * n * sizeof(float);

    float* M = (float*)malloc(bytes);
    float* N = (float*)malloc(bytes);
    float* P_cpu = (float*)malloc(bytes);
    float* P_gpu = (float*)malloc(bytes);

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

    int sizesA[5] = { 300, 750, 1500, 3000, 4500 };

    printf("\nPart (a):\n");
    printf("Matrix Size, Host To Device (ms), Device To Host (ms)\n");

    for (int i = 0; i < 5; i++)
    {
        int size = sizesA[i];
        size_t bytesA = size * size * sizeof(float);

        float* A = (float*)malloc(bytesA);
        float* B = (float*)malloc(bytesA);

        initMatrix(A, size);
        initMatrix(B, size);

        float h2d = measureHostToDeviceTime(A, B, size);
        float d2h = measureDeviceToHostTime(A, B, size);

        printf("%d, %.6f, %.6f\n", size, h2d, d2h);

        free(A);
        free(B);
    }

    int sizesB[2] = { 300, 750 };

    printf("\nPart (b):\n");
    printf("Matrix Size, CPU (ms), GPU 1 block 1 thread (ms), GPU block width 1 (ms)\n");

    for (int i = 0; i < 2; i++)
    {
        int size = sizesB[i];
        size_t bytesB = size * size * sizeof(float);

        float* A = (float*)malloc(bytesB);
        float* B = (float*)malloc(bytesB);
        float* P = (float*)malloc(bytesB);

        initMatrix(A, size);
        initMatrix(B, size);

        double cpuTime = measureCpuTime(A, B, P, size);
        float gpu1 = measureGpuKernelTime(A, B, size, 1, true);
        float gpu2 = measureGpuKernelTime(A, B, size, 1, false);

        printf("%d, %.6f, %.6f, %.6f\n", size, cpuTime, gpu1, gpu2);

        free(A);
        free(B);
        free(P);
    }

    int sizesC[5] = { 300, 750, 1500, 3000, 4500 };
    int blockWidths[5] = { 2, 5, 10, 15, 25 };

    printf("\nPart (c):\n");
    printf("Matrix Size, Block Width, Kernel Time (ms)\n");

    for (int i = 0; i < 5; i++)
    {
        int size = sizesC[i];
        size_t bytesC = size * size * sizeof(float);

        float* A = (float*)malloc(bytesC);
        float* B = (float*)malloc(bytesC);

        initMatrix(A, size);
        initMatrix(B, size);

        for (int j = 0; j < 5; j++)
        {
            int bw = blockWidths[j];

            float time = measureGpuKernelTime(A, B, size, bw, false);

            printf("%d, %d, %.6f\n", size, bw, time);
        }

        free(A);
        free(B);
    }

    return 0;
}
