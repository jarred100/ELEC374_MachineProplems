#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>

#define DEFAULT_BLOCK_WIDTH 16

void initMatrix(float* A, int Width)
{
    for (int i = 0; i < Width * Width; i++)
    {
        A[i] = (float)rand() / RAND_MAX;
    }
}

void cpuMatrixMul(float* M, float* N, float* P, int Width)
{
    for (int Row = 0; Row < Width; Row++)
    {
        for (int Col = 0; Col < Width; Col++)
        {
            float Pvalue = 0.0f;

            for (int k = 0; k < Width; k++)
            {
                Pvalue += M[Row * Width + k] * N[k * Width + Col];
            }

            P[Row * Width + Col] = Pvalue;
        }
    }
}

bool compareMatrices(const float* A, const float* B, int Width, float tol)
{
    for (int i = 0; i < Width * Width; i++)
    {
        if (fabs(A[i] - B[i]) > tol)
        {
            return false;
        }
    }
    return true;
}

// One thread computes one output element
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < Width && Col < Width)
    {
        float Pvalue = 0.0f;

        for (int k = 0; k < Width; k++)
        {
            Pvalue += M[Row * Width + k] * N[k * Width + Col];
        }

        P[Row * Width + Col] = Pvalue;
    }
}

// Single block, single thread version for part (b)
__global__ void MatrixMulSingleThreadKernel(float* M, float* N, float* P, int Width)
{
    for (int Row = 0; Row < Width; Row++)
    {
        for (int Col = 0; Col < Width; Col++)
        {
            float Pvalue = 0.0f;

            for (int k = 0; k < Width; k++)
            {
                Pvalue += M[Row * Width + k] * N[k * Width + Col];
            }

            P[Row * Width + Col] = Pvalue;
        }
    }
}

float measureHostToDeviceTime(const float* h_M, const float* h_N, int Width)
{
    float* d_M = 0;
    float* d_N = 0;

    int nbytes = Width * Width * sizeof(float);

    cudaMalloc((void**)&d_M, nbytes);
    cudaMalloc((void**)&d_N, nbytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();

    float gpu_time = 0.0f;

    cudaEventRecord(start, 0);

    cudaMemcpy(d_M, h_M, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, nbytes, cudaMemcpyHostToDevice);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_M);
    cudaFree(d_N);

    return gpu_time;
}

float measureDeviceToHostTime(float* h_M, float* h_N, int Width)
{
    float* d_M = 0;
    float* d_N = 0;

    int nbytes = Width * Width * sizeof(float);

    cudaMalloc((void**)&d_M, nbytes);
    cudaMalloc((void**)&d_N, nbytes);

    cudaMemcpy(d_M, h_M, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, nbytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();

    float gpu_time = 0.0f;

    cudaEventRecord(start, 0);

    cudaMemcpy(h_M, d_M, nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_N, d_N, nbytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_M);
    cudaFree(d_N);

    return gpu_time;
}

double measureCpuTime(float* h_M, float* h_N, float* h_P, int Width)
{
    auto start = std::chrono::high_resolution_clock::now();

    cpuMatrixMul(h_M, h_N, h_P, Width);

    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = stop - start;
    return elapsed.count();
}

float measureGpuKernelTime(const float* h_M, const float* h_N, int Width, int blockWidth, int singleThread)
{
    float* d_M = 0;
    float* d_N = 0;
    float* d_P = 0;

    int nbytes = Width * Width * sizeof(float);

    cudaMalloc((void**)&d_M, nbytes);
    cudaMalloc((void**)&d_N, nbytes);
    cudaMalloc((void**)&d_P, nbytes);

    cudaMemcpy(d_M, h_M, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, nbytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();

    float gpu_time = 0.0f;

    cudaEventRecord(start, 0);

    if (singleThread)
    {
        MatrixMulSingleThreadKernel << <1, 1 >> > (d_M, d_N, d_P, Width);
    }
    else
    {
        int NumBlocks = (Width + blockWidth - 1) / blockWidth;

        dim3 threads(blockWidth, blockWidth);
        dim3 blocks(NumBlocks, NumBlocks);

        MatrixMulKernel << <blocks, threads >> > (d_M, d_N, d_P, Width);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return gpu_time;
}

int main()
{
    srand(0);

    // Correctness test
    int Width = 300;
    size_t size = Width * Width * sizeof(float);

    float* M = (float*)malloc(size);
    float* N = (float*)malloc(size);
    float* P_cpu = (float*)malloc(size);
    float* P_gpu = (float*)malloc(size);

    initMatrix(M, Width);
    initMatrix(N, Width);

    cpuMatrixMul(M, N, P_cpu, Width);

    float* d_M;
    float* d_N;
    float* d_P;

    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);

    int NumBlocks = (Width + DEFAULT_BLOCK_WIDTH - 1) / DEFAULT_BLOCK_WIDTH;
    dim3 dimGrid(NumBlocks, NumBlocks);
    dim3 dimBlock(DEFAULT_BLOCK_WIDTH, DEFAULT_BLOCK_WIDTH);

    MatrixMulKernel << <dimGrid, dimBlock >> > (d_M, d_N, d_P, Width);
    cudaDeviceSynchronize();

    cudaMemcpy(P_gpu, d_P, size, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    if (compareMatrices(P_cpu, P_gpu, Width, 1e-3f))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    free(M);
    free(N);
    free(P_cpu);
    free(P_gpu);

    // Part (a)
    int sizesA[5] = { 300, 750, 1500, 3000, 4500 };

    printf("\nPart (a):\n");
    printf("Matrix Size, Host To Device (ms), Device To Host (ms)\n");

    for (int i = 0; i < 5; i++)
    {
        int WidthA = sizesA[i];
        size_t sizeA = WidthA * WidthA * sizeof(float);

        float* A = (float*)malloc(sizeA);
        float* B = (float*)malloc(sizeA);

        initMatrix(A, WidthA);
        initMatrix(B, WidthA);

        float h2d = measureHostToDeviceTime(A, B, WidthA);
        float d2h = measureDeviceToHostTime(A, B, WidthA);

        printf("%d, %.6f, %.6f\n", WidthA, h2d, d2h);

        free(A);
        free(B);
    }

    // Part (b)
    int sizesB[2] = { 300, 750 };

    printf("\nPart (b):\n");
    printf("Matrix Size, CPU (ms), GPU 1 block 1 thread (ms), GPU block width 1 (ms)\n");

    for (int i = 0; i < 2; i++)
    {
        int WidthB = sizesB[i];
        size_t sizeB = WidthB * WidthB * sizeof(float);

        float* A = (float*)malloc(sizeB);
        float* B = (float*)malloc(sizeB);
        float* P = (float*)malloc(sizeB);

        initMatrix(A, WidthB);
        initMatrix(B, WidthB);

        double cpuTime = measureCpuTime(A, B, P, WidthB);
        float gpu1 = measureGpuKernelTime(A, B, WidthB, 1, true);
        float gpu2 = measureGpuKernelTime(A, B, WidthB, 1, false);

        printf("%d, %.6f, %.6f, %.6f\n", WidthB, cpuTime, gpu1, gpu2);

        free(A);
        free(B);
        free(P);
    }

    // Part (c)
    int sizesC[5] = { 300, 750, 1500, 3000, 4500 };
    int blockWidths[5] = { 2, 5, 10, 15, 25 };

    printf("\nPart (c):\n");
    printf("Matrix Size, Block Width, Kernel Time (ms)\n");

    for (int i = 0; i < 5; i++)
    {
        int WidthC = sizesC[i];
        size_t sizeC = WidthC * WidthC * sizeof(float);

        float* A = (float*)malloc(sizeC);
        float* B = (float*)malloc(sizeC);

        initMatrix(A, WidthC);
        initMatrix(B, WidthC);

        for (int j = 0; j < 5; j++)
        {
            int bw = blockWidths[j];
            float time = measureGpuKernelTime(A, B, WidthC, bw, false);

            printf("%d, %d, %.6f\n", WidthC, bw, time);
        }

        free(A);
        free(B);
    }

    return 0;
}
