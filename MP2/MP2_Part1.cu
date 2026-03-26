// Jarred Brown
// 20395573

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define TOL 1e-2f
#define MAX_TILE 25   // Maximum tile size allowed

void initMatrix(float* A, int n)
{
    for (int i = 0; i < n * n; i++)
        A[i] = (float)rand() / RAND_MAX;
}

void cpuMatMul(float* P, const float* M, const float* N, int n)
{
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += M[row * n + k] * N[k * n + col];
            P[row * n + col] = sum;
        }
    }
}

int checkResult(const float* A, const float* B, int n)
{
    for (int i = 0; i < n * n; i++)
    {
        if (fabs(A[i] - B[i]) > TOL)
            return 0;
    }
    return 1;
}

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width, int TILE_WIDTH)
{
    __shared__ float Mds[MAX_TILE][MAX_TILE];
    __shared__ float Nds[MAX_TILE][MAX_TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0.0f;
    int phases = (Width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < phases; ph++)
    {
        if (Row < Width && ph * TILE_WIDTH + tx < Width)
            Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;

        if (Col < Width && ph * TILE_WIDTH + ty < Width)
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        else
            Nds[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
            Pvalue += Mds[ty][k] * Nds[k][tx];

        __syncthreads();
    }

    if (Row < Width && Col < Width)
        P[Row * Width + Col] = Pvalue;
}

void matrixMultiply(float* h_P, float* h_M, float* h_N, int Width, int TILE_WIDTH)
{
    float* d_M, * d_N, * d_P;
    int size = Width * Width * sizeof(float);

    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    int numBlocks = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 dimGrid(numBlocks, numBlocks);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    cudaEvent_t start, stop;
    float ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    MatrixMulKernel << <dimGrid, dimBlock >> > (d_M, d_N, d_P, Width, TILE_WIDTH);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    printf("%d, %d, %f\n", Width, TILE_WIDTH, ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main()
{
    int sizes[] = { 300, 750, 1500, 3000, 4500 };
    int tileWidths[] = { 2, 5, 10, 15, 25 };

    printf("Matrix Size,  Tile Width,  Kernel Time (ms)\n");

    for (int s = 0; s < 5; s++)
    {
        int Width = sizes[s];
        int size = Width * Width * sizeof(float);

        float* h_M = (float*)malloc(size);
        float* h_N = (float*)malloc(size);
        float* h_P = (float*)malloc(size);
        float* h_CPU = (float*)malloc(size);

        initMatrix(h_M, Width);
        initMatrix(h_N, Width);

        for (int t = 0; t < 5; t++)
        {
            int TILE_WIDTH = tileWidths[t];

            // GPU multiplication
            matrixMultiply(h_P, h_M, h_N, Width, TILE_WIDTH);
            
            cpuMatMul(h_CPU, h_M, h_N, Width);

            if (checkResult(h_P, h_CPU, Width))
                printf("Test PASSED\n");
            else
                printf("Test FAILED\n");
        }

        free(h_M);
        free(h_N);
        free(h_P);
        free(h_CPU);
    }

    return 0;
}
