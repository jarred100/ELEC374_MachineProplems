#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define TOL 1e-2f

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

template <int TILE_WIDTH>
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

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
    float *d_M, *d_N, *d_P;
    int size = Width * Width * sizeof(float);

    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    int numBlocks = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 dimGrid(numBlocks, numBlocks);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    switch (TILE_WIDTH)
    {
        case 2:  MatrixMulKernel<2><<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width); break;
        case 5:  MatrixMulKernel<5><<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width); break;
        case 10: MatrixMulKernel<10><<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width); break;
        case 15: MatrixMulKernel<15><<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width); break;
        case 25: MatrixMulKernel<25><<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width); break;
    }

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main()
{
    int Width = 300;   // change this
    int TILE_WIDTH = 10; // change this

    int size = Width * Width * sizeof(float);

    float* h_M = (float*)malloc(size);
    float* h_N = (float*)malloc(size);
    float* h_P = (float*)malloc(size);
    float* h_CPU = (float*)malloc(size);

    initMatrix(h_M, Width);
    initMatrix(h_N, Width);

    matrixMultiply(h_P, h_M, h_N, Width, TILE_WIDTH);
    cpuMatMul(h_CPU, h_M, h_N, Width);

    if (checkResult(h_P, h_CPU, Width))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    free(h_M);
    free(h_N);
    free(h_P);
    free(h_CPU);

    return 0;
}
