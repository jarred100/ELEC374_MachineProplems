// Jarred Brown
// 20395573

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_TILE_HEIGHT 32
#define MAX_TILE_WIDTH 32

void initMatrix(float* A, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
        A[i] = (float)rand() / RAND_MAX;
}

__global__ void MatrixMulKernelGeneral(float* M, float* N, float* P,
    int rows_M, int cols_M, int cols_N,
    int TILE_HEIGHT, int TILE_WIDTH)
{
    __shared__ float Mds[MAX_TILE_HEIGHT][MAX_TILE_WIDTH];
    __shared__ float Nds[MAX_TILE_HEIGHT][MAX_TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = blockIdx.y * TILE_HEIGHT + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0.0f;
    int numPhases = (cols_M + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < numPhases; ph++)
    {
        int m_col = ph * TILE_WIDTH + tx;
        int n_row = ph * TILE_WIDTH + ty;

        Mds[ty][tx] = (Row < rows_M && m_col < cols_M) ? M[Row * cols_M + m_col] : 0.0f;
        Nds[ty][tx] = (n_row < cols_M && Col < cols_N) ? N[n_row * cols_N + Col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
            Pvalue += Mds[ty][k] * Nds[k][tx];

        __syncthreads();
    }

    if (Row < rows_M && Col < cols_N)
        P[Row * cols_N + Col] = Pvalue;
}

void matrixMultiplyGPU(float* h_P, float* h_M, float* h_N,
    int rows_M, int cols_M, int cols_N,
    int TILE_HEIGHT, int TILE_WIDTH)
{
    float* d_M, * d_N, * d_P;
    cudaMalloc((void**)&d_M, rows_M * cols_M * sizeof(float));
    cudaMalloc((void**)&d_N, cols_M * cols_N * sizeof(float));
    cudaMalloc((void**)&d_P, rows_M * cols_N * sizeof(float));

    cudaMemcpy(d_M, h_M, rows_M * cols_M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, cols_M * cols_N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 dimGrid((cols_N + TILE_WIDTH - 1) / TILE_WIDTH,
        (rows_M + TILE_HEIGHT - 1) / TILE_HEIGHT);

    cudaEvent_t start, stop;
    float ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    MatrixMulKernelGeneral << <dimGrid, dimBlock >> > (d_M, d_N, d_P,
        rows_M, cols_M, cols_N,
        TILE_HEIGHT, TILE_WIDTH);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("Matrix %dx%d x %dx%d, Tile %dx%d, Kernel Time: %.2f ms (%.2f s)\n",
        rows_M, cols_M, cols_M, cols_N, TILE_HEIGHT, TILE_WIDTH, ms, ms / 1000.0f);

    cudaMemcpy(h_P, d_P, rows_M * cols_N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main()
{
    int TILE_HEIGHT = 14;
    int TILE_WIDTH = 17;

    
    int rows_M1 = 600, cols_M1 = 650, cols_N1 = 700;
    float* M1 = (float*)malloc(rows_M1 * cols_M1 * sizeof(float));
    float* N1 = (float*)malloc(cols_M1 * cols_N1 * sizeof(float));
    float* P1 = (float*)malloc(rows_M1 * cols_N1 * sizeof(float));

    initMatrix(M1, rows_M1, cols_M1);
    initMatrix(N1, cols_M1, cols_N1);

   
    matrixMultiplyGPU(P1, M1, N1, rows_M1, cols_M1, cols_N1, TILE_HEIGHT, TILE_WIDTH);

    free(M1); free(N1); free(P1);

    
    int rows_M2 = 2200, cols_M2 = 2050, cols_N2 = 2100;
    float* M2 = (float*)malloc(rows_M2 * cols_M2 * sizeof(float));
    float* N2 = (float*)malloc(cols_M2 * cols_N2 * sizeof(float));
    float* P2 = (float*)malloc(rows_M2 * cols_N2 * sizeof(float));

    initMatrix(M2, rows_M2, cols_M2);
    initMatrix(N2, cols_M2, cols_N2);

    printf("\n");
    matrixMultiplyGPU(P2, M2, N2, rows_M2, cols_M2, cols_N2, TILE_HEIGHT, TILE_WIDTH);

    free(M2); free(N2); free(P2);

    return 0;
}
