#include <stdio.h>
#include <cuda_runtime.h>

int getCoresPerSM(int major, int minor){
    if (major == 2) return 32;
    if (major == 3) return 192;
    if (major == 5) return 128;
    if (major == 6) {
        if (minor == 0) return 64;
        if (minor == 1 || minor == 2) return 128;
    }
    if (major == 7) return 64;
    if (major == 8) {
        if (minor == 0) return 64;
        if (minor == 6) return 128; 
    }

    return 0;
}

int main()
{
    int nd = 0;
    cudaGetDeviceCount(&nd);

    printf("Number of CUDA Devices: %d\n", nd);

    for (int d = 0; d < nd; d++)
    {
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, d);

        int coresPerSM = getCoresPerSM(dp.major, dp.minor);
        int totalCores = coresPerSM * dp.multiProcessorCount;

        printf("Device %d: %s\n", d, dp.name);

        printf("Compute Capability: %d.%d\n", dp.major, dp.minor);
        printf("Clock Rate: %d MHz\n", dp.clockRate / 1000);
        printf("Streaming Multiprocessors (SM): %d\n", dp.multiProcessorCount);
        printf("CUDA Cores: %d\n", totalCores);
        printf("Warp Size: %d\n", dp.warpSize);

     
        printf("\nGlobal Memory: %.2f GB\n", dp.totalGlobalMem / (1024.0 * 1024 * 1024));
        printf("Constant Memory: %.2f KB\n", dp.totalConstMem / 1024.0);
        printf("Shared Memory per Block: %.2f KB\n", dp.sharedMemPerBlock / 1024.0);

        printf("\nRegisters per Block: %d\n", dp.regsPerBlock);
        printf("Max Threads per Block: %d\n", dp.maxThreadsPerBlock);

        printf("\nMax Block Dimensions: %d x %d x %d\n",dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);

        printf("Max Grid Dimensions: %d x %d x %d\n", dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);
    }

    return 0;
}
