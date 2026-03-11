#include <stdio.h>
#include <cuda_runtime.h>

int getCoresPerSM(int major, int minor)
{
    // Architecture lookup table
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
        if (minor == 6) return 128; // RTX 30 series
    }

    return 0;
}

int main()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    printf("Number of CUDA Devices: %d\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        int coresPerSM = getCoresPerSM(prop.major, prop.minor);
        int totalCores = coresPerSM * prop.multiProcessorCount;

        printf("Device %d: %s\n", i, prop.name);

        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock Rate: %d MHz\n", prop.clockRate / 1000);
        printf("Streaming Multiprocessors (SM): %d\n", prop.multiProcessorCount);
        printf("CUDA Cores: %d\n", totalCores);
        printf("Warp Size: %d\n", prop.warpSize);

     
        printf("\nGlobal Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
        printf("Constant Memory: %.2f KB\n", prop.totalConstMem / 1024.0);
        printf("Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);

        printf("\nRegisters per Block: %d\n", prop.regsPerBlock);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);

        printf("\nMax Block Dimensions: %d x %d x %d\n",
            prop.maxThreadsDim[0],
            prop.maxThreadsDim[1],
            prop.maxThreadsDim[2]);

        printf("Max Grid Dimensions: %d x %d x %d\n",
            prop.maxGridSize[0],
            prop.maxGridSize[1],
            prop.maxGridSize[2]);
    }

    return 0;
}
