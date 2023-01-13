#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#include "file_system.h"

constexpr auto DATAFILE = "./data.bin";
constexpr auto OUTFILE = "./snapshot.bin";

constexpr auto SUPERBLOCK_SIZE = 4096; // 32K/8 bits = 4 K
constexpr auto STORAGE_BLOCK_SIZE = 32;
constexpr auto FCB_SIZE = 32; // 32 bytes per FCB
constexpr auto FCB_ENTRIES = 1024;
constexpr auto FCB_TOTAL_SIZE = FCB_SIZE * FCB_ENTRIES;

constexpr auto MAX_FILENAME_SIZE = 20;
constexpr auto MAX_FILE_SIZE = 1024;
constexpr auto MAX_FILE_NUM = FCB_ENTRIES;
constexpr auto MAX_FILES_SIZE = MAX_FILE_SIZE * MAX_FILE_NUM;

constexpr auto FILE_BASE_ADDRESS = SUPERBLOCK_SIZE + FCB_TOTAL_SIZE; // 4096+32768
constexpr auto VOLUME_SIZE = SUPERBLOCK_SIZE + FCB_TOTAL_SIZE + MAX_FILES_SIZE; // 4096+32768+1048576


// data input and output
__device__ __managed__ uchar input[MAX_FILES_SIZE];
__device__ __managed__ uchar output[MAX_FILES_SIZE];

// volume (disk storage)
__device__ __managed__ uchar volume[VOLUME_SIZE];


__device__ void user_program(FileSystem* fs, uchar* input, uchar* output);

__global__ void mykernel(uchar* input, uchar* output) {

    // Initialize the file system
    FileSystem fs;
    fs_init(&fs, volume, SUPERBLOCK_SIZE, FCB_SIZE, FCB_ENTRIES, VOLUME_SIZE, STORAGE_BLOCK_SIZE,
            MAX_FILENAME_SIZE, MAX_FILE_NUM, MAX_FILES_SIZE, FILE_BASE_ADDRESS);

    // user program the access pattern for testing file operations
    user_program(&fs, input, output);
}

__host__ void write_binaryFile(const char* file_name, const void* buffer, const int buffer_size) {
    FILE* fp = fopen(file_name, "wb");
    fwrite(buffer, 1, buffer_size, fp);
    fclose(fp);
}

__host__ int load_binaryFile(const char* file_name, void* buffer, const int buffer_size) {
    FILE* fp = fopen(file_name, "rb");

    if (!fp) {
        printf("***Unable to open file %s***\n", file_name);
        exit(1);
    }

    // Get file length
    fseek(fp, 0, SEEK_END);
    const int file_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_len > buffer_size) {
        printf("****invalid testcase!!****\n");
        printf("****software warning: the file: %s size****\n", file_name);
        printf("****is greater than buffer size****\n");
        exit(EXIT_FAILURE);
    }

    // Read file contents into buffer
    fread(buffer, file_len, 1, fp);
    fclose(fp);
    return file_len;
}

int main() {
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    load_binaryFile(DATAFILE, input, MAX_FILES_SIZE);
    
    // Launch to GPU kernel with single thread
    mykernel<<<1, 1>>>(input, output);

    const cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "mykernel launch failed: %s\n", cudaGetErrorString(cuda_status));
        return 0;
    }

    cudaDeviceSynchronize();
    cudaDeviceReset();

    write_binaryFile(OUTFILE, output, MAX_FILES_SIZE);

    return 0;
}
