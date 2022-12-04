#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>

#include "virtual_memory.h"

constexpr auto DATAFILE = "./data.bin";
constexpr auto OUTFILE = "./snapshot.bin";

// page size is 32bytes
constexpr auto PAGE_SIZE = 1 << 5;
// 16 KB in page table
constexpr auto INVERT_PAGE_TABLE_SIZE = 1 << 14;
// 32 KB in shared memory
constexpr auto PHYSICAL_MEM_SIZE = 1 << 15;
// 128 KB in global memory
constexpr auto STORAGE_SIZE = 1 << 17;

constexpr auto VIRTUAL_PAGE_NUM = (PHYSICAL_MEM_SIZE + STORAGE_SIZE) / PAGE_SIZE;
constexpr auto PHYSICAL_PAGE_NUM = PHYSICAL_MEM_SIZE / PAGE_SIZE;

// count the page fault times
__device__ __managed__ int pagefault_num = 0;

// data input and output
__device__ __managed__ uchar results[STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];

// memory allocation for virtual_memory
// secondary memory
__device__ __managed__ uchar storage[STORAGE_SIZE];
// page table
extern __shared__ u32 pt[];
// swap table
__device__ __managed__ u32 swap_table[VIRTUAL_PAGE_NUM];
__device__ __managed__ uchar swap_buffer[PAGE_SIZE];
__device__ __managed__ uchar storage_bitmap[VIRTUAL_PAGE_NUM];

__device__ void user_program(VirtualMemory* vm, uchar* input, uchar* results, int input_size);

__global__ void mykernel(int input_size) {
    // memory allocation for virtual_memory
    // take shared memory as physical memory
    __shared__ uchar data[PHYSICAL_MEM_SIZE];

    VirtualMemory vm;
    vm_init(&vm, data, storage, pt, swap_table, storage_bitmap, swap_buffer, &pagefault_num, PAGE_SIZE,
            INVERT_PAGE_TABLE_SIZE, PHYSICAL_MEM_SIZE, STORAGE_SIZE, PHYSICAL_PAGE_NUM);

    // user program the access pattern for testing paging
    user_program(&vm, input, results, input_size);
}

__host__ void write_binary_file(char* file_name, void* buffer, int buffer_size) {
    FILE* fp = fopen(file_name, "wb");
    fwrite(buffer, 1, buffer_size, fp);
    fclose(fp);
}

__host__ int load_binary_file(char* file_name, void* buffer, int buffer_size) {
    FILE* fp = fopen(file_name, "rb");
    if (!fp) {
        const auto err_msg = std::string("Unable to open file ") + file_name;
        throw std::runtime_error(err_msg);
    }

    // Get file length
    fseek(fp, 0, SEEK_END);
    const int file_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_len > buffer_size) {
        printf("****invalid test case!!****\n");
        const auto err_msg =
            std::string("File size: ") + file_name + " is greater than buffer size.";
        throw std::runtime_error(err_msg);
    }

    // Read file contents into buffer
    fread(buffer, file_len, 1, fp);
    fclose(fp);

    return file_len;
}

int main() {
    const int input_size = load_binary_file(const_cast<char*>(DATAFILE), input, STORAGE_SIZE);

    /* Launch kernel function in GPU, with single thread
    and dynamically allocate INVERT_PAGE_TABLE_SIZE bytes of share memory,
    which is used for variables declared as "extern __shared__" */
    mykernel<<<1, 1, INVERT_PAGE_TABLE_SIZE>>>(input_size);

    const cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        const auto err_msg =
            std::string("mykernel launch failed: ") + cudaGetErrorString(cuda_status);
        throw std::runtime_error(err_msg);
    }

    printf("input size: %d\n", input_size);

    cudaDeviceSynchronize();
    cudaDeviceReset();

    write_binary_file(const_cast<char*>(OUTFILE), results, STORAGE_SIZE);

    printf("pagefault number is %d\n", pagefault_num);

    return 0;
}
