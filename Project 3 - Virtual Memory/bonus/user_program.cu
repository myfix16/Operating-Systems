#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include "virtual_memory.h"

__device__ void user_program(VirtualMemory* vm, uchar* input, uchar* results, int input_size) {
    for (int i = 0; i < input_size; i++) {
        vm_write(vm, i, input[i]);
    }
    const int write_fault = *vm->pagefault_num_ptr;

    for (int i = input_size - 1; i >= input_size - 32769; i--) {
        vm_read(vm, i);
    }
    const int read_fault = *vm->pagefault_num_ptr - write_fault;

    vm_snapshot(vm, results, 0, input_size);

    const int snapshot_fault = *vm->pagefault_num_ptr - write_fault - read_fault;
    printf("Page faults: %u\n", *vm->pagefault_num_ptr);
}
