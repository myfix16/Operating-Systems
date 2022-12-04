#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include "virtual_memory.h"

__device__ void user_program(VirtualMemory* vm, uchar* input, uchar* results, int input_size) {
    for (int i = 0; i < input_size; i++) {
        vm_write(vm, i, input[i]);
    }
    printf("Write finished\n");
    const int write_fault = *vm->pagefault_num_ptr;

    for (int i = input_size - 1; i >= input_size - 32769; i--) {
        vm_read(vm, i);
    }
    printf("Read finished\n");
    const int read_fault = *vm->pagefault_num_ptr - write_fault;

    printf("Doing snapshots...\n");
    vm_snapshot(vm, results, 0, input_size);

    const int snapshot_fault = *vm->pagefault_num_ptr - write_fault - read_fault;
    printf("Write fault: %d\n", write_fault);
    printf("Read fault: %d\n", read_fault);
    printf("Snapshot fault: %d\n", snapshot_fault);
}

__device__ void user_program2(VirtualMemory* vm, uchar* input, uchar* results, int input_size) {
    // write the data.bin to the VM starting from address 32*1024 [32K, 160K]
    for (int i = 0; i < input_size; i++)
        vm_write(vm, 32 * 1024 + i, input[i]);
    printf("Write finished\n");
    const int write_fault = *vm->pagefault_num_ptr;

    // write (32KB-32B) data to the VM starting from 0 [0, 32KB-32B]
    for (int i = 0; i < 32 * 1023; i++)
        vm_write(vm, i, input[i + 32 * 1024]);
    printf("Write 2 finished\n");
    const int read_fault = *vm->pagefault_num_ptr - write_fault;

    // readout VM[32K, 160K] and output to snapshot.bin, which should be the same with data.bin
    vm_snapshot(vm, results, 32 * 1024, input_size);
    const int snapshot_fault = *vm->pagefault_num_ptr - write_fault - read_fault;
    printf("Write fault: %d\n", write_fault);
    printf("write 2 fault: %d\n", read_fault);
    printf("Snapshot fault: %d\n", snapshot_fault);
}