#pragma once

#include <cinttypes>
#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned char uchar;
typedef uint32_t u32;
// type for an memory address
typedef u32 addr_t;
// type for page table entry
typedef u32 pte_t;

struct VirtualMemory {
    // physical memory
    uchar* buffer;
    // disk storage, stores pages swapped from memory
    uchar* storage;
    pte_t* invert_page_table;
    /*
     * Index: virtual page number
     * Value: [addr_t] The address where the page is stored in storage area
     */
    u32* swap_table;
    // used when both buffer and storage are full, can contain 1 page
    uchar* swap_buffer;
    /*
     * Index: storage page number
     * Value: [bool] whether the corresponding page is stored in storage area
     */
    uchar* storage_bitmap;
    int* pagefault_num_ptr;

    int PAGESIZE;
    int INVERT_PAGE_TABLE_SIZE;
    int PHYSICAL_MEM_SIZE;
    int STORAGE_SIZE;
    int PAGE_ENTRIES;

    u32 mru;
    u32 lru;
    u32 lru2;
    // number of valid entries in the page table
    int entry_num;
};

__device__ void vm_init(VirtualMemory* vm, uchar* buffer, uchar* storage, pte_t* invert_page_table,
                        u32* swap_table, uchar* storage_bitmap, uchar* swap_buffer,
                        int* pagefault_num_ptr, int page_size, int invert_page_table_size,
                        int physical_mem_size, int storage_size, int page_entries);
__device__ uchar vm_read(VirtualMemory* vm, addr_t vaddr);
__device__ void vm_write(VirtualMemory* vm, addr_t vaddr, uchar value);
__device__ void vm_snapshot(VirtualMemory* vm, uchar* results, int offset, int input_size);
