#include <cassert>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

#include "virtual_memory.h"

// max virtual address for 160KB virtual memory
constexpr __device__ u32 MAX_VA = 160 * 1024 - 1;

constexpr __device__ u32 INVALID_ADDR = 0xFFFFFFFFu;
constexpr __device__ u32 PPN_MASK = 0x3FFu;
constexpr __device__ u32 VPN_MASK = 0x1FFFu;
constexpr __device__ u32 OFFSET_MASK = 0x1Fu;
constexpr __device__ int OFFSET_LEN = 5;
constexpr __device__ int VALID_BIT_START = 31;
constexpr __device__ int LRU_BIT_START = 21;

__device__ inline void check_valid_va(const addr_t vaddr) {
    if (vaddr > MAX_VA) {
        printf("Invalid virtual address: %u. Max allowed: %u\n", vaddr, MAX_VA);
        assert(false);
    }
}

__device__ inline void no_result_error() {
    printf("No result can be found. This is probably caused by "
           "referencing a page that has not been written to yet.\n");
    assert(false);
}

// get page number from a virtual address
__device__ inline u32 get_page_number(const addr_t addr) {
    return addr >> OFFSET_LEN;
}

// get page offset from a virtual address
__device__ inline u32 get_offset(const addr_t addr) {
    return addr & OFFSET_MASK;
}

// check whether a page entry is valid (valid bit == 1)
__device__ inline bool is_valid(const pte_t pte) {
    return pte >> VALID_BIT_START == 1;
}

// get virtual page number from a page table entry
__device__ inline u32 pte_get_vpn(const pte_t pte) {
    return pte & VPN_MASK;
}

// get `next LRU` address from a page table entry
__device__ inline u32 pte_next_lru_idx(const pte_t pte) {
    return (pte >> LRU_BIT_START) & PPN_MASK;
}

// set a page table entry
__device__ inline void pte_set(pte_t& pte, const bool valid, const u32 next_lru, const u32 vpn) {
    const u32 valid_bit = valid ? 1 : 0;
    pte = (valid_bit << VALID_BIT_START) + (next_lru << LRU_BIT_START) + vpn;
}

// set target's bits starting from `start` with length denoted by `mask` to `val`
// e.g. set_bits(0b00000, 1, 0b111, 0b101) -> 0b01010
__device__ inline void set_bits(u32& target, const u32 start, const u32 mask, const u32 val) {
    target = (target & ~(mask << start)) | (val << start);
}

// set `next LRU` entry of a page table entry to given value, assuming the entry is valid
__device__ inline void pte_set_next_lru_idx(pte_t& pte, const u32 idx) {
    set_bits(pte, LRU_BIT_START, PPN_MASK, idx);
}

// set VPN
__device__ inline void pte_set_vpn(pte_t& pte, const u32 vpn) {
    set_bits(pte, 0, VPN_MASK, vpn);
}

// generate memory address from page number and page offset
__device__ inline addr_t full_addr(const u32 page_number, const u32 offset) {
    return (page_number << 5) + offset;
}


__device__ void clean_virtual_memory(VirtualMemory* vm) {
    vm->mru = vm->lru = vm->lru2 = PPN_MASK;
    vm->entry_num = 0;

    // initialize swap table
    const u32 virtual_page_num = (vm->PHYSICAL_MEM_SIZE + vm->STORAGE_SIZE) / vm->PAGESIZE;
    for (u32 i = 0; i < virtual_page_num; ++i) {
        vm->swap_table[i] = INVALID_ADDR;
    }
    memset(vm->storage_bitmap, 0, vm->STORAGE_SIZE);
}

__device__ void vm_init(VirtualMemory* vm, uchar* buffer, uchar* storage, pte_t* invert_page_table,
                        u32* swap_table, uchar* storage_bitmap, uchar* swap_buffer,
                        int* pagefault_num_ptr, int page_size, int invert_page_table_size,
                        int physical_mem_size, int storage_size, int page_entries) {
    // initialize constants
    vm->PAGESIZE = page_size;
    vm->INVERT_PAGE_TABLE_SIZE = invert_page_table_size;
    vm->PHYSICAL_MEM_SIZE = physical_mem_size;
    vm->STORAGE_SIZE = storage_size;
    vm->PAGE_ENTRIES = page_entries;

    // initialize variables
    vm->buffer = buffer;
    vm->storage = storage;
    vm->invert_page_table = invert_page_table;
    vm->pagefault_num_ptr = pagefault_num_ptr;
    vm->swap_table = swap_table;
    vm->storage_bitmap = storage_bitmap;
    vm->swap_buffer = swap_buffer;

    // before first vm_write or vm_read
    clean_virtual_memory(vm);
}

__device__ void lru_mv_to_head(VirtualMemory* vm, u32 tgt_idx, u32 prev_idx) {
    // if this is already the MRU, do nothing
    if (tgt_idx == vm->mru) return;

    // move the node to the head (i.e. MRU)
    pte_t& tgt_entry = vm->invert_page_table[tgt_idx];
    pte_t& prev_entry = vm->invert_page_table[prev_idx];
    pte_set_next_lru_idx(prev_entry, pte_next_lru_idx(tgt_entry));
    pte_set_next_lru_idx(tgt_entry, vm->mru);
    vm->mru = tgt_idx;

    // set LRU if necessary
    if (vm->lru == tgt_idx) { vm->lru = prev_idx; }
}

// bring desired page from 'disk' into 'memory' (and evict existing pages if necessary)
__device__ u32 do_swap(VirtualMemory* vm, const u32 vpn, const bool is_read) {
    // move the tail (victim) to the head of LRU list
    const u32 victim_idx = vm->lru;
    const addr_t victim_pa = full_addr(victim_idx, 0);
    pte_t& victim_pte = vm->invert_page_table[victim_idx];
    const u32 victim_vpn = pte_get_vpn(victim_pte);
    lru_mv_to_head(vm, victim_idx, vm->lru2);

    // swap out (to swap buffer)
    memcpy(vm->swap_buffer, &vm->buffer[victim_pa], vm->PAGESIZE);

    // swap in
    //* if this VPN corresponds to a valid page (i.e. previously swapped out page),
    //* bring that back to memory. Otherwise, leave the memory content (garbage now) alone
    pte_set_vpn(victim_pte, vpn);
    const addr_t tgt_storage_addr = vm->swap_table[vpn];
    if (tgt_storage_addr != INVALID_ADDR) {
        memcpy(&vm->buffer[victim_pa], &vm->storage[tgt_storage_addr], vm->PAGESIZE);
        vm->storage_bitmap[get_page_number(tgt_storage_addr)] = 0;
        vm->swap_table[vpn] = INVALID_ADDR;
    }
    else if (is_read) { no_result_error(); }

    // find an empty slot and write swap buffer content to storage
    u32 storage_page_num = 0;
    while (vm->storage_bitmap[storage_page_num] != 0) {
        storage_page_num++;
    }
    //* write data
    const addr_t victim_storage_addr = full_addr(storage_page_num, 0);
    vm->storage_bitmap[storage_page_num] = 1;
    vm->swap_table[victim_vpn] = victim_storage_addr;
    memcpy(&vm->storage[victim_storage_addr], vm->swap_buffer, vm->PAGESIZE);

    return victim_idx;
}

/// <summary>
/// Searches through the LRU list to find the entry with desired virtual page number.
/// </summary>
/// <param name="vm">Pointer to a VirtualMemory struct</param>
/// <param name="vpn">The virtual page number to search for</param>
/// <param name="result">The resulting PPN</param>
/// <returns>true if the result is found before the end of the list, false otherwise.</returns>
__device__ bool search_vpn(VirtualMemory* vm, const u32 vpn, u32& result) {
    if (vm->entry_num == 0) return false;

    const u32 lru_idx = vm->lru;
    const u32 mru_idx = vm->mru;
    u32 prev_idx = INVALID_ADDR, cur_idx = mru_idx;

    while (prev_idx != lru_idx) {
        const pte_t& cur_entry = vm->invert_page_table[cur_idx];
        // check current element
        if (pte_get_vpn(cur_entry) == vpn) {
            result = cur_idx;
            lru_mv_to_head(vm, cur_idx, prev_idx);
            return true;
        }
        // move a step forward
        vm->lru2 = prev_idx;
        prev_idx = cur_idx;
        cur_idx = pte_next_lru_idx(cur_entry);
    }
    ++(*vm->pagefault_num_ptr);
    return false;
}

// given a virtual address, return the corresponding physical address
__device__ addr_t find_pa(VirtualMemory* vm, const addr_t vaddr, const bool is_read) {
    check_valid_va(vaddr);
    const u32 vpn = get_page_number(vaddr);
    const u32 page_offset = get_offset(vaddr);
    u32 ppn;
    // search for the page from memory first
    if (!search_vpn(vm, vpn, ppn) || !is_valid(vm->invert_page_table[ppn])) {
        // if there's nothing to look for in disk
        if (vm->entry_num < vm->PAGE_ENTRIES) {
            // if we are reading, that means we are trying to access
            // something that's not written before
            if (is_read) { no_result_error(); }
            // if we are writing, we need to create a new memory entry
            else {
                // first time add node
                if (vm->entry_num == 0) {
                    ++(*vm->pagefault_num_ptr);
                    vm->lru = 0;
                }
                pte_set(vm->invert_page_table[vm->entry_num], true, vm->mru, vpn);
                ppn = vm->mru = vm->entry_num++;
            }
        }
        // else, page fault encountered. We check the 'disk' (global memory)
        else { ppn = do_swap(vm, vpn, is_read); }
    }
    return full_addr(ppn, page_offset);
}

__device__ uchar vm_read(VirtualMemory* vm, const addr_t vaddr) {
    return vm->buffer[find_pa(vm, vaddr, true)];
}

__device__ void vm_write(VirtualMemory* vm, const addr_t vaddr, const uchar value) {
    vm->buffer[find_pa(vm, vaddr, false)] = value;
}

// Complete snapshot function together with vm_read to
// load elements from data to result buffer
__device__ void vm_snapshot(VirtualMemory* vm, uchar* results, int offset, int input_size) {
    for (int i = 0; i < input_size; i++) {
        results[i] = vm_read(vm, i + offset);
    }
}
