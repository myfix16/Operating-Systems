#pragma once

#include <cinttypes>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

using uchar = unsigned char;
using u32 = uint32_t;
using u16 = uint16_t;

constexpr int G_WRITE = 1;
constexpr int G_READ = 0;
constexpr int LS_D = 0;
constexpr int LS_S = 1;
constexpr int RM = 2;

enum OpType { Read = G_READ, Write = G_WRITE };

enum GSysType { Ls_D = LS_D, Ls_S = LS_S, Rm = RM };

struct FileSystem {
    uchar* volume;
    int num_files;
    u16 head_fcb;
    u16 tail_fcb;

    int SUPERBLOCK_SIZE;
    int FCB_SIZE;
    int FCB_ENTRIES;
    int STORAGE_SIZE;
    int STORAGE_BLOCK_SIZE;
    int MAX_FILENAME_SIZE;
    int MAX_FILE_NUM;
    int MAX_FILE_SIZE; // max size of all files
    int FILE_BASE_ADDRESS;


    class FCB {
        static constexpr auto FILENAME_OFFSET = 0;
        static constexpr auto FILESIZE_OFFSET = 20;
        static constexpr auto START_BLOCK_OFFSET = 22;
        static constexpr auto TIME_CREATED_OFFSET = 24;
        static constexpr auto TIME_MODIFIED_OFFSET = 26;
        static constexpr auto NEXT_OFFSET = 28;

        friend struct FileSystem;

    public:
        int file_idx = -1;
        
        __device__ bool empty() const;
        __device__ const char* filename() const;
        __device__ void set_filename(const char* name) const;
        __device__ void clear_filename() const;
        __device__ uchar* file_content() const;
        __device__ u16& file_size() const;
        __device__ u16& time_created() const;
        __device__ u16& time_modified() const;
        __device__ u16& next() const;

    // private:
        FileSystem* fs_ = nullptr;
        char* filename_ptr_ = nullptr;
        u16* filesize_ptr_ = nullptr;
        u16* block_begin_ptr_ = nullptr;
        u16* time_created_ptr_ = nullptr;
        u16* time_modified_ptr_ = nullptr;
        u16* next_ptr_ = nullptr;

        __device__ explicit FCB(FileSystem* fs);
        __device__ FCB(FileSystem* fs, int index, bool create=false);
        __device__ inline u16& block_idx_begin() const;
        __device__ inline u16 block_idx_end() const;
    };

    /**
     * \brief Super block is implemented using a bitmap
     */
    class VCB {
        friend struct FileSystem;

    public:
        __device__ void occupy_file(const FCB& file) const;
        __device__ void free_file(const FCB& file) const;

    private:
        uchar* volume_;
        FileSystem* fs_;

        __device__ explicit VCB(FileSystem* fs);
        __device__ inline u16 get_bit_mask(u32 block_idx) const;
        __device__ inline void free(u32 block_idx) const;
        __device__ inline void occupy(u32 block_idx) const;
    };

    __device__ FCB fcb(int index, bool create=false);
    __device__ FCB fcb();
    __device__ VCB vcb();
    __device__ u16 last_block_idx();
    __device__ u16 size_2_blocks(int size) const;
    __device__ u16 search_file(const char* filename, u16* prev_ptr);
    __device__ bool is_full();
    __device__ void write(const FCB& file, const uchar* data, int size);
    __device__ void rm(const FCB& file);
};

using FCB = FileSystem::FCB;
using VCB = FileSystem::VCB;

struct FCBComp {
    __device__ virtual bool operator()(const FCB& lhs, const FCB& rhs) const = 0;
};

struct ModifiedTimeGreater : FCBComp {
    __device__ bool operator()(const FCB& lhs, const FCB& rhs) const override {
        return lhs.time_modified() > rhs.time_modified();
    }
};

struct FileSizeGreater : FCBComp {
    __device__ bool operator()(const FCB& lhs, const FCB& rhs) const override {
        if (lhs.file_size() > rhs.file_size()) return true;
        if (lhs.file_size() < rhs.file_size()) return false;
        return lhs.time_created() < rhs.time_created();
    }
};

__device__ void fs_init(FileSystem* fs, uchar* volume, int superblock_size, int fcb_size,
                        int fcb_entries, int volume_size, int storage_block_size,
                        int max_filename_size, int max_file_num, int max_file_size,
                        int file_base_address);

__device__ u32 fs_open(FileSystem* fs, char* s, int op);
__device__ void fs_read(FileSystem* fs, uchar* output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem* fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem* fs, int op);
__device__ void fs_gsys(FileSystem* fs, int op, char* s);
