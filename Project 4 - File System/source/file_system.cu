#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#include "file_system.h"

constexpr u32 INVALID_FCB_INDEX = 0xFFFFFFFFu;
constexpr u16 INVALID_FCB_INDEX_U16 = 0xFFFFu;

__device__ __managed__ u16 gtime = 0;
__device__ void print_list(FileSystem* fs);

namespace utils
{
    // credit to https://stackoverflow.com/questions/34873209/implementation-of-strcmp
    __device__ bool str_equal(const char* s1, const char* s2) {
        auto p1 = reinterpret_cast<const uchar*>(s1);
        auto p2 = reinterpret_cast<const uchar*>(s2);
        while (*p1 && *p1 == *p2) {
            ++p1;
            ++p2;
        }
        return !((*p1 > *p2) - (*p2 > *p1));
    }

    __device__ void strcpy(uchar* dest, const uchar* src) {
        while (*src) {
            *dest = *src;
            ++dest;
            ++src;
        }
        *dest = '\0';
    }

    __device__ inline void strcpy(char* dest, const char* src) {
        strcpy(reinterpret_cast<uchar*>(dest), reinterpret_cast<const uchar*>(src));
    }

    __device__ inline void swap(FCB& fcb1, FCB& fcb2) noexcept {
        const FCB temp = fcb1;
        fcb1 = fcb2;
        fcb2 = temp;
    }

    // selection sort
    __device__ void sort(FCB files[], const int size, const FCBComp& comp) {
        for (int i = 0; i < size - 1; ++i) {
            int min_idx = i;
            for (int j = i + 1; j < size; ++j) {
                if (comp(files[j], files[min_idx])) min_idx = j;
            }
            if (min_idx != i) swap(files[i], files[min_idx]);
        }
    }
} // namespace utils

__device__ inline void invalid_value_error(const char* name, const u32 idx) {
    printf("[Error] Invalid %s: %u!\n", name, idx);
}

__device__ bool FCB::empty() const {
    return filename_ptr_[0] == '\0';
}
__device__ const char* FCB::filename() const {
    return filename_ptr_;
}

__device__ void FCB::set_filename(const char* name) const {
    if (filename_ptr_ == nullptr) {
        invalid_value_error("filename_ptr(nullptr) of FCB %d: %d\n", file_idx);
    }
    utils::strcpy(filename_ptr_, name);
    filename_ptr_[19] = '\0';
}

__device__ void FCB::clear_filename() const {
    filename_ptr_[0] = '\0';
}

__device__ inline u16& FCB::block_idx_begin() const {
    return *block_begin_ptr_;
}

__device__ inline u16 FCB::block_idx_end() const {
    // printf("Calling block_idx_end\n");
    const u16 size = file_size();
    const u16 end_idx = block_idx_begin() + fs_->size_2_blocks(size);
    return end_idx;
}

__device__ uchar* FCB::file_content() const {
    return &fs_->volume[fs_->FILE_BASE_ADDRESS + file_idx * fs_->STORAGE_BLOCK_SIZE];
}

__device__ u16& FCB::file_size() const {
    return *filesize_ptr_;
}

__device__ u16& FCB::time_created() const {
    return *time_created_ptr_;
}

__device__ u16& FCB::time_modified() const {
    return *time_modified_ptr_;
}

__device__ u16& FCB::next() const {
    return *next_ptr_;
}

__device__ FCB::FCB(FileSystem* fs) {
    this->fs_ = fs;
}
__device__ FCB::FCB(FileSystem* fs, const int index, const bool create) {
    // printf("[FCB constructor] idx=%d, total files: %d\n", index, fs->num_files);
    this->fs_ = fs;
    this->file_idx = index;

    if (index >= fs->FCB_ENTRIES) {
        printf("[Error in FCB constructor] index %d out of range\n", index);
    }
    const auto fcb_ptr = &fs->volume[fs->SUPERBLOCK_SIZE + index * fs->FCB_SIZE];
    this->filename_ptr_ = reinterpret_cast<char*>(fcb_ptr + FILENAME_OFFSET);
    this->block_begin_ptr_ = reinterpret_cast<u16*>(fcb_ptr + START_BLOCK_OFFSET);
    this->filesize_ptr_ = reinterpret_cast<u16*>(fcb_ptr + FILESIZE_OFFSET);
    this->time_created_ptr_ = reinterpret_cast<u16*>(fcb_ptr + TIME_CREATED_OFFSET);
    this->time_modified_ptr_ = reinterpret_cast<u16*>(fcb_ptr + TIME_MODIFIED_OFFSET);
    this->next_ptr_ = reinterpret_cast<u16*>(fcb_ptr + NEXT_OFFSET);

    // initialize for new files
    if (create) {
        // printf("[FCB constructor] creating FSB with idx: %d\n", index);
        block_idx_begin() = fs->num_files == 0 ? 0 : fs->last_block_idx();
        file_size() = 0;
        time_created() = time_modified() = gtime++;
        // maintain FCB linked list
        next() = INVALID_FCB_INDEX_U16;

        if (fs->num_files == 0)
            fs->head_fcb = fs->tail_fcb = index;
        else {
        // print_list(fs);

        // printf("head is %d tail is %d\n", fs->head_fcb, fs->tail_fcb);
            fs->fcb(fs->tail_fcb).next() = index;
        // printf("head is %d tail is %d\n", fs->head_fcb, fs->tail_fcb);
            fs->tail_fcb = index;
        }
    }
}

__device__ inline u16 VCB::get_bit_mask(const u32 block_idx) const {
    return 0x80 >> block_idx % 8;
}

__device__ inline void VCB::free(const u32 block_idx) const {
    volume_[block_idx / 8] &= ~get_bit_mask(block_idx);
}

__device__ inline void VCB::occupy(const u32 block_idx) const {
    volume_[block_idx / 8] |= get_bit_mask(block_idx);
}

__device__ void VCB::occupy_file(const FCB& file) const {
    const u16 begin = file.block_idx_begin();
    const u16 end = file.block_idx_end();
    for (int i = begin; i < end; i++) {
        occupy(i);
    }
}

__device__ void VCB::free_file(const FCB& file) const {
    const u16 begin = file.block_idx_begin();
    const u16 end = file.block_idx_end();
    for (int i = begin; i < end; i++) {
        free(i);
    }
}

__device__ VCB::VCB(FileSystem* fs) {
    this->fs_ = fs;
    this->volume_ = fs->volume;
}

__device__ FCB FileSystem::fcb() {
    return FCB(this);
}

__device__ FCB FileSystem::fcb(const int index, const bool create) {
    // printf("FCB init: index = %d\n", index);
    // printf("FS num files: %d\n", num_files);
    return FCB(this, index, create);
}

__device__ VCB FileSystem::vcb() {
    return VCB(this);
}

__device__ u16 FileSystem::last_block_idx() {
    // printf("Calling last_block_idx\n");
    // printf("num files: %d\n", num_files);
    // printf("last file: %d\n", tail_fcb);
    return FCB(this, tail_fcb).block_idx_end();
}

__device__ bool FileSystem::is_full() {
    const int max_num_blocks = MAX_FILE_SIZE / STORAGE_BLOCK_SIZE;
    return last_block_idx() >= max_num_blocks;
}
__device__ void FileSystem::write(const FCB& file, const uchar* data, const int size) {
    // move data if size changes
    // print_list(this);
    if (size != file.file_size()) {
        // printf("[Write] num files: %d\n", num_files);
        const u16 last_block_idx = this->last_block_idx();
        // const u16 last_block_idx = temp.block_idx_end();
        ;
        // printf("[Write] retrieved last block idx: %d\n", last_block_idx);
        const VCB vcb(this);

        if (size < file.file_size()) {
            file.file_size() = size;
            FCB this_file = file;
            // do shrink
            for (int i = file.file_idx + 1; i < num_files; ++i) {
                auto next_file = FCB(this, i);
                next_file.block_idx_begin() = this_file.block_idx_end();
                this_file = next_file;
            }
            // update VCB
            for (int i = this->last_block_idx(); i < last_block_idx; ++i) {
                vcb.free(i);
            }
        }
        else {
            // check whether there is enough space to write
            const u16 delta_blocks = size_2_blocks(size) - size_2_blocks(file.file_size());
            const int max_num_blocks = MAX_FILE_SIZE / STORAGE_BLOCK_SIZE;
            if (this->last_block_idx() + delta_blocks > max_num_blocks) {
                invalid_value_error("new size, too large", size);
                return;
            }
            // do expanding
            file.file_size() = size;
            for (int i = num_files - 1; i > file.file_idx; --i) {
                auto next_file = FCB(this, i);
                next_file.block_idx_begin() += delta_blocks;
            }
            // update VCB
            for (int i = last_block_idx; i < this->last_block_idx(); ++i) {
                vcb.occupy(i);
            }
        }
    // print_list(this);
    }
    if (data != nullptr) utils::strcpy(file.file_content(), data);
    file.time_modified() = gtime++;
}

__device__ void FileSystem::rm(const FCB& file) {
    printf("[RM] %s\n", file.filename());
    write(file, nullptr, 0);
    file.clear_filename();
    num_files--;
}

__device__ u16 FileSystem::size_2_blocks(const int size) const {
    const u16 quotient = size / STORAGE_BLOCK_SIZE;
    const u16 remainder = size % STORAGE_BLOCK_SIZE ? 1 : 0;
    return quotient + remainder;
}

__device__ void print_list(FileSystem* fs) {
    if (fs->num_files == 0) return;
    printf("printing list, head %d count %d\n", fs->head_fcb, fs->num_files);
    u16 cur_idx = fs->head_fcb;
    while (cur_idx != INVALID_FCB_INDEX_U16) {
        // FCB cur_entry = fs->fcb(cur_idx);
        FCB cur_entry(fs, cur_idx);
        printf("%s:%d->", cur_entry.filename(), cur_entry.file_idx);
        cur_idx = cur_entry.next();
    }
    printf("NULL\n");
}

__device__ u16 FileSystem::search_file(const char* filename, u16* prev_ptr) {
    // printf("Searching for file: %s\n", filename);
    // printf("num files: %d\n", num_files);
    // printf("printing list\n");
    // print_list(this);
    u16 cur_idx = head_fcb, prev_idx = INVALID_FCB_INDEX_U16;
    while (cur_idx != INVALID_FCB_INDEX_U16) {
        FCB cur_entry = fcb(cur_idx);
        if (utils::str_equal(filename, cur_entry.filename())) {
            // todo: correct time modified?
            cur_entry.time_modified() = gtime++;
            if (prev_ptr != nullptr) *prev_ptr = prev_idx;
            // printf("Found with idx: %d\n", cur_idx);
            return cur_idx;
        }
        prev_idx = cur_idx;
        cur_idx = cur_entry.next();
    }
    // printf("Not found\n");
    return INVALID_FCB_INDEX_U16;
}

__device__ void fs_init(FileSystem* fs, uchar* volume, const int superblock_size,
                        const int fcb_size, const int fcb_entries, const int volume_size,
                        const int storage_block_size, const int max_filename_size,
                        const int max_file_num, const int max_file_size,
                        const int file_base_address) {
    // init variables
    fs->volume = volume;
    fs->num_files = 0;

    // init constants
    fs->SUPERBLOCK_SIZE = superblock_size;
    fs->FCB_SIZE = fcb_size;
    fs->FCB_ENTRIES = fcb_entries;
    fs->STORAGE_SIZE = volume_size;
    fs->STORAGE_BLOCK_SIZE = storage_block_size;
    fs->MAX_FILENAME_SIZE = max_filename_size;
    fs->MAX_FILE_NUM = max_file_num;
    fs->MAX_FILE_SIZE = max_file_size;
    fs->FILE_BASE_ADDRESS = file_base_address;
}
// __device__ void print_list(FileSystem* fs);
__device__ u32 fs_open(FileSystem* fs, char* s, const int op) {
    // search through all entries to find the desired file
    if (fs->num_files > 0) {

        const u16 target_idx = fs->search_file(s, nullptr);
        if (target_idx != INVALID_FCB_INDEX_U16) {
            fs->fcb(target_idx).time_modified() = gtime++;
            return target_idx;
        }
    }
    // We cannot find desired file
    // if is reading, raise an error
    if (op == OpType::Read) {
        printf("[Error] File not found: %s\n", s);
        return INVALID_FCB_INDEX;
    }
    // else, try to find free space and create an empty file
    if (fs->is_full() || fs->num_files >= fs->MAX_FILE_NUM) {
        printf("Error: No empty storage available\n");
        return INVALID_FCB_INDEX;
    }
    // create a new empty file
    int new_index = -1;
    for (int i = 0; i < fs->MAX_FILE_NUM; ++i) {
        FCB current(fs, i);
        if (utils::str_equal("", current.filename())) {
            new_index = i;
            break;
        }
    }
    // const u16 new_index = fs->search_file("", nullptr);
    const FCB target_entry = fs->fcb(new_index, true);
    fs->num_files++;
    target_entry.set_filename(s);

    return target_entry.file_idx;
}

/* Implement read operation here */
__device__ void fs_read(FileSystem* fs, uchar* output, u32 size, const u32 fp) {
    if (fp > fs->MAX_FILE_NUM) {
        invalid_value_error("index", fp);
        return;
    }
    const FCB file = fs->fcb(fp);
    utils::strcpy(output, file.file_content());
}

/* Implement write operation here */
__device__ u32 fs_write(FileSystem* fs, uchar* input, const u32 size, const u32 fp) {
    if (fp > fs->MAX_FILE_NUM) {
        invalid_value_error("index", fp);
        return INVALID_FCB_INDEX;
    }
    if (size > fs->MAX_FILE_SIZE / fs->MAX_FILE_NUM) {
        invalid_value_error("size", size);
        return INVALID_FCB_INDEX;
    }

    fs->write(fs->fcb(fp), input, size);

    return 0;
}

/* Implement LS_D and LS_S operation here */
__device__ void fs_gsys(FileSystem* fs, const int op) {
    FCB* files = nullptr;
    cudaMalloc(&files, fs->MAX_FILE_NUM * sizeof(FCB));
    int count = 0;
    for (int i = 0; i < fs->MAX_FILE_NUM; i++) {
        FCB current = fs->fcb(i);
        if (!current.empty()) { files[count++] = current; }
    }

    switch (op) {
    case GSysType::Ls_D:
        utils::sort(files, count, ModifiedTimeGreater{});
        printf("===sort by modified time===\n");
        for (int i = 0; i < count; i++) {
            printf("%s\n", files[i].filename());
        }
        break;
    case GSysType::Ls_S:
        utils::sort(files, count, FileSizeGreater{});
        printf("===sort by file size===\n");
        for (int i = 0; i < count; i++) {
            printf("%s %d\n", files[i].filename(), files[i].file_size());
        }
        break;
    default:
        invalid_value_error("op", op);
    }

    cudaFree(files);
}

/* Implement rm operation here */
__device__ void fs_gsys(FileSystem* fs, const int op, char* s) {
    if (op != GSysType::Rm) {
        invalid_value_error("op, rm expected", op);
        return;
    }

    // find and remove the file by erasing meta data
    u16 prev_idx;
    const u16 target_idx = fs->search_file(s, &prev_idx);
    if (target_idx != INVALID_FCB_INDEX_U16) {
        if (fs->num_files == 1) fs->head_fcb = fs->tail_fcb = INVALID_FCB_INDEX_U16;
        else if (target_idx == fs->head_fcb) {
            fs->head_fcb = fs->fcb(fs->head_fcb).next();
            fs->fcb(target_idx).next() = INVALID_FCB_INDEX_U16;
        }
        else {
            fs->fcb(prev_idx).next() = fs->fcb(target_idx).next();
            if (target_idx == fs->tail_fcb) fs->tail_fcb = prev_idx;
        }
        fs->rm(fs->fcb(target_idx));
    }
    else
        printf("[Error] Cannot remove non-existent file: %s\n", s);
}
