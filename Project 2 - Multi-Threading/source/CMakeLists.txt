project(hw2_main)

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)

add_executable(hw2 hw2.cpp)

# link pthreads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(hw2 PRIVATE Threads::Threads)
