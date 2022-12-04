cmake_minimum_required(VERSION 3.16)

project(ThreadPool)

set(CMAKE_C_STANDARD 99)

if(GCC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb3 -Wall")
endif()

include_directories(${PROJECT_SOURCE_DIR})
add_executable(httpserver httpserver.c libhttp.c async.c util.c)

# link pthreads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(httpserver PRIVATE Threads::Threads)