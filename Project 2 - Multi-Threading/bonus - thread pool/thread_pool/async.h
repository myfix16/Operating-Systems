#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>
#include <stdbool.h>


typedef struct my_task {
    void (*func)(int);
    int args;
    struct my_task* next;
} my_task_t;

typedef struct my_queue {
    int thread_num;
    int task_num;
    my_task_t* task_head;
    my_task_t* task_tail;
    bool exit;
    pthread_t* threads;
    pthread_mutex_t task_mutex;
    pthread_cond_t start;
    pthread_cond_t finish;
} my_queue_t;

void async_init(int);
void async_run(void (*handler)(int), int args);
void async_destroy();
void worker_start(long id);

// linked list operations
void add_task(my_task_t* work);
my_task_t* get_task();

#endif
