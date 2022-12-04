#include "async.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_NULL(ptr, msg)                                                                                                               \
    if (ptr == NULL) {                                                                                                                     \
        perror(msg);                                                                                                                       \
        exit(EXIT_FAILURE);                                                                                                                \
    }


static my_queue_t* queue;

/// create num_threads threads and initialize the thread pool
void async_init(int num_threads) {
    printf("Initializing thread pool with %d threads...\n", num_threads);
    // init the thread pool
    queue = malloc(sizeof(my_queue_t));
    CHECK_NULL(queue, "malloc of queue failed")
    queue->thread_num = num_threads;
    queue->task_num = 0;
    queue->task_head = queue->task_tail = NULL;
    queue->exit = false;
    pthread_mutex_init(&queue->task_mutex, NULL);
    pthread_cond_init(&queue->start, NULL);
    pthread_cond_init(&queue->finish, NULL);

    // init worker threads
    pthread_t thread;
    queue->threads = malloc(sizeof(pthread_t) * num_threads);
    CHECK_NULL(queue->threads, "malloc of threads array failed")
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&thread, NULL, (void*)worker_start, (void*)(long)i)) {
            perror("pthread_create failed");
            exit(EXIT_FAILURE);
        }
    }
    printf("Thread pool initialization finished.\n");
}

void async_run(void (*handler)(int), int args) {
    // create a new task
    my_task_t* task = malloc(sizeof(my_task_t));
    CHECK_NULL(task, "task creation failed")
    task->func = handler;
    task->args = args;
    task->next = NULL;

    add_task(task);
}

void async_destroy() {
    printf("destroying thread pool...\n");
    pthread_mutex_lock(&queue->task_mutex);
    queue->exit = true;
    // clean undone tasks if necessary
    while (queue->task_head != NULL) {
        my_task_t* task = queue->task_head;
        queue->task_head = queue->task_head->next;
        free(task);
    }
    pthread_mutex_unlock(&queue->task_mutex);

    // wait for all threads to finish
    for (int i = 0; i < queue->thread_num; ++i) {
        if (pthread_join(queue->threads[i], NULL)) {
            perror("A worker thread failed to join");
            exit(EXIT_FAILURE);
        }
    }

    free(queue->threads);
    free(queue);
    printf("Thread pool destroyed.\n");
}

void worker_start(const long thread_id) {
    while (true) {
        pthread_mutex_lock(&queue->task_mutex);

        // if there is no task to do, suspend until start signal arises
        while (!queue->exit && queue->task_num == 0) {
            pthread_cond_wait(&queue->start, &queue->task_mutex);
        }
        if (queue->exit) break;

        // get a task from the task list and do it
        my_task_t* task = get_task();
        printf("[Thread %ld] retrieved a task, %d tasks in the queue\n", thread_id, queue->task_num);
        pthread_mutex_unlock(&queue->task_mutex);
        printf("[Thread %ld] doing the task\n", thread_id);
        if (task != NULL) {
            task->func(task->args);
            free(task);
        }
        printf("[Thread %ld] task done\n", thread_id);
    }
    pthread_mutex_unlock(&queue->task_mutex);
    pthread_exit(NULL);
}

/// append `task` to the end of the linked list
void add_task(my_task_t* task) {
    if (task == NULL) return;

    // append the task
    pthread_mutex_lock(&queue->task_mutex);
    if (queue->task_num == 0) queue->task_head = queue->task_tail = task;
    else {
        queue->task_tail->next = task;
        queue->task_tail = task;
    }
    queue->task_num++;
    pthread_mutex_unlock(&queue->task_mutex);

    printf("task added, %d tasks in the queue\n", queue->task_num);

    // signal the threads to start
    pthread_cond_signal(&queue->start);
}

/// get the first task from the linked list
my_task_t* get_task() {
    if (queue->task_num == 0) return NULL;

    queue->task_num--;
    my_task_t* task = queue->task_head;
    if (queue->task_num == 0) queue->task_head = queue->task_tail = NULL;
    else queue->task_head = queue->task_head->next;
    return task;
}