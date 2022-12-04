#include <unistd.h>
#include <sys/wait.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>

void* thread_wait(void* args);

pthread_cond_t condvar;
pthread_mutex_t mu;
int flag;


int main() {
    srand(time(NULL));
    int total = 14-rand()%4;
    pid_t pid;
    for(int i=0;i<total;i++) {
        if(pid=fork()) {
            int magic = 4-rand()%3;
            if(i%magic==0) break;
        }
        else {
            if(i%3==0) continue;
            char *const __argv[] = {"/tmp/halt", NULL};
            execv(__argv[0],__argv);
        }
    }
    //srand(time(NULL));
    int thead_magic = rand()%12;
    pthread_t* threads = malloc(sizeof(pthread_t)*thead_magic);
    pthread_mutex_init(&mu,NULL);
    pthread_cond_init(&condvar,NULL);
    flag = 1;
    for(int i=0; i<thead_magic; i++){
        pthread_create(threads+i,NULL,thread_wait,NULL);
    }
    wait(NULL);
    pthread_mutex_lock(&mu);
    flag = 0;
    pthread_mutex_unlock(&mu);
    pthread_cond_broadcast(&condvar);
    for(int i=0; i<thead_magic; i++){
        pthread_join(threads[i],NULL);
    }
    free(threads);
}

void* thread_wait(void* args) {
    pthread_mutex_lock(&mu);
    while(flag) pthread_cond_wait(&condvar,&mu);
    pthread_mutex_unlock(&mu);
    return NULL;
}