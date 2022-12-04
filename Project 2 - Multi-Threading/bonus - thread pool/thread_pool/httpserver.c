#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <assert.h>

#include "libhttp.h"
#include "util.h"
#include "async.h"

/*
 * Global configuration variables.
 */

int num_threads;
int server_port;
char *server_files_directory;
char *server_proxy_hostname;
int server_proxy_port;

/*
 * Reads an HTTP request from stream (fd), and writes an HTTP response
 * containing:
 *
 *   1) If user requested an existing file, respond with the file
 *   2) If user requested a directory and index.html exists in the directory,
 *      send the index.html file.
 *   3) If user requested a directory and index.html doesn't exist, send a list
 *      of files in the directory with links to each.
 *   4) Send a 404 Not Found response.
 */

void handle_files_request(int fd) {
    struct http_request *request = http_request_parse(fd);

    if(request == NULL) goto EXIT;

    char *full_path = join_path(server_files_directory, request->path, NULL);
    /* printf("%s\n %s\n %s\n", server_files_directory, request->path, full_path); */

    struct stat sb;
    stat(full_path, &sb);
    if (S_ISDIR(sb.st_mode)) {

        DIR *dirp;
        struct dirent *dp;
        dirp = opendir(full_path);
        while((dp = readdir(dirp))!=NULL) {
            if(strcmp(dp->d_name, "index.html")==0) {
                char *filename = join_path(full_path, "/index.html", NULL);
                reply_with_file(fd, filename, 200);
                free(filename);
                free(full_path);
                closedir(dirp);
                goto EXIT;
            }
        }
        closedir(dirp);

        char *template = "<html><body><center><p><a href=\"%s\">%s</a></p>";
        size_t template_len = strlen(template), parent_path_len = strlen(request->path);
        char *temp = NULL, *saved, *send_buffer = (char *)malloc(1);
        send_buffer[0] = '\0';
        dirp = opendir(full_path);
        char *display_name, *url;
        while((dp = readdir(dirp))!=NULL) {
            if(!strcmp(dp->d_name, ".")) continue;
            else if(!strcmp(dp->d_name, "..")) {
                display_name = "Parent Directory";
                url = get_parent_name(request->path);
            } else {
                display_name = dp->d_name;
                url = join_path(request->path, dp->d_name, NULL);
            }
            temp = (char *) malloc(template_len + parent_path_len + 2*dp->d_reclen);
            sprintf(temp, template, url, display_name);
            saved = send_buffer;
            send_buffer = join_string(send_buffer, temp, NULL);
            free(saved);
            free(temp);
            free(url);
        }
        saved = send_buffer;
        size_t content_len;
        send_buffer = join_string(send_buffer,  "</center></body></html>", &content_len);
        free(saved);

        http_start_response(fd, 200);
        http_send_header(fd, "Content-Type", "text/html");
        dprintf(fd, "Content-Length: %lu\r\n", content_len);
        http_end_headers(fd);

        http_send_string(fd, send_buffer);
        closedir(dirp);
        free(send_buffer);
    } else {
        if (access(full_path, F_OK) == -1) {
            reply_with_file(fd, "./404.html", 404);
            goto EXIT;
        }

        reply_with_file(fd, full_path, 200);
    }
EXIT: close(fd);
}

void file_request_handler(int req_fd) {
    printf("Process %d, thread %lu will handle request.\n", getpid(),  pthread_self());
    handle_files_request(req_fd);
}


/*
 * Opens a connection to the proxy target (hostname=server_proxy_hostname and
 * port=server_proxy_port) and relays traffic to/from the stream fd and the
 * proxy target. HTTP requests from the client (fd) should be sent to the
 * proxy target, and HTTP responses from the proxy target should be sent to
 * the client (fd).
 *
 *   +--------+     +------------+     +--------------+
 *   | client | <-> | httpserver | <-> | proxy target |
 *   +--------+     +------------+     +--------------+
 */
/* create a socket connected to remote server */
int connect_to_host(char *hostname, int port) {
    int fd = socket(PF_INET, SOCK_STREAM, 0);
    if (fd == -1) {
        perror("Failed to create a new socket");
        exit(errno);
    }

    /* get ip address of hostname */
    struct hostent* ent = gethostbyname(hostname);
    if(ent == NULL || ent->h_addr_list == NULL) {
        perror("Failed to get host ip");
        exit(errno);
    }
    struct in_addr** addr = (struct in_addr**)ent->h_addr_list;

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr= *addr[0];
    server_addr.sin_port = htons(port);

    if((connect(fd, (struct sockaddr*)&server_addr, sizeof(struct sockaddr)))!=0) {
        perror("Failed to connect server");
        exit(errno);
    }

    return fd;
}

struct fd_pair {
    int *read_fd;
    int *write_fd;
    char* type;
    unsigned long id;
};

void* relay_message(void* endpoints) {
    struct fd_pair* pair = (struct fd_pair*)endpoints;   
    
    char buffer[4096];
    int read_ret, write_ret;
    // printf("%s thread %lu start to work\n", pair->type, pair->id);
    while((read_ret=read(*pair->read_fd, buffer, sizeof(buffer)-1))>0) {
        write_ret = http_send_data(*pair->write_fd, buffer, read_ret);
        if(write_ret<0) break;
    }
    printf("%s thread %lu exited\n", pair->type, pair->id);
    return NULL;
}

static unsigned long id;
pthread_mutex_t id_mutex = PTHREAD_MUTEX_INITIALIZER;

void proxy_request_handler(int req_fd) {
    unsigned long local_id;

    int target_fd = connect_to_host(server_proxy_hostname, server_proxy_port);

    pthread_mutex_lock(&id_mutex);
    local_id = id++;
    pthread_mutex_unlock(&id_mutex);

    printf("Thread %lu will handle proxy request %lu.\n", pthread_self(), local_id);

    struct fd_pair pairs[2];
    pairs[0].read_fd = &req_fd;
    pairs[0].write_fd = &target_fd;
    pairs[0].type = "request";
    pairs[0].id = local_id;

    pairs[1].read_fd = &target_fd;
    pairs[1].write_fd = &req_fd;
    pairs[1].type = "response";
    pairs[1].id = local_id;

    pthread_t threads[2];
    pthread_create(threads, NULL, relay_message, pairs);
    pthread_create(threads+1, NULL, relay_message, pairs+1);
    pthread_join(threads[0],NULL);
    pthread_join(threads[1],NULL);

    close(req_fd);
    close(target_fd);
    printf("Socket closed, proxy request %lu finished.\n\n", local_id);

}


/* create a server socket so that we can accept client connections */
int create_server_socket(int port) {
    struct sockaddr_in server_address;
    int socket_number = socket(PF_INET, SOCK_STREAM, 0);
    if (socket_number == -1) {
        perror("Failed to create a new socket");
        exit(errno);
    }

    int socket_option = 1;
    if (setsockopt(socket_number, SOL_SOCKET, SO_REUSEADDR, &socket_option,
                sizeof(socket_option)) == -1) {
        perror("Failed to set socket options");
        exit(errno);
    }

    memset(&server_address, 0, sizeof(server_address));
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY;
    server_address.sin_port = htons(port);

    if (bind(socket_number, (struct sockaddr *) &server_address,
                sizeof(server_address)) == -1) {
        perror("Failed to bind on socket");
        exit(errno);
    }
    return socket_number;
}

/*
 * Opens a TCP stream socket on all interfaces with port number PORTNO. Saves
 * the fd number of the server socket in *socket_number. For each accepted
 * connection, calls request_handler with the accepted fd number.
 */
void serve_forever(int *socket_number, void (*request_handler)(int)) {
    struct sockaddr_in client_address;
    size_t client_address_length = sizeof(client_address);
    int client_socket_number;

    *socket_number = create_server_socket(server_port);
    if (listen(*socket_number, 1024) == -1) {
        perror("Failed to listen on socket");
        exit(errno);
    }

    printf("Listening on port %d...\n", server_port);

    while (1) {
        client_socket_number = accept(*socket_number,
                (struct sockaddr *) &client_address,
                (socklen_t *) &client_address_length);
        if (client_socket_number < 0) {
            perror("Error accepting socket");
            continue;
        }

        printf("Accepted connection from %s on port %d\n",
                inet_ntoa(client_address.sin_addr),
                client_address.sin_port);
        async_run(request_handler,client_socket_number);
    }

    shutdown(*socket_number, SHUT_RDWR);
    close(*socket_number);
}

int server_fd;
void signal_callback_handler(int signum) {
    printf("Caught signal %d: %s\n", signum, strsignal(signum));
    printf("Closing socket %d\n", server_fd);
    if (close(server_fd) < 0) perror("Failed to close server_fd (ignoring)\n");
    exit(0);
}

char *USAGE =
"Usage: ./httpserver --files files/ --port 8000 [--num-threads 5]\n"
"       ./httpserver --proxy inst.eecs.berkeley.edu:80 --port 8000 [--num-threads 5]\n";

void exit_with_usage() {
    fprintf(stderr, "%s", USAGE);
    exit(EXIT_SUCCESS);
}

int main(int argc, char **argv) {
    signal(SIGINT, signal_callback_handler);
    signal(SIGPIPE, SIG_IGN);

    /* Default settings */
    server_port = 8000;
    num_threads = 10;
    void (*request_handler)(int) = NULL;

    int i;
    for (i = 1; i < argc; i++) {
        if (strcmp("--files", argv[i]) == 0) {
            server_files_directory = argv[++i];
            if (!server_files_directory) {
                fprintf(stderr, "Expected argument after --files\n");
                exit_with_usage();
            }
        } else if (strcmp("--proxy", argv[i]) == 0) {
            char *proxy_target = argv[++i];
            if (!proxy_target) {
                fprintf(stderr, "Expected argument after --proxy\n");
                exit_with_usage();
            }

            char *colon_pointer = strchr(proxy_target, ':');
            if (colon_pointer != NULL) {
                *colon_pointer = '\0';
                server_proxy_hostname = proxy_target;
                server_proxy_port = atoi(colon_pointer + 1);
            } else {
                server_proxy_hostname = proxy_target;
                server_proxy_port = 80;
            }
        } else if (strcmp("--port", argv[i]) == 0) {
            char *server_port_string = argv[++i];
            if (!server_port_string) {
                fprintf(stderr, "Expected argument after --port\n");
                exit_with_usage();
            }
            server_port = atoi(server_port_string);
        } else if (strcmp("--num-threads", argv[i]) == 0) {
            char *num_threads_str = argv[++i];
            if (!num_threads_str || (num_threads = atoi(num_threads_str)) < 1) {
                fprintf(stderr, "Expected positive integer after --num-threads\n");
                exit_with_usage();
            }
        } else if (strcmp("--help", argv[i]) == 0) {
            exit_with_usage();
        } else {
            fprintf(stderr, "Unrecognized option: %s\n", argv[i]);
            exit_with_usage();
        }
    }

    if (server_files_directory == NULL && server_proxy_hostname == NULL) {
        fprintf(stderr, "Please specify either \"--files [DIRECTORY]\" or \n"
                "                      \"--proxy [HOSTNAME:PORT]\"\n");
        exit_with_usage();
    } else if(server_files_directory != NULL){
        request_handler = file_request_handler;
    } else if(server_proxy_hostname != NULL) {
        request_handler = proxy_request_handler;
    }

    async_init(num_threads);

    serve_forever(&server_fd, request_handler);

    async_destroy();

    return EXIT_SUCCESS;
}
