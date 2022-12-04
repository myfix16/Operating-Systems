#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <fcntl.h>
#include <pthread.h>
#include <termios.h>
#include <unistd.h>


namespace utils {
    int rand(int min, int max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(min, max);
        return dis(gen);
    }
}

#define MUTEX_LOCK_AND_DO(mutex, command) \
    pthread_mutex_lock((mutex));          \
    command;                              \
    pthread_mutex_unlock((mutex));

/// clear the screen
#define CLEAR_SCREEN std::cout << "\033c"

struct Node {
    int x, y;
    Node(int _x, int _y) : x(_x), y(_y) {};
    Node() = default;
} frog;

/* global configurations */
constexpr int ROW = 10;
constexpr int COLUMN = 50;
constexpr int COL_ACTUAL = COLUMN - 1;  //! important: map[row][COLUMN] == '\0'
constexpr int LOG_LEN_BASE = 8;
constexpr int LOG_LEN_VAR = 15;
constexpr int FRAME_INTERVAL = 100000;  // time between two game frames
const std::unordered_map<char, std::pair<int, int>> key_map = {
    {'w', {-1, 0}},
    {'a', {0, -1}},
    {'s', {1, 0}},
    {'d', {0, 1}},
    {'W', {-1, 0}},
    {'A', {0, -1}},
    {'S', {1, 0}},
    {'D', {0, 1}}
};

/* runtime game variables */
char map[ROW + 10][COLUMN]{};
// the length of the wood on each row
int wood_length[ROW];
// the (left edge) position of the wood on each row
int wood_position[ROW];
// game status.
enum Status { Play = 0, Win = 1, Lose = 2, Quit = 3 } game_status = Play;

/* pthreads */
pthread_mutex_t game_mutex, frog_mutex, map_mutex;


/// Determine a keyboard is hit or not. If yes, return 1. If not, return 0.
int kbhit() {
    struct termios oldt{}, newt{};
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);

    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);

    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

/// Clear previous output and print the new map
void print_map() {
    CLEAR_SCREEN;
    // print new map
    for (int i = 0; i <= ROW; ++i) std::cout << map[i] << '\n';
    std::cout << std::endl;
}

void update_game_status() {
    // get the frog's position
    Node frog_pos = frog;
    char map_frog = map[frog_pos.x][frog_pos.y];

    /* update game status */
    // frog reaches the other side
    if (frog_pos.x == 0) {
        MUTEX_LOCK_AND_DO(&game_mutex, game_status = Status::Win)
    }
    // frog falls into water or leaves map boundary
    else if (map_frog == ' ' || frog_pos.x < 0 || frog_pos.x > ROW
             || frog_pos.y < 0 || frog_pos.y > COLUMN - 2) {
        MUTEX_LOCK_AND_DO(&game_mutex, game_status = Status::Lose)
    }
}

/// Read keyboard input and move the frog
void* frog_ctrl(void* args) {
    std::pair<int, int> direction;
    char key;
    Node new_pos{};
    while (game_status == Play) {
        /* process keyboard input */
        if (kbhit()) {
            std::cin >> key;
            if (key == 'q' || key == 'Q') {
                MUTEX_LOCK_AND_DO(&game_mutex, game_status = Status::Quit);
                break;
            }
            if (key_map.find(key) != key_map.end()){
                direction = key_map.at(key);
                //! ignore invalid movements, i.e. movements that make the frog out of world
                new_pos = Node(frog.x + direction.first, frog.y + direction.second);
                if (new_pos.x < 0 || new_pos.x > ROW || new_pos.y < 0 || new_pos.y >= COL_ACTUAL)
                    continue;
            }

            // move the frog and update game status
            MUTEX_LOCK_AND_DO(&map_mutex, map[frog.x][frog.y] = (frog.x == 0 || frog.x == ROW) ? '|' : '=')
            MUTEX_LOCK_AND_DO(&frog_mutex,
                frog.x = new_pos.x;
                frog.y = new_pos.y;
            )
            update_game_status();
            if (game_status == Status::Play || game_status == Status::Win){
                MUTEX_LOCK_AND_DO(&map_mutex, map[frog.x][frog.y] = '0')
            }
        }
    }
    pthread_exit(nullptr);
}

/// Get `index`'s corresponding valid column index of the map
#define ACTUAL_INDEX(index) (((index) + COL_ACTUAL) % COL_ACTUAL)

void* logs_move(void* t) {
    const long id = reinterpret_cast<long>(t);
    const int len = wood_length[id];
    const int delta_pos = (id % 2) ? -1 : 1;
    int start_pos = wood_position[id];
    int end_pos = start_pos + len - 1;

    while (game_status == Play) {
        MUTEX_LOCK_AND_DO(&frog_mutex, Node frog_pos = frog)

        /* move the wood: odd indexed left, even indexed right*/
        pthread_mutex_lock(&map_mutex);
        char* row = map[id];
        // clear the tailing '='
        row[ACTUAL_INDEX((id % 2) ? end_pos : start_pos)] = ' ';
        // update start and end position
        start_pos = ACTUAL_INDEX(start_pos + delta_pos);
        end_pos = start_pos + len - 1;
        // draw the starting '=
        row[ACTUAL_INDEX((id % 2) ? start_pos : end_pos)] = '=';
        start_pos = ACTUAL_INDEX(start_pos);

        /* if frog is on board, update frog's position */
        if (frog_pos.x == id && (frog_pos.y >= start_pos || frog_pos.y <= end_pos)) {
            // clear the original frog
            row[frog_pos.y] = '=';
            MUTEX_LOCK_AND_DO(&frog_mutex, frog.y += delta_pos)
            update_game_status();
            // put a new frog if necessary
            if (game_status == Status::Play) row[frog_pos.y + delta_pos] = '0';
        }
        pthread_mutex_unlock(&map_mutex);

        usleep(FRAME_INTERVAL);
    }
    pthread_exit(nullptr);
}

/// While the game not ends, update one frame and sleep for a small interval
void* refresh_screen(void* args) {
    while (game_status == Play) {
        MUTEX_LOCK_AND_DO(&map_mutex, print_map())
        usleep(FRAME_INTERVAL);
    }

    // display the final status for a while
    if (game_status == Status::Win) {
        print_map();
        usleep(FRAME_INTERVAL);
    }

    pthread_exit(nullptr);
}

int main(int argc, char* argv[]) {
    /* Initialize the river map and frog's starting position */
    frog = Node(ROW, (COLUMN - 1) / 2);
    for (int i = 1; i < ROW; ++i)
        std::fill_n(map[i], COLUMN - 1, ' ');
    for (int j = 0; j < COLUMN - 1; ++j)
        map[ROW][j] = map[0][j] = '|';

    map[frog.x][frog.y] = '0';

    /* Initialize wood's length and position */
    for (int i = 1; i < ROW; i++) {
        int wood_len = utils::rand(0, LOG_LEN_VAR) + LOG_LEN_BASE;
        int wood_pos = utils::rand(0, COLUMN - wood_len - 1);
        wood_length[i] = wood_len;
        wood_position[i] = wood_pos;
        // update the map
        std::fill_n(&map[i][wood_pos], wood_len, '=');
    }

    /* Print the map into screen */
    print_map();

    /*  Create pthreads for wood move and frog control.  */
    pthread_mutex_init(&game_mutex, nullptr);
    pthread_mutex_init(&map_mutex, nullptr);
    pthread_t frog_ctrl_thread, screen_render_thread;
    pthread_t log_move_threads[ROW - 1];

    // initialize threads
    if (pthread_create(&frog_ctrl_thread, nullptr, frog_ctrl, nullptr))
        throw std::runtime_error("Failed to create thread for frog control.");
    if (pthread_create(&screen_render_thread, nullptr, refresh_screen, nullptr))
        throw std::runtime_error("Failed to create thread for game flow.");
    for (long id = 1; id < ROW; ++id) {
        if (pthread_create(&log_move_threads[id - 1], nullptr, logs_move, reinterpret_cast<void*>(id)))
            throw std::runtime_error("Failed to create threads for log movement.");
    }

    // sync threads
    for (pthread_t& log_move_thread : log_move_threads) {
        if (pthread_join(log_move_thread, nullptr))
            throw std::runtime_error("Failed to join threads for log movement.");
    }
    if (pthread_join(frog_ctrl_thread, nullptr))
        throw std::runtime_error("Failed to join thread for frog control.");
    if (pthread_join(screen_render_thread, nullptr))
        throw std::runtime_error("Failed to join thread for game flow.");

    /*  Display the output for user: win, lose or quit.  */
    CLEAR_SCREEN;
    switch (game_status) {
        case Status::Win:
            std::cout << "You win :D" << std::endl;
            break;
        case Status::Lose:
            std::cout << "You lose :(" << std::endl;
            break;
        case Status::Quit:
            std::cout << "You quit the game." << std::endl;
            break;
        default:
            throw std::logic_error("Invalid game status: " + std::to_string(game_status) + '\n');
    }

    /* Clean up and exit */
    pthread_mutex_destroy(&game_mutex);
    pthread_mutex_destroy(&map_mutex);
    pthread_mutex_destroy(&frog_mutex);

    return 0;
}
