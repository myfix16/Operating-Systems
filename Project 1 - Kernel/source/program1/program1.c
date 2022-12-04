#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static const char *signal_info[] = {
	"NULL",	       "SIGHUP",      "SIGINT",	     "SIGQUIT",
	"SIGILL",      "SIGTRAP",     "SIGABRT",     "SIGBUS",
	"SIGFPE",      "SIGKILL",     "SIGUSR1",     "SIGSEGV",
	"SIGUSR2",     "SIGPIPE",     "SIGALRM",     "SIGTERM",
	"SIGSTKFLT",   "SIGCHLD",     "SIGCONT",     "SIGSTOP",
	"SIGTSTP",     "SIGTTIN",     "SIGTTOU",     "SIGURG",
	"SIGXCPU",     "SIGXFSZ",     "SIGVTALRM",   "SIGPROF",
	"SIGWINCH",    "SIGIO",	      "SIGPWR",	     "SIGSYS",
	"SIGRTMIN",    "SIGRTMIN+1",  "SIGRTMIN+2",  "SIGRTMIN+3",
	"SIGRTMIN+4",  "SIGRTMIN+5",  "SIGRTMIN+6",  "SIGRTMIN+7",
	"SIGRTMIN+8",  "SIGRTMIN+9",  "SIGRTMIN+10", "SIGRTMIN+11",
	"SIGRTMIN+12", "SIGRTMIN+13", "SIGRTMIN+14", "SIGRTMIN+15",
	"SIGRTMAX-14", "SIGRTMAX-13", "SIGRTMAX-12", "SIGRTMAX-11",
	"SIGRTMAX-10", "SIGRTMAX-9",  "SIGRTMAX-8",  "SIGRTMAX-7",
	"SIGRTMAX-6",  "SIGRTMAX-5",  "SIGRTMAX-4",  "SIGRTMAX-3",
	"SIGRTMAX-2",  "SIGRTMAX-1",  "SIGRTMAX"
};

int main(int argc, char *argv[])
{
	/* fork a child process */
	pid_t pid;
	int status;

	printf("Process start to fork.\n");
	pid = fork();
	if (pid < 0) {
		perror("Fork child process failed.\n");
		exit(EXIT_FAILURE);
	}

	// code for child process
	/* execute test program */
	if (pid == 0) {
		sleep(1);

		// prepare arguments
		char *args[argc];

		printf("I'm the Child Process, my pid = %d\n", getpid());

		// copy commandline arguments
		for (int i = 0; i < argc - 1; ++i) {
			args[i] = argv[i + 1];
		}
		args[argc - 1] = NULL;

		printf("Child process start to execute test program:\n");

		execve(args[0], args, NULL);

		// if current program is still running, throw an error
		perror("execve error");
		exit(EXIT_FAILURE);
	}

	// code for parent process
	else {
		printf("I'm the Parent Process, my pid = %d\n", getpid());

		/* wait for child process terminates */
		waitpid(pid, &status, WUNTRACED);

		/* check child process'  termination status */
		printf("Parent process receives SIGCHLD signal\n");

		if (WIFEXITED(status)) {
			printf("Normal termination with EXIT STATUS = %d.\n",
			       WEXITSTATUS(status));
		} else if (WIFSIGNALED(status)) {
			int sig = WTERMSIG(status);
			printf("child process get %s signal\n",
			       signal_info[sig]);
		} else if (WIFSTOPPED(status)) {
			int sig = WSTOPSIG(status);
			printf("child process get %s signal\n",
			       signal_info[sig]);
		} else {
			printf("child process continued.\n");
		}
		exit(EXIT_SUCCESS);
	}
}
