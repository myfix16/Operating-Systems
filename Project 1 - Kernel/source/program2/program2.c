#include <linux/delay.h>
#include <linux/err.h>
#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Xi Mao <119020038@link.cuhk.edu.cn>");
MODULE_DESCRIPTION(
	"It's a simple kernel module that executes a sample program.");

#define WEXITSTATUS(status) ((status & 0xff00) >> 8)
#define WTERMSIG(status) (status & 0x7f)
#define WSTOPSIG(status) (WEXITSTATUS(status))
#define WIFEXITED(status) (WTERMSIG(status) == 0)
#define WIFSIGNALED(status) (((signed char)(((status & 0x7f) + 1) >> 1)) > 0)
#define WIFSTOPPED(status) (((status)&0xff) == 0x7f)

#define NSIG 62

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

// extern kernel functions and structs
struct wait_opts {
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;

	struct waitid_info *wo_info;
	int wo_stat;
	struct rusage *wo_rusage;

	wait_queue_entry_t child_wait;
	int notask_error;
};

// clang-format off
extern int do_execve(struct filename* filename,
                     const char __user* const __user* __argv,
                     const char __user* const __user* __envp);

extern long do_wait(struct wait_opts* wo);
// clang-format on

// function declarations
int my_exec(void);
int my_wait(pid_t pid, int *status);
int my_fork(void *argc);
void check_status(int status);
const char *strsignal(int status);

// implement exec function
int my_exec(void)
{
	const char *filepath = "/tmp/test";
	struct filename *name = getname_kernel(filepath);
	int result;

	printk("[program2] : child process\n");
	result = do_execve(name, NULL, NULL);

	if (result == 0) {
		return 0;
	}
	do_exit(result);
}

// implement wait function
int my_wait(pid_t pid, int *status)
{
	int result;
	struct pid *wo_pid = find_get_pid(pid);
	enum pid_type type = PIDTYPE_PID;
	struct wait_opts wo = {
		.wo_type = type,
		.wo_pid = wo_pid,
		.wo_flags = WEXITED | WUNTRACED,
		.wo_info = NULL,
		// ! wo_stat is an int in 5.10.146, so assigning a point here will be problematic.
		// ! directly retrive wo.wo_stat after do_wait() instead.
		// .wo_stat = status,
		.wo_rusage = NULL
	};

	result = do_wait(&wo);
	put_pid(wo_pid);
	*status = wo.wo_stat;
	return result;
}

// implement fork function
int my_fork(void *argc)
{
	// set default sigaction for current process
	int i, status, result;
	pid_t pid;
	struct k_sigaction *k_action = &current->sighand->action[0];
	struct kernel_clone_args kargs = { .flags = SIGCHLD,
					   .parent_tid = NULL,
					   .child_tid = NULL,
					   .stack = (unsigned long)&my_exec,
					   .stack_size = 0 };

	for (i = 0; i < _NSIG; i++) {
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	/* fork a process using kernel_clone or kernel_thread */
	/* execute a test program in child process */
	pid = kernel_clone(&kargs);
	printk("[program2] : The child process has pid = %d\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n",
	       current->pid);

	/* wait until child process terminates */
	msleep(50);
	result = my_wait(pid, &status);

	check_status(status);

	do_exit(0);
}

// check the return status
void check_status(int status)
{
	if (WIFEXITED(status)) {
		printk("[program2] : Normal termination with EXIT STATUS = %d.\n",
		       WEXITSTATUS(status));
	} else if (WIFSTOPPED(status)) {
		int sig = WSTOPSIG(status);
        printk("[program2] : get %s signal\n", strsignal(sig));
		printk("[program2] : child process stopped\n");
		printk("[program2] : The return signal is: %d\n", sig);
	} else if (WIFSIGNALED(status)) {
		int sig = WTERMSIG(status);
		printk("[program2] : get %s signal\n", strsignal(sig));
		printk("[program2] : child process terminated\n");
		printk("[program2] : The return signal is: %d\n", sig);
	} else {
		printk("[program2] : child process continued\n");
	}
}

static int __init program2_init(void)
{
	struct task_struct *task;

	printk("[program2] : module_init Xi Mao 119020038\n");

	/* write your code here */
	/* create a kernel thread to run my_fork */
	printk("[program2] : module_init create kthread start\n");
	task = kthread_create(&my_fork, NULL, "Program2Thread");

	// wake up new thread if ok
	if (!IS_ERR(task)) {
		printk("[program2] : module_init kthread start\n");
		wake_up_process(task);
	}

	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : module_exit./my\n");
}

// convert signal number to signal name
const char *strsignal(int status)
{
	if (status < 0 || status > NSIG) {
		return "UNKNOWN";
	}
	return signal_info[status];
}

module_init(program2_init);
module_exit(program2_exit);
