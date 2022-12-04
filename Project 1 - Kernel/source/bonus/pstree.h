#ifndef _PSTREE_H
#define _PSTREE_H
#endif

#include <ctype.h>
#include <stdbool.h>
#include <stdlib.h>
#include <sys/types.h>

// node of the process tree
typedef struct proc_node {
	char name[256];
	pid_t pid;
	pid_t ppid;
	pid_t pgid;
	char flags;
	struct proc_node *parent;
	int num_children;
	struct child_node *children;
	struct proc_node *next;
} proc_node;

typedef struct child_node {
	proc_node *proc;
	int count;
	struct child_node *next;
} child_node;

// pstree formatter
typedef struct format {
	const char *single;
	const char *first;
	const char *branch;
	const char *vertical;
	const char *last;
	const char *blank;
} format;

// read a specific process's information and build a corresponding node.
int read_proc_info(const char *dirname);

// read a process's threads' infomation
int read_thread_info(const char *dirname, const char *tname, pid_t proc_pid,
		     pid_t proc_gid);

// create a proc_node and append it into the global node vector.
proc_node *create_node(const char *procname, pid_t pid, pid_t ppid, pid_t pgid,
		       bool is_thread);

// find the node with given pid.
proc_node *find_node(pid_t pid);

// set child's parent to parent and append child to parent's children list.
void add_child(proc_node *parent, proc_node *child);

// build the process tree (adding parent and child link).
void build_tree(void);

// compare whether two process trees are equal
bool tree_equal(const proc_node *a, const proc_node *b);

// trim a tree's duplicate node
void trim_tree(proc_node *root);

// recursively print the process tree.
void print_tree(proc_node *root, const char *indent, int count, int closing);
