#include "pstree.h"

#include <assert.h>
#include <dirent.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>

#include "utils.h"
#include "vector.h"

#define PROC_DIR "/proc"
#define ROOT_PID (1)

#ifndef DT_DIR // gcc 11.2 in ubuntu 22.04 doesn't detect this macro
#define DT_DIR (4)
#endif

#define CHECK_ALLOC_FAIL(ptr)                                                  \
	if (ptr == NULL) {                                                     \
		perror("malloc failed");                                       \
		exit(EXIT_FAILURE);                                            \
	}

#define PFLAG_THREAD 0x02
#define THREAD_FORMAT "{%s}" // Format for thread names

/* UTF-8 defines by Johan Myreen, updated by Ben Winslow */
#define UTF_V "\342\224\202" /* U+2502, Vertical line drawing char */
#define UTF_VR "\342\224\234" /* U+251C, Vertical and right */
#define UTF_H "\342\224\200" /* U+2500, Horizontal */
#define UTF_UR "\342\224\224" /* U+2514, Up and right */
#define UTF_HD "\342\224\254" /* U+252C, Horizontal and down */

format ascii_formatter = { "---", "-+-", " |-", " | ", " `-", "   " };
format utf_formatter = {
	.single = UTF_H UTF_H UTF_H,
	.first = UTF_H UTF_HD UTF_H,
	.branch = " " UTF_VR UTF_H,
	.vertical = " " UTF_V " ",
	.last = " " UTF_UR UTF_H,
	.blank = "   ",
};
static format formatter;

// command line options
static int compact = 1, use_unicode = 0, pids = 0, pgids = 0, hide_threads = 0;
static int root_pid = ROOT_PID;

static proc_node *head;
static vector *roots = NULL;

proc_node *create_node(const char *procname, pid_t pid, pid_t ppid, pid_t pgid,
		       bool is_thread)
{
	proc_node *node = (proc_node *)malloc(sizeof(proc_node));
	CHECK_ALLOC_FAIL(node)

	node->pid = pid;
	node->ppid = ppid;
	node->pgid = pgid;
	node->flags = 0;
	if (is_thread) {
		node->flags |= PFLAG_THREAD;
		sprintf(node->name, THREAD_FORMAT, procname);
	} else
		strcpy(node->name, procname);
	node->num_children = 0;
	node->children = NULL;
	node->parent = NULL;

	node->next = head;
	head = node;

	return node;
}

proc_node *find_node(pid_t pid)
{
	for (proc_node *node = head; node != NULL; node = node->next) {
		if (node->pid == pid)
			return node;
	}
	return NULL;
}

void add_child(proc_node *parent, proc_node *child)
{
	child_node *cnode = (child_node *)malloc(sizeof(child_node));
	CHECK_ALLOC_FAIL(cnode)

	cnode->proc = child;
	cnode->count = 1;
	cnode->next = parent->children;
	parent->children = cnode;
	child->parent = parent;
	parent->num_children++;
}

void build_tree(void)
{
	roots = vector_new(16);

	for (proc_node *node = head; node != NULL; node = node->next) {
		proc_node *pnode = find_node(node->ppid);
		// if a node has a parent, add it to the parent's children list
		if (pnode != NULL) {
			add_child(pnode, node);
		}
		// else, it is the root (or orphan) node
		else {
			vector_push(roots, node);
		}
	}
}

bool tree_equal(const proc_node *a, const proc_node *b)
{
	if (a == NULL || b == NULL)
		return false;
	if (a->num_children != b->num_children)
		return false;
	if (a->flags != b->flags)
		return false;
	if (strcmp(a->name, b->name) != 0)
		return false;

	for (child_node *walk_a = a->children, *walk_b = b->children;
	     walk_a && walk_b; walk_a = walk_a->next, walk_b = walk_b->next)
		if (!tree_equal(walk_a->proc, walk_b->proc))
			return false;

	return true;
}

void trim_tree(proc_node *root)
{
	if (root == NULL)
		return;

	// recursively trim children
	child_node *child = root->children;
	while (child != NULL) {
		trim_tree(child->proc);
		child = child->next;
	}

	// compare children and remove duplicates
	child = root->children;
	while (child != NULL) {
		child_node *current = child;
		child_node *next = child->next;
		while (next != NULL) {
			if (tree_equal(child->proc, next->proc)) {
				child->count++;
				current->next = next->next;
				free(next);
				next = current->next;
			} else {
				current = current->next;
				next = next->next;
			}
		}
		child = child->next;
	}
}

void print_tree(proc_node *root, const char *indent, const int count,
		int closing)
{
	/* the implementation is based on pstree src code and
       https://rivers-shall.github.io/2019/03/05/NJU-OS-M1-pstree/ */

	assert(closing >= 0);
	if (count > 1)
		++closing;

	char comm[256];
	if (compact) {
		if (count > 1)
			printf("%d*[", count);
		strcpy(comm, root->name);
	} else {
		if (pids && !pgids)
			sprintf(comm, "%s(%d)", root->name, root->pid);
		else if (!pids && pgids)
			sprintf(comm, "%s(%d)", root->name, root->pgid);
		else if (pids && pgids)
			sprintf(comm, "%s(%d,%d)", root->name, root->pid,
				root->pgid);
		else
			strcpy(comm, root->name);
	}

	printf("%s", comm);
	char *new_indent =
		(char *)malloc(strlen(indent) + strlen(comm) + 4 + 1 +
			       10); // 3 for formatter, 1 for terminator

	if (compact)
		trim_tree(root);

	if (root->children == NULL) { // no children
		while (closing--)
			putchar(']');
		puts("");
		free(new_indent);
		return;
	} else if (root->children->next == NULL) { // one child
		printf("%s", formatter.single);
		sprintf(new_indent, "%s%s%*s", indent, formatter.blank,
			(int)strlen(comm), "");
		print_tree(root->children->proc, new_indent,
			   root->children->count, closing);
	} else { // multiple children
		child_node *current = root->children;
		printf("%s", formatter.first);
		sprintf(new_indent, "%s%*s%s", indent, (int)strlen(comm), "",
			formatter.vertical);
		print_tree(current->proc, new_indent, current->count, closing);

		for (current = current->next; current->next != NULL;
		     current = current->next) {
			printf("%s%*s%s", indent, (int)strlen(comm), "",
			       formatter.branch);
			sprintf(new_indent, "%s%*s%s", indent,
				(int)strlen(comm), "", formatter.vertical);
			print_tree(current->proc, new_indent, current->count,
				   closing);
		}

		printf("%s%*s%s", indent, (int)strlen(comm), "",
		       formatter.last);
		sprintf(new_indent, "%s%s%*s", indent, formatter.blank,
			(int)strlen(comm), "");
		print_tree(current->proc, new_indent, current->count, closing);
	}

	free(new_indent);
}

int read_proc_info(const char *dirname)
{
	// try to open the file: /proc/{PID}/status
	char filename[256];
	sprintf(filename, "%s/status", dirname);
	FILE *file_ptr = fopen(filename, "r");
	if (file_ptr == NULL) {
		return EXIT_FAILURE;
	}

	// read the first few lines to get process name, pid, and parent pid
	char buffer[256], procname[256];
	int pid, ppid, pgid;
	for (int i = 0; i < 15;
	     i++) { // there are 15 lines to read before we can get all information need
		if (fgets(buffer, sizeof(buffer), file_ptr) == NULL) {
			break;
		}

		char *key = strtok(buffer, ":"), *value = strtok(NULL, ":");
		if (key != NULL && value != NULL) {
			key = trim_inplace(key);
			value = trim_inplace(value);
			if (strcmp(key, "Pid") == 0) {
				pid = atoi(value);
			} else if (strcmp(key, "PPid") == 0) {
				ppid = atoi(value);
			} else if (strcmp(key, "NSpgid") == 0) {
				pgid = atoi(value);
			} else if (strcmp(key, "Name") == 0) {
				strcpy(procname, value);
			}
		}
	}

	create_node(procname, pid, ppid, pgid, false);

	// close the process state file
	fclose(file_ptr);

	// handle threads
	if (!hide_threads) {
		char task_dirname[256];
		sprintf(task_dirname, "%s/task", dirname);
		DIR *dir_ptr = NULL;
		struct dirent *dir_entry = NULL;
		if ((dir_ptr = opendir(task_dirname)) == NULL) {
			perror("Error: unable to open /proc/PID/task directory");
			exit(EXIT_FAILURE);
		}

		// read and parse thread files one by one
		while ((dir_entry = readdir(dir_ptr)) != NULL) {
			if (dir_entry->d_type == DT_DIR &&
			    isdigit(dir_entry->d_name[0])) {
				if (atoi(dir_entry->d_name) == pid)
					continue;

				sprintf(filename, "%s/%s", task_dirname,
					dir_entry->d_name);
				read_thread_info(filename, procname, pid, pgid);
			}
		}

		closedir(dir_ptr);
	}

	return 0;
}

int read_thread_info(const char *dirname, const char *tname, pid_t proc_pid,
		     pid_t proc_gid)
{
	// try to open the file: /proc/{PID}/task/{TID}/status
	char filename[256];
	sprintf(filename, "%s/status", dirname);
	FILE *file_ptr = fopen(filename, "r");
	if (file_ptr == NULL) {
		return EXIT_FAILURE;
	}

	// read the first few lines to get process name, pid, and parent pid
	char buffer[256];
	int pid;
	for (int i = 0; i < 7;
	     i++) { // there are 7 lines to read before we can get all information need
		if (fgets(buffer, sizeof(buffer), file_ptr) == NULL) {
			break;
		}

		char *key = strtok(buffer, ":"), *value = strtok(NULL, ":");
		if (key != NULL && value != NULL) {
			key = trim_inplace(key);
			value = trim_inplace(value);
			if (strcmp(key, "Pid") == 0) {
				pid = atoi(value);
			}
		}
	}

	create_node(tname, pid, proc_pid, proc_gid, true);

	// close the process state file
	fclose(file_ptr);
	return 0;
}

void parse_options(int argc, char **argv)
{
	/* options implemented:
            PID: show the tree of the process with the given PID
            -p: show pid
            -g: show pgid
            -c: don't compact identical subtrees
            -T: hide threads
            -U: use utf-8 characters to draw lines
     */

	struct option options[] = { { "compact-not", 0, NULL, 'c' },
				    { "show-pids", 0, NULL, 'p' },
				    { "show-pgids", 0, NULL, 'g' },
				    { "hide-threads", 0, NULL, 'T' },
				    { "unicode", 0, NULL, 'U' },
				    { 0, 0, 0, 0 } };

	int c;
	while ((c = getopt_long(argc, argv, "cpgTU", options, NULL)) != -1) {
		switch (c) {
		case 'c':
			compact = 0;
			break;
		case 'p':
			pids = 1;
			compact = 0;
			break;
		case 'g':
			pgids = 1;
			compact = 0;
			break;
		case 'T':
			hide_threads = 1;
			break;
		case 'U':
			use_unicode = 1;
			break;
		}
	}
	if (optind == argc - 1) {
		if (isdigit(argv[optind][0])) {
			root_pid = atoi(argv[optind]);
		}
	}

	formatter = use_unicode ? utf_formatter : ascii_formatter;
}

int main(int argc, char **argv)
{
	// parse arguments
	parse_options(argc, argv);

	// traverse directories in /proc to find and create process nodes
	DIR *dir_ptr = NULL;
	struct dirent *dir_entry = NULL;
	char dirname[256];
	if ((dir_ptr = opendir(PROC_DIR)) == NULL) {
		perror("Error: unable to open /proc directory");
		exit(EXIT_FAILURE);
	}

	// read and parse process files one by one
	while ((dir_entry = readdir(dir_ptr)) != NULL) {
		if (dir_entry->d_type == DT_DIR &&
		    isdigit(dir_entry->d_name[0])) {
			sprintf(dirname, "%s/%s", PROC_DIR, dir_entry->d_name);
			read_proc_info(dirname);
		}
	}

	// create the process tree
	build_tree();

	// find the root node and fix orphans
	proc_node *root = find_node(root_pid);
	/* custom pid */
	if (root_pid != ROOT_PID) {
		if (root == NULL) {
			fprintf(stderr, "Error: process %d not found\n",
				root_pid);
			exit(EXIT_FAILURE);
		}
	}
	/* default pid */
	else {
		if (root == NULL) {
			root = create_node("?", ROOT_PID, 0, 0, false);
		}
		for (int i = 0; i < roots->size; i++) {
			proc_node *node = (proc_node *)vector_get(roots, i);
			if (node->ppid == 0)
				continue;
			if (node->parent == NULL) {
				add_child(root, node);
			}
		}
	}

	// print the tree
	print_tree(root, "", 1, 0);

	// clean up
	vector_free(roots);
	closedir(dir_ptr);

	return EXIT_SUCCESS;
}