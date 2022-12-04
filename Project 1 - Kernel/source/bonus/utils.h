#ifndef _UTILS_H
#define _UTILS_H
#endif

#include <stdlib.h>

// trim a string's leading and tailing spaces, and return a new string
char *trim(const char *str, size_t len, size_t *out_size);

// trim a string's leading and tailing spaces in place
char *trim_inplace(char *str);
