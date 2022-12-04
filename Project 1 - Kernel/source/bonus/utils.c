#include "utils.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

char *trim(const char *str, size_t len, size_t *out_size)
{
	if (len == 0)
		return 0;

	const char *end;

	// Trim leading space
	while (isspace((unsigned char)*str))
		str++;

	if (*str == 0) // All spaces?
	{
		*out_size = 1;
		return NULL;
	}

	// Trim trailing space
	end = str + strlen(str) - 1;
	while (end > str && isspace((unsigned char)*end))
		end--;
	end++;

	// Set output size to minimum of trimmed string length and buffer size minus 1
	*out_size = (end - str) < len - 1 ? (end - str) : len - 1;

	// Copy trimmed string and add null terminator
	char *out = malloc(sizeof(char) * (*out_size + 1));
	memcpy(out, str, *out_size);
	out[*out_size] = 0;

	return out;
}

char *trim_inplace(char *str)
{
	char *end;

	// Trim leading space
	while (isspace((unsigned char)*str))
		str++;

	if (*str == 0) // All spaces?
		return str;

	// Trim trailing space
	end = str + strlen(str) - 1;
	while (end > str && isspace((unsigned char)*end))
		end--;

	// Write new null terminator character
	end[1] = '\0';

	return str;
}