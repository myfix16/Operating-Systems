#include <stdlib.h>

typedef struct vector {
	int size;
	int capacity;
	void **data;
} vector;

vector *vector_new(int capacity);

void vector_free(vector *v);

void vector_push(vector *v, void *data);

void *vector_pop(vector *v);

void *vector_get(vector *v, int index);

void vector_set(vector *v, int index, void *data);
