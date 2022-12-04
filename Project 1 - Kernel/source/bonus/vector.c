#include "vector.h"
#include <stdio.h>

vector *vector_new(int capacity)
{
	vector *v = malloc(sizeof(vector));
	v->capacity = capacity;
	v->size = 0;
	v->data = malloc(capacity * sizeof(void *));
	return v;
}

void vector_free(vector *v)
{
	free(v->data);
	free(v);
}

void vector_push(vector *v, void *data)
{
	if (v->size == v->capacity) {
		v->capacity *= 2;
		v->data = realloc(v->data, v->capacity * sizeof(void *));
	}
	v->data[v->size++] = data;
}

void *vector_pop(vector *v)
{
	return v->data[--v->size];
}

void *vector_get(vector *v, int index)
{
	if (index >= v->size) {
		perror("index out of bounds");
		return NULL;
	}
	return v->data[index];
}

void vector_set(vector *v, int index, void *data)
{
	if (index >= v->size) {
		perror("index out of bounds");
		return;
	}
	v->data[index] = data;
}