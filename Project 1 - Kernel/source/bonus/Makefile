CC:= gcc
CFLAGS:= -std=c11 -Wall -g
DIR_GUARD:= mkdir -p build

all: pstree

pstree: pstree.c pstree.h build/utils.o build/vector.o
	$(DIR_GUARD)
	$(CC) $(CFLAGS) -o build/pstree pstree.c build/utils.o build/vector.o
	
build/utils.o: utils.c utils.h
	$(DIR_GUARD)
	$(CC) $(CFLAGS) -c $< -o $@
	
build/vector.o: vector.c vector.h
	$(DIR_GUARD)
	$(CC) $(CFLAGS) -c $< -o $@
	
.PHONY: clean run
clean:
	rm -rf build
	
run:
	./build/pstree