CFILES:= $(shell ls|grep .c)
PROGS:=$(patsubst %.c,%,$(CFILES))

all: $(PROGS)

%:%.c
	mkdir -p build
	$(CC) -o build/$@ $<

.PHONY: clean test testall

clean:
	rm -rf build

test:
	./build/program1 ./build/$(tgt)

testall: all
	for exe in $(shell find build -not -type d -not -name program1) ; do \
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"; \
  		./build/program1 $$exe; \
  	done
