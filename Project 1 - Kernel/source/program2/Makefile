obj-m	:= program2.o
KVERSION := $(shell uname -r)
PWD := $(shell pwd)
BUILD_DIR := $(PWD)/build
CC := gcc
TEST_CFILES = $(shell ls test | grep .c)
TEST_PROGS = $(addprefix build/, $(patsubst %.c,%,$(TEST_CFILES)))

.PHONY: all clean run testall
all:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(BUILD_DIR) src=$(PWD) modules 

clean:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(BUILD_DIR) src=$(PWD) clean
	-rm -rf build
	-rm /tmp/test

test: test.c
	$(CC) -o $(BUILD_DIR)/test test.c
	cp $(BUILD_DIR)/test /tmp/test
	
run:
	sudo insmod build/program2.ko
	sleep 3
	sudo rmmod build/program2.ko
	sudo dmesg --color=always | tail -n 10
	
testall: all
	$(MAKE) -C ./test all
	rm -f /tmp/test
	for exe in $(TEST_PROGS) ; do \
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"; \
  		if ! cp $$exe /tmp/test; then \
			rm -f /tmp/test; \
			sleep 1; \
			cp $$exe /tmp/test; \
		fi; \
		$(MAKE) run; \
  	done
	$(MAKE) -C ./test clean
	
