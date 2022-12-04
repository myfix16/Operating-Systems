CFILES:= $(shell ls|grep .c)
PROGS:=$(patsubst %.c,%,$(CFILES))

all: $(PROGS)

%:%.c
	mkdir -p ../build
	$(CC) -o ../build/$@ $<

.PHONY: clean

clean:
	rm -f $(addprefix ../build/, $(PROGS))
