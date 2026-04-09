CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -DGGML_TQ_STANDALONE -lm
LDFLAGS = -lm

.PHONY: all clean test

all: test-tq

test-tq: test-tq.c ggml-tq.c ggml-tq.h
	$(CC) $(CFLAGS) -o $@ test-tq.c ggml-tq.c $(LDFLAGS)

test: test-tq
	./test-tq

clean:
	rm -f test-tq
