CC=gcc -Wall -O3

CFLAGS=-Iinc

LDFLAGS=-lm -fopenmp -mavx2 -march=native

BIN=pathtracer

all : $(BIN)

% : %.c
	$(CC) -o $@ $^ $(LDFLAGS)

clean :
	rm -f $(BIN) *.o *~




