CC=gcc -Wall -O3

CFLAGS=-Iinc

LDFLAGS=-lm 

BIN=pathtracer_sequentiel

all : $(BIN)

% : %.c
	$(CC) -o $@ $^ $(LDFLAGS)

clean :
	rm -f $(BIN) *.o *~




