CC=gcc -Wall -O3

CFLAGS=-Iinc

LDFLAGS=-lm -fopenmp -mavx2 -march=native

BIN=pathtracer

SAMPLE=50
WIDTH=320
HEIGHT=200

all : $(BIN)

% : %.c
	$(CC) -o $@ $^ $(LDFLAGS)

clean :
	rm -f $(BIN) *.o *~

check:
	valgrind --leak-check=yes ./$(BIN)

exec :
	./$(BIN) $(SAMPLE) $(WIDTH) $(HEIGHT)