CC=mpicc -Wall -O3

CFLAGS=-Iinc

LDFLAGS=-lm -fopenmp -mavx2 -march=native

BIN=pathtracer

NB_PROC=4

SAMPLE=120

WIDTH=320
HEIGHT=200

HOST=hostfile401

RUNFLAG=#-hostfile ${HOST}

all : $(BIN)

% : %.c
	$(CC) -o $@ $^ $(LDFLAGS)

clean :
	rm -f $(BIN) *.o *~

check:
	valgrind --leak-check=yes ./$(BIN)

exec :
	mpirun -n $(NB_PROC) $(RUNFLAG)  ./$(BIN) $(SAMPLE) $(WIDTH) $(HEIGHT)


