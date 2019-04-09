CC=mpicc -Wall -O3

CFLAGS=-Iinc

LDFLAGS=-lm -fopenmp

BIN=pathtracer

NB_PROC=2

SAMPLE=500

WIDTH=1920
HEIGHT=1080

HOST=hostfile401

RUNFLAG=-hostfile ${HOST}

all : $(BIN)

% : %.c
	$(CC) -o $@ $^ $(LDFLAGS)

clean :
	rm -f $(BIN) *.o *~

exec :
	mpirun -n $(NB_PROC) $(RUNFLAG)  ./$(BIN) $(SAMPLE) $(WIDTH) $(HEIGHT)


