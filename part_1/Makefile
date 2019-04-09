CC=mpicc -Wall -O3

CFLAGS=-Iinc

LDFLAGS=-lm

BIN=pathtracer

NB_PROC=4

HOST=hostfile408

RUNFLAG=#-hostfile ${HOST}

all : $(BIN)

% : %.c
	$(CC) -o $@ $^ $(LDFLAGS)

clean :
	rm -f $(BIN) *.o *~

exec :
	mpirun -n $(NB_PROC) $(RUNFLAG)  ./$(BIN)


