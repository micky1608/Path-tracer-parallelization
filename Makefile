CC=mpicc -Wall -O3

CFLAGS=-Iinc

LDFLAGS=-lm 

BIN=pathtracer

NB_PROC=2

all : $(BIN)

% : %.c
	$(CC) -o $@ $^ $(LDFLAGS)

clean :
	rm -f $(BIN) *.o *~

exec :
	mpirun -n $(NB_PROC) ./$(BIN)


