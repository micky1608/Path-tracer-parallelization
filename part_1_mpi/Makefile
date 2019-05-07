CC=mpicc -Wall -O3

CFLAGS=-Iinc

LDFLAGS=-lm

BIN=pathtracer

NB_PROC=4

SAMPLE=50
WIDTH=320
HEIGHT=200

HOST=hostfile

RUNFLAG=

#si il existe un fichier hostfile
ifneq ("$(wildcard hostfile)","")
	RUNFLAG=-hostfile ${HOST}
	N=$(shell wc -l hostfile | grep -o -e [0-9]*)
	NB_PROC=$(shell echo $$((2*$N)))
endif

all : $(BIN)

% : %.c
	$(CC) -o $@ $^ $(LDFLAGS)

clean :
	rm -f $(BIN) *.o *~

check:
	valgrind --leak-check=yes ./$(BIN)

exec :
	mpirun -n $(NB_PROC) $(RUNFLAG) ./$(BIN) $(SAMPLE) $(WIDTH) $(HEIGHT)


