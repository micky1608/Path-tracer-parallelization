CC=mpicc -Wall -O3 -fopenmp -mavx2 -march=native

#Pour desactiver la vectorisation automatique 
#CC=mpicc -Wall -O2 -fopenmp -mavx2 -march=native -fgcse-after-reload -finline-functions -fipa-cp-clone -floop-interchange -floop-unroll-and-jam -fpeel-loops -fpredictive-commoning -fsplit-paths -ftree-loop-distribute-patterns -ftree-loop-distribution -ftree-partial-pre -funswitch-loops -fno-tree-vectorize

LDFLAGS=-lm 
BIN=pathtracer

NB_PROC=2

SAMPLE=50
WIDTH=320
HEIGHT=200

HOST=hostfile

RUNFLAG=

MAP=

#si il existe un fichier hostfile
ifneq ("$(wildcard ${HOST})","")
	RUNFLAG=-hostfile ${HOST}
	N=$(shell wc -l ${HOST} |  grep -o -e [0-9]*)
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
	mpirun -n $(NB_PROC) $(RUNFLAG) $(MAP) ./$(BIN) $(SAMPLE) $(WIDTH) $(HEIGHT)


