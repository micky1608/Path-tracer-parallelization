all:
	(cd part_0_sequentiel ; make)	
	(cd part_1_mpi ; make)
	(cd part_2_omp ; make)
	(cd part_3_simd ; make)

clean: 
	(cd part_0_sequentiel ; make clean)	
	(cd part_1_mpi ; make clean)
	(cd part_2_omp ; make clean)
	(cd part_3_simd ; make clean)
