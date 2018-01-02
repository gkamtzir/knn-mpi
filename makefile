CC=mpicc

all: knn_mpi knn_mpi_non_blocking knn_mpi_no_communications knn_mpi_mp knn_mpi_non_blocking_mp knn_mpi_no_communications_mp

knn_mpi: knn_mpi.c
	$(CC) -std=c99 -O3 -o mpi knn_mpi.c

knn_mpi_mp: knn_mpi_mp.c
	$(CC) -std=c99 -O3 -o mpi_mp knn_mpi_mp.c -fopenmp

knn_mpi_non_blocking: knn_mpi_non_blocking.c
	$(CC) -std=c99 -O3 -o mpi_non knn_mpi_non_blocking.c

knn_mpi_non_blocking_mp: knn_mpi_non_blocking_mp.c
	$(CC) -std=c99 -O3 -o mpi_non_mp knn_mpi_non_blocking_mp.c -fopenmp

knn_mpi_no_communications: knn_mpi_no_communications.c
	$(CC) -std=c99 -O3 -o mpi_no_com knn_mpi_no_communications.c

knn_mpi_no_communications_mp: knn_mpi_no_communications_mp.c
	$(CC) -std=c99 -O3 -o mpi_no_com_mp knn_mpi_no_communications_mp.c -fopenmp

clean:
	rm mpi mpi_mp mpi_non mpi_non_mp mpi_no_com mpi_no_com_mp
