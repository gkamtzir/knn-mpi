/**
* FILE: knn_mpi_no_communications.c
* THMMY, 7th semester, Parallel and Distributed Systems: 2st assignment
* Distributed KNN algorithm.
* This implementation focuses on the time spent on calculations only.
* Author:
*   Kamtziridis Georgios, 8542, gkamtzir@auth.gr
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

int get_most_frequent_element(int *array, int size);

void calculate_distances(double **source, double **target, int* id, double **distances,
	int **distances_ids, int n, int d, int k, int count);

int main(int argc, char **argv)
{

	int s;
	int n;
	int d;
	int k;

	FILE *file;
	char data_file_name[64];
	char labels_file_name[64];

	int file_size;
	int data_per_process;
	int maximum_number_of_dimensions;

	int *source_id;
	int *target_id;
	double **source;
	double **target;
	double **distances;
	int ** distances_ids;

	//MPI variables.
	int rank, size, previous, next, tag = 5;

	if (argc != 5)
	{
		printf("You have to pass 4 parameters: \n");
		printf("1)s <# of dataset (1 -> 10.000 elements (784 dimensions each), 2-> 60.000 (30 dimensions each))\n");
		printf("2)n <# of points>\n 3)d <# of dimensions>\n 4)k <# of neighbors>\n");
		exit(1);
	}

	//Getting user's input.
	s = atoi(argv[1]);
	n = atoi(argv[2]);
	d = atoi(argv[3]);
	k = atoi(argv[4]);

	if (s != 1 && s != 2)
	{

		printf("Invalid dataset number\n");
		exit(1);

	}
	else
	{

		if (s == 1)
		{

			if (n <= 10000 && n > 0 && d <= 784 && d > 0 && k <= 50 && k > 0)
			{

				sprintf(data_file_name, "data.bin");
				sprintf(labels_file_name, "labels.bin");
				maximum_number_of_dimensions = 784;

			} else {

				printf("Invalid parameters\n");
				exit(1);

			}

		}
		else
		{

			if (n <= 60000 && n > 0 && d <= 30 && d > 0 && k <= 50 && k > 0)
			{

				sprintf(data_file_name, "data_svd.bin");
				sprintf(labels_file_name, "labels_svd.bin");
				maximum_number_of_dimensions = 30;

			} else {

				printf("Invalid parameters\n");
				exit(1);

			}

		}

	}


	//Time variables.
	struct timeval startwtime, endwtime;
	//Counting the number of data exchanges.
	int count;

	//Initializing MPI.
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (n % size != 0)
	{

		printf("The data could not be divided evenly.");
		printf("Please try a different combination of processes - points\n");
		exit(1);

	}

	data_per_process = n / size;

	next = (rank + 1) % size;
	previous = (rank + size - 1) % size;

	//Allocating memory.
	source_id = (int *)malloc(data_per_process * sizeof(int));
	target_id = (int *)malloc(n * sizeof(int));
	source = (double **)malloc(data_per_process * sizeof(double*));
	distances = (double **)malloc(data_per_process * sizeof(double*));
	distances_ids = (int **)malloc(data_per_process * sizeof(int*));

	//Allocating serialized memory.
	double *target_data = (double *)malloc(d * n * sizeof(double));
	target = (double **)malloc(n * sizeof(double *));
	for (int i = 0; i < n; i++)
    	target[i] = &(target_data[d * i]);


	if (source == NULL || distances == NULL || source_id == NULL || target_id == NULL
		|| distances_ids == NULL || target == NULL)
	{
		printf("Error! Memory not allocated.");
		exit(1);
	}

	for (int i = 0; i < data_per_process; i++)
	{

		source[i] = (double *)malloc(d * sizeof(double));
		distances[i] = (double *)malloc(k * sizeof(double));
		distances_ids[i] = (int *)malloc(k * sizeof(int));

		if (source[i] == NULL || distances[i] == NULL || distances_ids[i] == NULL
			|| target == NULL)
		{

				printf("Error! Memory not allocated.");
				exit(1);

		}


		for (int j = 0; j < k; j++)
		{

			distances[i][j] = -1.0;
			distances_ids[i][j] = -1;

		}

	}

	//Reading data from the file.
	file = fopen(data_file_name, "rb");

	if (file == NULL)
	{

		printf("Could not open file\n");
		exit(1);

	}

	fseek(file, rank * data_per_process * sizeof(double) * maximum_number_of_dimensions, SEEK_CUR);
	for (int i = 0; i < data_per_process; i++)
	{

		fread(source[i], sizeof(double), d, file);
		fseek(file, (maximum_number_of_dimensions - d) * sizeof(double), SEEK_CUR);

	}
	fclose(file);

	file = fopen(data_file_name, "rb");

	if (file == NULL)
	{

		printf("Could not open file\n");
		exit(1);

	}

	for (int i = 0; i <  n; i++)
	{

		fread(target[i], sizeof(double), d, file);
		fseek(file, (maximum_number_of_dimensions - d) * sizeof(double), SEEK_CUR);

	}
	fclose(file);

	//Reading labels from the file.
	file = fopen(labels_file_name, "rb");

	if (file == NULL)
	{

		printf("Could not open file\n");
		exit(1);

	}

	fseek(file, rank * data_per_process * sizeof(int), SEEK_CUR);
	fread(target_id, sizeof(int), data_per_process, file);
	fclose(file);

	file = fopen(labels_file_name, "rb");

	if (file == NULL)
	{

		printf("Could not open file\n");
		exit(1);

	}

	fseek(file, rank * data_per_process * sizeof(int), SEEK_CUR);
	fread(source_id, sizeof(int), data_per_process, file);
	fclose(file);

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0)
	{

  	gettimeofday (&startwtime, NULL);

	}

	count = 0;

	while(1)
	{

		//Calculating the k-nearest neighbors.
		calculate_distances(source, target, target_id, distances, distances_ids, data_per_process, d, k, count);

		//Checking if the process has finished.
		if (count == (size-1))
		{

			printf("Process %d has finished \n", rank);
			break;

		}

		count++;

	}

	MPI_Barrier(MPI_COMM_WORLD);

	//Printing the runtime.
	if (rank == 0)
	{

 		gettimeofday (&endwtime, NULL);
  		double exec_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
         	 + endwtime.tv_sec - startwtime.tv_sec);
		printf("Runtime: %.4f \n", exec_time);

	}

	//Freeing memory.
	for (int i = 0; i < data_per_process; i++)
	{

		free(source[i]);
		free(distances[i]);
		free(distances_ids[i]);

	}

	free(source);
	free(source_id);

	free(distances);
	free(distances_ids);

	free(target[0]);
	free(target_id);

	free(target);
	
	MPI_Finalize();

	return 0;

}

void calculate_distances(double **source, double **target, int* id, double **distances, int **distances_ids, int n, int d, int k, int count)
{

	for (int i = 0; i < n; i++)
	{
		double distance;
		int position;

		for (int j = count * n; j < count * n + n; j++)
		{

			if (i == j) continue;

			distance = 0.0;

			for (int l = 0; l < d; l++)
			{

				distance += (double)(source[i][l] - target[j][l]) * (double)(source[i][l] - target[j][l]);

			}

			position = -1;

			for (int u = 0; u < k; u++)
			{

				if (distances[i][u] > distance || distances[i][u] == -1.0)
				{

					position = u;
					break;

				}

			}

			if (position != -1)
			{

				for (int w = k - 2; w > position - 1; w--)
				{

					distances[i][w + 1] = distances[i][w];
					distances_ids[i][w + 1] = distances_ids[i][w];

				}

				distances[i][position] = distance;
				distances_ids[i][position] = id[j];

			}

		}

	}

}
