/*
 * matrix_mpi.c
 * Distributed-memory parallel matrix multiplication using MPI.
 * Divides rows of matrix A across MPI ranks, broadcasts B,
 * computes local sub-results, then gathers into full C at rank 0.
 *
 * Compile : mpicc -O2 -o matrix_mpi matrix_mpi.c
 * Run     : mpirun -np 4 --hostfile hosts ./matrix_mpi
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

#define N 4000

int main(int argc, char *argv[])
{
    int rank, num_procs;
    int i, j, k;
    int local_rows;           /* rows handled by this process */
    char node_name[256];

    double *A = NULL;         /* full matrix A  — only on rank 0 */
    double *B = NULL;         /* full matrix B  — all ranks      */
    double *C = NULL;         /* full matrix C  — only on rank 0 */
    double *sub_A;            /* row-slice of A for this rank    */
    double *sub_C;            /* row-slice of C for this rank    */

    double t_start, t_end;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    gethostname(node_name, sizeof(node_name));

    /* N must be evenly divisible by the number of processes */
    if (N % num_procs != 0)
    {
        if (rank == 0)
            printf("Error: matrix size must be divisible by number of processes.\n");

        MPI_Finalize();
        return 0;
    }

    local_rows = N / num_procs;

    sub_A = (double *)malloc(local_rows * N * sizeof(double));
    sub_C = (double *)malloc(local_rows * N * sizeof(double));
    B     = (double *)malloc(N * N * sizeof(double));

    if (rank == 0)
    {
        A = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));

        printf("Allocating and initializing %d x %d matrices on rank 0...\n", N, N);

        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
                C[i * N + j] = 0.0;
            }
        }
    }

    /* Synchronize all processes before timing starts */
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    /* Distribute rows of A and replicate B to every rank */
    MPI_Scatter(A, local_rows * N, MPI_DOUBLE,
                sub_A, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    printf("Rank %d running on %s — processing %d rows\n", rank, node_name, local_rows);

    /* Each rank computes its local portion of C */
    for (i = 0; i < local_rows; i++)
    {
        for (j = 0; j < N; j++)
        {
            sub_C[i * N + j] = 0.0;
            for (k = 0; k < N; k++)
            {
                sub_C[i * N + j] += sub_A[i * N + k] * B[k * N + j];
            }
        }
    }

    /* Collect all partial results back to rank 0 */
    MPI_Gather(sub_C, local_rows * N, MPI_DOUBLE,
               C, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t_end = MPI_Wtime();

    if (rank == 0)
    {
        printf("\nMPI Distributed Matrix Multiplication Done\n");
        printf("Matrix Size     = %d x %d\n", N, N);
        printf("Total MPI Ranks = %d\n", num_procs);
        printf("Elapsed Time    = %f seconds\n", t_end - t_start);
        printf("Verification    — C[0][0] = %.2f\n", C[0]);

        free(A);
        free(C);
    }

    free(B);
    free(sub_A);
    free(sub_C);

    MPI_Finalize();
    return 0;
}
