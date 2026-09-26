/*
 * matrix_openmp.c
 * Shared-memory parallel matrix multiplication using OpenMP.
 * Distributes outer loop iterations across available CPU threads.
 * Compile: gcc -O2 -fopenmp -o matrix_openmp matrix_openmp.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4000

int main()
{
    int i, j, k;
    double *A, *B, *C;
    double t_start, t_end;

    /* Allocate memory for all three matrices */
    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    C = (double *)malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    /* Initialize matrices: A = 1.0, B = 1.0, C = 0.0 */
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }
    }

    t_start = omp_get_wtime();

    /* Parallelize the outermost loop; j and k are thread-private */
    #pragma omp parallel for private(j, k)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
            {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    t_end = omp_get_wtime();

    printf("OpenMP Parallel Matrix Multiplication Done\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Active Thread Count = %d\n", omp_get_max_threads());
    printf("Wall-Clock Time = %f seconds\n", t_end - t_start);
    printf("Result Verification — C[0][0] = %.2f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}
