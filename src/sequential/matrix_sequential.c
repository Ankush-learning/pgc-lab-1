/*
 * matrix_sequential.c
 * Sequential (single-threaded) matrix multiplication baseline.
 * Computes C = A * B for N x N matrices using a triple-nested loop.
 * Compile: gcc -O2 -o matrix_sequential matrix_sequential.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4000

int main()
{
    int i, j, k;
    double *A, *B, *C;
    clock_t t_start, t_end;

    /* Allocate flat 1-D arrays for all three matrices */
    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    C = (double *)malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    printf("Setting up %d x %d matrices...\n", N, N);

    /* Initialize: A[i][j] = 1.0, B[i][j] = 1.0, C[i][j] = 0.0 */
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }
    }

    t_start = clock();

    /* Triple-nested multiply: C[i][j] += A[i][k] * B[k][j] */
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

    t_end = clock();

    printf("\nSequential Matrix Multiplication Done\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Total Execution Time = %f seconds\n", (double)(t_end - t_start) / CLOCKS_PER_SEC);
    printf("Result Verification — C[0][0] = %.2f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}
