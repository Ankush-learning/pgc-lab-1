/*
 * matrix_cuda.cu
 * GPU-accelerated matrix multiplication using CUDA.
 * Launches a 2-D thread grid where each thread computes one element of C.
 *
 * Compile : nvcc -O2 -o matrix_cuda matrix_cuda.cu
 * Run     : ./matrix_cuda
 *
 * Grid  : (N/16) x (N/16) blocks  → 250 x 250 = 62 500 blocks
 * Block : 16 x 16 threads          →              256 threads/block
 * Total : 16 000 000 concurrent GPU threads
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 4000

/* Each GPU thread computes one output element C[row][col] */
__global__ void matMulKernel(float *A, float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float accum = 0.0f;

        for (int k = 0; k < n; k++)
        {
            accum += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = accum;
    }
}

int main()
{
    size_t mem_size = N * N * sizeof(float);

    /* Host (CPU) pointers */
    float *h_A, *h_B, *h_C;
    /* Device (GPU) pointers */
    float *d_A, *d_B, *d_C;

    h_A = (float *)malloc(mem_size);
    h_B = (float *)malloc(mem_size);
    h_C = (float *)malloc(mem_size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        printf("Host memory allocation failed\n");
        return 1;
    }

    /* Initialise: all A and B elements = 1.0, C = 0.0 */
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }

    /* Allocate GPU device memory */
    cudaMalloc((void **)&d_A, mem_size);
    cudaMalloc((void **)&d_B, mem_size);
    cudaMalloc((void **)&d_C, mem_size);

    /* Set up CUDA events for precise timing */
    cudaEvent_t ev_total_start, ev_total_stop;
    cudaEvent_t ev_kernel_start, ev_kernel_stop;

    cudaEventCreate(&ev_total_start);
    cudaEventCreate(&ev_total_stop);
    cudaEventCreate(&ev_kernel_start);
    cudaEventCreate(&ev_kernel_stop);

    cudaEventRecord(ev_total_start);

    /* Transfer input data: host → device (H2D) */
    cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice);

    /* Configure 2-D thread hierarchy */
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaEventRecord(ev_kernel_start);

    /* Launch the GPU kernel */
    matMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaEventRecord(ev_kernel_stop);
    cudaEventSynchronize(ev_kernel_stop);

    /* Transfer results back: device → host (D2H) */
    cudaMemcpy(h_C, d_C, mem_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(ev_total_stop);
    cudaEventSynchronize(ev_total_stop);

    float kernel_ms = 0.0f;
    float total_ms  = 0.0f;

    cudaEventElapsedTime(&kernel_ms, ev_kernel_start, ev_kernel_stop);
    cudaEventElapsedTime(&total_ms,  ev_total_start,  ev_total_stop);

    printf("CUDA Matrix Multiplication Done\n");
    printf("Matrix Size         = %d x %d\n", N, N);
    printf("Grid Dimensions     = %d x %d blocks\n", grid.x, grid.y);
    printf("Block Dimensions    = %d x %d threads\n", block.x, block.y);
    printf("Kernel Time         = %.6f seconds\n", kernel_ms / 1000.0f);
    printf("Total CUDA Time     = %.6f seconds\n", total_ms  / 1000.0f);
    printf("Result Verification — C[0][0] = %.2f\n", h_C[0]);

    /* Release GPU memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    /* Release host memory */
    free(h_A);
    free(h_B);
    free(h_C);

    cudaEventDestroy(ev_total_start);
    cudaEventDestroy(ev_total_stop);
    cudaEventDestroy(ev_kernel_start);
    cudaEventDestroy(ev_kernel_stop);

    return 0;
}
