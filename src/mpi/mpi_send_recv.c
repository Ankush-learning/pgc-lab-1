/*
 * mpi_send_recv.c
 * Basic MPI point-to-point communication test.
 * Rank 0 sends an integer value to Rank 1; Rank 1 receives and prints it.
 *
 * Compile : mpicc -o mpi_send_recv mpi_send_recv.c
 * Run     : mpirun -np 4 --hostfile hosts ./mpi_send_recv
 */

#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    int rank, num_procs;
    int msg_val;
    char node_name[256];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    gethostname(node_name, sizeof(node_name));
    printf("Process rank %d is active on node: %s\n", rank, node_name);

    if (num_procs < 2)
    {
        if (rank == 0)
            printf("At least 2 MPI processes are required to run this test.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0)
    {
        msg_val = 10;
        printf("Rank 0 [%s]: Dispatching msg_val = %d to Rank 1\n", node_name, msg_val);
        MPI_Send(&msg_val, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1)
    {
        MPI_Recv(&msg_val, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank 1 [%s]: Received msg_val = %d from Rank 0\n", node_name, msg_val);
    }

    MPI_Finalize();
    return 0;
}
