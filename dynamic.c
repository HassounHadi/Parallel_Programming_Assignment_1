#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <mpi.h>

#define MAX_ITERATIONS 1000

int main(int argc, char **argv) {
    int i, j, rank, size;
    double REAL, IMAGINARY;
    int *ComputedMandelbrot;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int w = 800;
    int h = 600;
    int maxX = w - 1;
    int maxY = h - 1;
    double minX = -2.0;
    double minY = -1.0;
    double maxXRange = 3.0;
    double maxYRange = 2.0;
    int chunkSize = (h + size - 1) / size;

    if (rank == 0) {
        ComputedMandelbrot = (int *) malloc(w * h * sizeof(int));
    }

    int *DataQueue = (int *) malloc(chunkSize * w * sizeof(int));

    int startRow = rank * chunkSize;
    int endRow = (rank + 1) * chunkSize;

    if (endRow > h) {
        endRow = h;
    }

    for (i = startRow; i < endRow; i++) {
        for (j = 0; j < w; j++) {
            REAL = (double) (j * maxXRange) / maxX + minX;
            IMAGINARY = (double) (i * maxYRange) / maxY + minY;

            double complex c = REAL + IMAGINARY * I;
            double complex z = 0 + 0 * I;
            int iterations = 0;

            while (cabs(z) < 2 && iterations < MAX_ITERATIONS) {
                z = z * z + c;
                iterations++;
            }

            DataQueue[(i - startRow) * w + j] = iterations;
        }
    }

    // Gather results from each process
    MPI_Gather(DataQueue, chunkSize * w, MPI_INT, ComputedMandelbrot, chunkSize * w, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Print final result
        for (i = 0; i < h; i++) {
            for (j = 0; j < w; j++) {
                printf("%d ", ComputedMandelbrot[i * w + j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}