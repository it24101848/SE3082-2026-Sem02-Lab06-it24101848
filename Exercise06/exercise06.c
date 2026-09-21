#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000
#define STRIP_SIZE 1024

int main()
{
    double *A = (double *)malloc(N * sizeof(double));
    double *B = (double *)malloc(N * sizeof(double));
    double *C = (double *)malloc(N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Memory allocation failed.\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    /* Initialize the arrays */
    for (int i = 0; i < N; i++)
    {
        A[i] = (double)i;
        B[i] = 2.0;
    }

    double start = omp_get_wtime();

    /*
     * Strip-mined parallel multiplication.
     * Each outer-loop iteration represents one strip.
     */
    #pragma omp parallel for schedule(static)
    for (int start_index = 0;
         start_index < N;
         start_index += STRIP_SIZE)
    {
        int end_index = start_index + STRIP_SIZE;

        if (end_index > N)
        {
            end_index = N;
        }

        #pragma omp simd
        for (int i = start_index; i < end_index; i++)
        {
            C[i] = A[i] * B[i];
        }
    }

    double end = omp_get_wtime();

    /* Verify the result */
    int correct = 1;

    for (int i = 0; i < N; i++)
    {
        if (C[i] != A[i] * B[i])
        {
            correct = 0;
            break;
        }
    }

    printf("Array size = %d\n", N);
    printf("Strip size = %d\n", STRIP_SIZE);
    printf("Threads = %d\n", omp_get_max_threads());
    printf("Execution time = %.6f seconds\n", end - start);

    if (correct)
    {
        printf("Verification = PASSED\n");
    }
    else
    {
        printf("Verification = FAILED\n");
    }

    printf("\nSample results:\n");

    for (int i = 0; i < 10; i++)
    {
        printf("A[%d] = %.1f, B[%d] = %.1f, C[%d] = %.1f\n",
               i, A[i], i, B[i], i, C[i]);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
