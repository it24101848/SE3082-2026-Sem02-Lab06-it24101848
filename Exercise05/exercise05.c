#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int fib(int n)
{
    int i, j;

    if (n < 2)
    {
        return n;
    }
    else
    {
        #pragma omp task shared(i)
        i = fib(n - 1);

        #pragma omp task shared(j)
        j = fib(n - 2);

        #pragma omp taskwait

        return i + j;
    }
}

int main(int argc, char *argv[])
{
    int n = 20;
    int result = 0;

    if (argc > 1)
    {
        n = atoi(argv[1]);
    }

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            result = fib(n);
        }
    }

    double end = omp_get_wtime();

    printf("Fibonacci(%d) = %d\n", n, result);
    printf("Threads = %d\n", omp_get_max_threads());
    printf("Execution time = %.6f seconds\n", end - start);

    return 0;
}
