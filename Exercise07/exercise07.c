#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 2000
#define HEIGHT 2000
#define MAX_ITER 1000

int mandelbrot(double real, double imag)
{
    double z_real = 0.0;
    double z_imag = 0.0;
    int iteration = 0;

    while ((z_real * z_real + z_imag * z_imag <= 4.0) &&
           (iteration < MAX_ITER))
    {
        double temp = z_real * z_real - z_imag * z_imag + real;

        z_imag = 2.0 * z_real * z_imag + imag;
        z_real = temp;

        iteration++;
    }

    return iteration;
}

int main()
{
    int *image = (int *)malloc(WIDTH * HEIGHT * sizeof(int));

    if (image == NULL)
    {
        printf("Memory allocation failed.\n");
        return 1;
    }

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            double real =
                -2.0 + (3.0 * x) / (WIDTH - 1);

            double imag =
                -1.5 + (3.0 * y) / (HEIGHT - 1);

            image[y * WIDTH + x] =
                mandelbrot(real, imag);
        }
    }

    double end = omp_get_wtime();

    long long checksum = 0;

    for (int i = 0; i < WIDTH * HEIGHT; i++)
    {
        checksum += image[i];
    }

    printf("Mandelbrot Set Calculation\n");
    printf("Image size = %d x %d\n", WIDTH, HEIGHT);
    printf("Maximum iterations = %d\n", MAX_ITER);
    printf("Threads = %d\n", omp_get_max_threads());
    printf("Checksum = %lld\n", checksum);
    printf("Execution time = %.6f seconds\n", end - start);

    free(image);

    return 0;
}
