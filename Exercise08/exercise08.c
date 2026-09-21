#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 2000
#define STEPS 100
#define DT 0.001
#define SOFTENING 0.01

typedef struct
{
    double x;
    double y;
    double vx;
    double vy;
    double fx;
    double fy;
} Particle;

int main()
{
    Particle *particles =
        (Particle *)malloc(N * sizeof(Particle));

    if (particles == NULL)
    {
        printf("Memory allocation failed.\n");
        return 1;
    }

    /* Initialize particles */
    for (int i = 0; i < N; i++)
    {
        particles[i].x = (double)(i % 100) * 0.01;
        particles[i].y = (double)(i / 100) * 0.01;

        particles[i].vx = 0.0;
        particles[i].vy = 0.0;

        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }

    double start = omp_get_wtime();

    for (int step = 0; step < STEPS; step++)
    {
        /*
         * Calculate force on each particle.
         * Each thread owns a different i value, so each thread
         * writes only to particles[i].fx and particles[i].fy.
         */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
        {
            double fx = 0.0;
            double fy = 0.0;

            for (int j = 0; j < N; j++)
            {
                if (i != j)
                {
                    double dx =
                        particles[j].x - particles[i].x;

                    double dy =
                        particles[j].y - particles[i].y;

                    double distance2 =
                        dx * dx + dy * dy + SOFTENING;

                    double inverseDistance =
                        1.0 / sqrt(distance2);

                    double inverseDistance3 =
                        inverseDistance *
                        inverseDistance *
                        inverseDistance;

                    fx += dx * inverseDistance3;
                    fy += dy * inverseDistance3;
                }
            }

            particles[i].fx = fx;
            particles[i].fy = fy;
        }

        /*
         * Update velocity and position.
         */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
        {
            particles[i].vx += particles[i].fx * DT;
            particles[i].vy += particles[i].fy * DT;

            particles[i].x += particles[i].vx * DT;
            particles[i].y += particles[i].vy * DT;
        }
    }

    double end = omp_get_wtime();

    /*
     * Calculate checksum for result verification.
     */
    double checksum = 0.0;

    for (int i = 0; i < N; i++)
    {
        checksum +=
            particles[i].x +
            particles[i].y +
            particles[i].vx +
            particles[i].vy;
    }

    printf("Molecular Dynamics Simulation\n");
    printf("Particles = %d\n", N);
    printf("Simulation steps = %d\n", STEPS);
    printf("Threads = %d\n", omp_get_max_threads());
    printf("Checksum = %.10f\n", checksum);
    printf("Execution time = %.6f seconds\n",
           end - start);

    printf("\nSample particle results:\n");

    for (int i = 0; i < 5; i++)
    {
        printf(
            "Particle %d: x = %.6f, y = %.6f, "
            "vx = %.6f, vy = %.6f\n",
            i,
            particles[i].x,
            particles[i].y,
            particles[i].vx,
            particles[i].vy);
    }

    free(particles);

    return 0;
}
