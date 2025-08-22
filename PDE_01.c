// src/main.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#ifndef ALPHA
#define ALPHA 1.0
#endif

static inline size_t idx(size_t i, size_t j, size_t nx) { return i*nx + j; }

void init_gaussian(double *u, size_t nx, size_t ny, double h) {
    double x0 = 0.5, y0 = 0.5, s2 = 0.01; // sigma^2
    for (size_t i = 0; i < ny; ++i) {
        double y = i*h;
        for (size_t j = 0; j < nx; ++j) {
            double x = j*h;
            double r2 = (x-x0)*(x-x0) + (y-y0)*(y-y0);
            u[idx(i,j,nx)] = exp(-r2/s2);
        }
    }
}

void apply_dirichlet(double *u, size_t nx, size_t ny, double val) {
    for (size_t j=0;j<nx;++j){ u[idx(0,j,nx)] = val; u[idx(ny-1,j,nx)] = val; }
    for (size_t i=0;i<ny;++i){ u[idx(i,0,nx)] = val; u[idx(i,nx-1,nx)] = val; }
}

int main(int argc, char** argv) {
    size_t nx = 512, ny = 512; // grid size
    size_t steps = 2000;       // time steps
    if (argc >= 3) { nx = (size_t)atol(argv[1]); ny = (size_t)atol(argv[2]); }
    if (argc >= 4) { steps = (size_t)atol(argv[3]); }

    double Lx = 1.0, Ly = 1.0;
    double hx = Lx/(nx-1), hy = Ly/(ny-1);
    if (fabs(hx-hy) > 1e-12) { fprintf(stderr, "use square grid for simplicity\n"); }

    // FTCS stability: r = alpha*dt/h^2 <= 1/4
    double h = hx;
    double r = 0.24; // safety margin
    double dt = r * h * h / ALPHA;

    size_t N = nx*ny;
    double *u  = (double*) aligned_alloc(64, N*sizeof(double));
    double *un = (double*) aligned_alloc(64, N*sizeof(double));
    if (!u || !un) { perror("alloc"); return 1; }

    init_gaussian(u, nx, ny, h);
    apply_dirichlet(u, nx, ny, 0.0);

    memcpy(un, u, N*sizeof(double));

    double t0 = omp_get_wtime();
    for (size_t n = 0; n < steps; ++n) {
        // interior update (i=1..ny-2, j=1..nx-2)
        #pragma omp parallel for collapse(2)
        for (size_t i = 1; i < ny-1; ++i) {
            for (size_t j = 1; j < nx-1; ++j) {
                double uc = u[idx(i,j,nx)];
                double lap = u[idx(i+1,j,nx)] + u[idx(i-1,j,nx)]
                           + u[idx(i,j+1,nx)] + u[idx(i,j-1,nx)] - 4.0*uc;
                un[idx(i,j,nx)] = uc + (ALPHA*dt/(h*h))*lap;
            }
        }
        apply_dirichlet(un, nx, ny, 0.0);
        // swap
        double *tmp = u; u = un; un = tmp;
    }
    double t1 = omp_get_wtime();

    double elapsed = t1 - t0;
    double mlups = (double)((nx*ny)*(unsigned long long)steps) / (elapsed*1e6);
    printf("nx=%zu ny=%zu steps=%zu time=%.3f s MLUPS=%.2f\n", nx, ny, steps, elapsed, mlups);

    // dump last frame to CSV (for quick plotting)
    FILE *f = fopen("heat_final.csv", "w");
    for (size_t i=0;i<ny;++i){
        for (size_t j=0;j<nx;++j){
            fprintf(f, "%lf%c", u[idx(i,j,nx)], (j+1==nx?'\n':','));
        }
    }
    fclose(f);

    free(u); free(un);
    return 0;
}
