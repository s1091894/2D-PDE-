//PDE_01.c
#include <stdio.h>
#include <stdlib.h>  // 包含 aligned_alloc 或 posix_memalign 聲明
#include <math.h>
#include <string.h>
#include <omp.h>

// 熱傳導係數 α (可由外部編譯參數 -DALPHA=... 修改，預設為 1.0)
#ifndef ALPHA
#define ALPHA 1.0
#endif

// 將 2D 索引 (i,j) 映射到 1D 陣列索引
static inline size_t idx(size_t i, size_t j, size_t nx) { 
    return i * nx + j; 
}

// 初始化：設定高斯分布的初始溫度場
void init_gaussian(double *u, size_t nx, size_t ny, double h) {
    double x0 = 0.5, y0 = 0.5, s2 = 0.01;
    for (size_t i = 0; i < ny; ++i) {
        double y = i * h;
        for (size_t j = 0; j < nx; ++j) {
            double x = j * h;
            double r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
            u[idx(i, j, nx)] = exp(-r2 / s2);
        }
    }
    // 調試輸出，檢查初始值
    printf("Initial u[0,0]=%f, u[256,256]=%f\n", u[idx(0, 0, nx)], u[idx(256, 256, nx)]);
}

// 設定 Dirichlet 邊界條件
void apply_dirichlet(double *u, size_t nx, size_t ny, double val) {
    for (size_t j = 0; j < nx; ++j) { 
        u[idx(0, j, nx)] = val; 
        u[idx(ny - 1, j, nx)] = val; 
    }
    for (size_t i = 0; i < ny; ++i) { 
        u[idx(i, 0, nx)] = val; 
        u[idx(i, nx - 1, nx)] = val; 
    }
}

// Thomas 算法求解三對角線性系統
void thomas_algorithm(double *a, double *b, double *c, double *d, double *x, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        if (b[i - 1] == 0.0) {
            printf("Error: Division by zero in Thomas algorithm at i=%zu\n", i);
            return;
        }
        double m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] -= m * d[i - 1];
    }
    x[n - 1] = d[n - 1] / b[n - 1];
    for (int i = (int)n - 2; i >= 0; --i) {
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i];
    }
}

int main(int argc, char** argv) {
    size_t nx = 512, ny = 512; 
    size_t steps = 2000;       
    if (argc >= 3) { nx = (size_t)atol(argv[1]); ny = (size_t)atol(argv[2]); }
    if (argc >= 4) { steps = (size_t)atol(argv[3]); }

    double Lx = 1.0, Ly = 1.0;
    double hx = Lx / (nx - 1), hy = Ly / (ny - 1);
    if (fabs(hx - hy) > 1e-12) { 
        fprintf(stderr, "use square grid for simplicity\n"); 
    }

    double h = hx;
    double dt = 0.001;  // 減小時間步長以提高穩定性
    double rx = ALPHA * dt / (h * h);
    double ry = ALPHA * dt / (h * h);

    size_t N = nx * ny;
    double *u  = (double*) aligned_alloc(64, N * sizeof(double));
    double *un = (double*) aligned_alloc(64, N * sizeof(double));
    if (!u || !un) { perror("alloc"); return 1; }

    init_gaussian(u, nx, ny, h);
    apply_dirichlet(u, nx, ny, 0.0);
    memcpy(un, u, N * sizeof(double));

    double *a = (double*)malloc((nx - 1) * sizeof(double));
    double *b = (double*)malloc((nx - 1) * sizeof(double));
    double *c = (double*)malloc((nx - 1) * sizeof(double));
    double *d = (double*)malloc((nx - 1) * sizeof(double));
    double *x = (double*)malloc((nx - 1) * sizeof(double));

    double t0 = omp_get_wtime();

    for (size_t n = 0; n < steps; ++n) {
        #pragma omp parallel for
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                size_t idx_i = i - 1;  // 調整索引以匹配 Thomas 陣列大小
                a[idx_i] = -0.5 * rx;
                b[idx_i] = 1.0 + rx;
                c[idx_i] = -0.5 * rx;
                d[idx_i] = u[idx(j, i, nx)] + 0.5 * ry * (u[idx(j + 1, i, nx)] - 2.0 * u[idx(j, i, nx)] + u[idx(j - 1, i, nx)]);
            }
            // 調整邊界項的索引
            d[0] += 0.5 * rx * u[idx(j, 0, nx)];
            if (nx > 2) d[nx - 3] += 0.5 * rx * u[idx(j, nx - 1, nx)];
            thomas_algorithm(a, b, c, d, x, nx - 2);
            for (size_t i = 1; i < nx - 1; ++i) {
                un[idx(j, i, nx)] = x[i - 1];
            }
        }
        apply_dirichlet(un, nx, ny, 0.0);
        double *tmp = u; u = un; un = tmp;

        #pragma omp parallel for
        for (size_t i = 1; i < nx - 1; ++i) {
            for (size_t j = 1; j < ny - 1; ++j) {
                size_t idx_j = j - 1;  // 調整索引以匹配 Thomas 陣列大小
                a[idx_j] = -0.5 * ry;
                b[idx_j] = 1.0 + ry;
                c[idx_j] = -0.5 * ry;
                d[idx_j] = u[idx(j, i, nx)] + 0.5 * rx * (u[idx(j, i + 1, nx)] - 2.0 * u[idx(j, i, nx)] + u[idx(j, i - 1, nx)]);
            }
            // 調整邊界項的索引
            d[0] += 0.5 * ry * u[idx(0, i, nx)];
            if (ny > 2) d[ny - 3] += 0.5 * ry * u[idx(ny - 1, i, nx)];
            thomas_algorithm(a, b, c, d, x, ny - 2);
            for (size_t j = 1; j < ny - 1; ++j) {
                un[idx(j, i, nx)] = x[j - 1];
            }
        }
        apply_dirichlet(un, nx, ny, 0.0);
        tmp = u; u = un; un = tmp;
    }

    double t1 = omp_get_wtime();

    double elapsed = t1 - t0;
    double mlups = (double)((nx * ny) * (unsigned long long)steps) / (elapsed * 1e6);
    printf("nx=%zu ny=%zu steps=%zu time=%.3f s MLUPS=%.2f\n", nx, ny, steps, elapsed, mlups);

    FILE *f = fopen("heat_final.csv", "w");
    if (f) {
        for (size_t i = 0; i < ny; ++i) {
            for (size_t j = 0; j < nx; ++j) {
                double val = u[idx(i, j, nx)];
                if (isnan(val)) val = 0.0; // 處理 nan
                fprintf(f, "%.6f%c", val, (j + 1 == nx ? '\n' : ','));
            }
        }
        fclose(f);
    } else {
        perror("fopen");
    }

    // 執行 Python 腳本
    char command[256];
#ifdef _WIN32
    sprintf(command, "python \"%s\"", "PDE_01.py");
#else
    sprintf(command, "python3 \"%s\"", "PDE_01.py");
#endif
    system(command);

    // 釋放記憶體
    free(a); free(b); free(c); free(d); free(x);
    free(u); free(un);  // 使用 free 釋放 aligned_alloc 分配的記憶體
    return 0;
}
