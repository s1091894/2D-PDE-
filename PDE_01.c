//code by chatgpt
// src/main.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

// 熱傳導係數 α (可由外部編譯參數 -DALPHA=... 修改，預設為 1.0)
#ifndef ALPHA
#define ALPHA 1.0
#endif

// 將 2D 索引 (i,j) 映射到 1D 陣列索引
static inline size_t idx(size_t i, size_t j, size_t nx) { 
    return i*nx + j; 
}

// 初始化：設定高斯分布的初始溫度場
// u(x,y,0) = exp(-((x-x0)^2 + (y-y0)^2) / σ^2)
void init_gaussian(double *u, size_t nx, size_t ny, double h) {
    double x0 = 0.5, y0 = 0.5, s2 = 0.01; // 高斯中心 (0.5,0.5)，σ^2=0.01
    for (size_t i = 0; i < ny; ++i) {
        double y = i*h;
        for (size_t j = 0; j < nx; ++j) {
            double x = j*h;
            double r2 = (x-x0)*(x-x0) + (y-y0)*(y-y0);
            u[idx(i,j,nx)] = exp(-r2/s2);
        }
    }
}

// 設定 Dirichlet 邊界條件 (邊界溫度固定為 val)
void apply_dirichlet(double *u, size_t nx, size_t ny, double val) {
    // 上下邊界
    for (size_t j=0;j<nx;++j){ 
        u[idx(0,j,nx)] = val; 
        u[idx(ny-1,j,nx)] = val; 
    }
    // 左右邊界
    for (size_t i=0;i<ny;++i){ 
        u[idx(i,0,nx)] = val; 
        u[idx(i,nx-1,nx)] = val; 
    }
}

int main(int argc, char** argv) {
    // 預設網格大小 (512x512)，時間步數 2000
    size_t nx = 512, ny = 512; 
    size_t steps = 2000;       
    if (argc >= 3) { nx = (size_t)atol(argv[1]); ny = (size_t)atol(argv[2]); }
    if (argc >= 4) { steps = (size_t)atol(argv[3]); }

    // 模擬區域大小 (Lx, Ly)，預設為 1.0
    double Lx = 1.0, Ly = 1.0;
    double hx = Lx/(nx-1), hy = Ly/(ny-1);
    if (fabs(hx-hy) > 1e-12) { 
        fprintf(stderr, "use square grid for simplicity\n"); 
    }

    // FTCS 穩定性條件: r = α*dt/h^2 <= 1/4
    double h = hx;
    double r = 0.24; // 安全係數 (稍小於 1/4)
    double dt = r * h * h / ALPHA;

    // 配置記憶體 (u: 目前場，un: 下一步場)
    size_t N = nx*ny;
    double *u  = (double*) aligned_alloc(64, N*sizeof(double));
    double *un = (double*) aligned_alloc(64, N*sizeof(double));
    if (!u || !un) { perror("alloc"); return 1; }

    // 初始化場
    init_gaussian(u, nx, ny, h);
    apply_dirichlet(u, nx, ny, 0.0);
    memcpy(un, u, N*sizeof(double));

    // 開始時間
    double t0 = omp_get_wtime();

    // 時間迴圈 (steps 次)
    for (size_t n = 0; n < steps; ++n) {
        // 更新內部節點 (i=1..ny-2, j=1..nx-2)
        #pragma omp parallel for collapse(2)
        for (size_t i = 1; i < ny-1; ++i) {
            for (size_t j = 1; j < nx-1; ++j) {
                double uc = u[idx(i,j,nx)];
                // 五點差分 stencil 計算 Laplacian
                double lap = u[idx(i+1,j,nx)] + u[idx(i-1,j,nx)]
                           + u[idx(i,j+1,nx)] + u[idx(i,j-1,nx)] - 4.0*uc;
                // FTCS 更新公式
                un[idx(i,j,nx)] = uc + (ALPHA*dt/(h*h))*lap;
            }
        }
        // 更新邊界條件
        apply_dirichlet(un, nx, ny, 0.0);
        // 交換指標 (避免複製陣列)
        double *tmp = u; u = un; un = tmp;
    }

    // 結束時間
    double t1 = omp_get_wtime();

    // 計算效能指標 (百萬格點更新率, MLUPS)
    double elapsed = t1 - t0;
    double mlups = (double)((nx*ny)*(unsigned long long)steps) / (elapsed*1e6);
    printf("nx=%zu ny=%zu steps=%zu time=%.3f s MLUPS=%.2f\n", nx, ny, steps, elapsed, mlups);

    // 輸出最後結果到 CSV (方便用 Python/Matlab/Gnuplot 畫圖)
    FILE *f = fopen("heat_final.csv", "w");
    for (size_t i=0;i<ny;++i){
        for (size_t j=0;j<nx;++j){
            fprintf(f, "%lf%c", u[idx(i,j,nx)], (j+1==nx?'\n':','));
        }
    }
    fclose(f);

    // 釋放記憶體
    free(u); free(un);
    return 0;
}
