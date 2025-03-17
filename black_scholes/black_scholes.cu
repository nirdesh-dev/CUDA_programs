#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 1000000  // 1 million options

// Normal cumulative distribution function (CNDF)
__device__ double normal_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

// Black-Scholes kernel running on the GPU
__global__ void black_scholes_kernel(double *d_S, double *d_K, double *d_T, 
                                     double r, double sigma, 
                                     double *d_call, double *d_put) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        double S = d_S[idx];
        double K = d_K[idx];
        double T = d_T[idx];

        double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
        double d2 = d1 - sigma * sqrt(T);

        d_call[idx] = S * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2);
        d_put[idx] = K * exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1);
    }
}

int main() {
    // Allocate memory on host (CPU)
    double *h_S = (double*)malloc(N * sizeof(double));
    double *h_K = (double*)malloc(N * sizeof(double));
    double *h_T = (double*)malloc(N * sizeof(double));
    double *h_call = (double*)malloc(N * sizeof(double));
    double *h_put = (double*)malloc(N * sizeof(double));

    double r = 0.01 + ((double)rand() / RAND_MAX) * 0.1;  // Random risk-free rate (1% to 11%)
    double sigma = 0.1 + ((double)rand() / RAND_MAX) * 0.5;  // Random volatility (10% to 60%)

    // Initialize input data with randomized values
    for (int i = 0; i < N; i++) {
        h_S[i] = 50.0 + ((double)rand() / RAND_MAX) * 200.0;  // Stock price (50 to 250)
        h_K[i] = 50.0 + ((double)rand() / RAND_MAX) * 200.0;  // Strike price (50 to 250)
        h_T[i] = 0.1 + ((double)rand() / RAND_MAX) * 5.0;  // Time to expiry (0.1 to 5 years)
    }

    // Allocate memory on GPU
    double *d_S, *d_K, *d_T, *d_call, *d_put;
    cudaMalloc((void**)&d_S, N * sizeof(double));
    cudaMalloc((void**)&d_K, N * sizeof(double));
    cudaMalloc((void**)&d_T, N * sizeof(double));
    cudaMalloc((void**)&d_call, N * sizeof(double));
    cudaMalloc((void**)&d_put, N * sizeof(double));

    // Copy data to GPU
    cudaMemcpy(d_S, h_S, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, h_T, N * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel with optimized grid size
    int threads_per_block = 1024;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    black_scholes_kernel<<<blocks_per_grid, threads_per_block>>>(d_S, d_K, d_T, r, sigma, d_call, d_put);

    // Copy results back to CPU
    cudaMemcpy(h_call, d_call, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_put, d_put, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Write results to file
    FILE *file = fopen("output.txt", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(file, "Stock Price | Strike Price | Time to Expiry | Call Price | Put Price\n");
    fprintf(file, "-------------------------------------------------------------\n");
    for (int i = 0; i < N; i++) {
        fprintf(file, "%11.2f | %12.2f | %14.2f | %10.5f | %9.5f\n", 
                h_S[i], h_K[i], h_T[i], h_call[i], h_put[i]);
    }
    fclose(file);

    printf("Results written to output.txt\n");

    // Free GPU and CPU memory
    cudaFree(d_S);
    cudaFree(d_K);
    cudaFree(d_T);
    cudaFree(d_call);
    cudaFree(d_put);
    free(h_S);
    free(h_K);
    free(h_T);
    free(h_call);
    free(h_put);

    return 0;
}
