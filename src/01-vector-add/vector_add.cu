#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float* a, float* b, float* c, size_t length) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    size_t length = 98432;

    srand(time(NULL));
    // host vectors
    float *h_a = (float*)malloc(length * sizeof(float));
    float *h_b = (float*)malloc(length * sizeof(float));
    float *h_c = (float*)malloc(length * sizeof(float));
    float *h_c_ref = (float*)malloc(length * sizeof(float));

    // initialize host vectors
    for (size_t i = 0; i < length; i++) {
        h_a[i] = rand();
        h_b[i] = rand();
        h_c_ref[i] = h_a[i] + h_b[i];
    }

    // device vectors
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;

    // allocate device memory
    cudaMalloc((void**)&d_a, length * sizeof(float));
    cudaMalloc((void**)&d_b, length * sizeof(float));
    cudaMalloc((void**)&d_c, length * sizeof(float));

    // copy host vectors to device
    cudaMemcpy(d_a, h_a, length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, length * sizeof(float), cudaMemcpyHostToDevice);

    // kernel function
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, length);

    // copy device vectors to host
    cudaMemcpy(h_c, d_c, length * sizeof(float), cudaMemcpyDeviceToHost);

    // check result
    bool success = true;
    for (size_t i = 0; i < length; i++) {
        if (fabs(h_c[i] - h_c_ref[i]) > 1e-6) {
            printf("Error at index %zu: %f != %f\n", i, h_c[i], h_c_ref[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Vector addition completed successfully! Processed %zu elements.\n", length);
    }

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_ref);

    return 0;
}