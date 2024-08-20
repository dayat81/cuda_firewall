#include "stdio.h"
#define N 100

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}
__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}
int main() {
    printf("Hello World from CPU!\n");
    cuda_hello<<<1,1>>>(); 
    // Allocate memory
    float *a, *b, *out; 
    float *d_a, *d_b, *d_out;
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }
    // Allocate device memory for a
    cudaMalloc((void**)&d_a, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);

   // Allocate device memory for b
    cudaMalloc((void**)&d_b, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

   // Allocate device memory for out
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_out, out, sizeof(float) * N, cudaMemcpyHostToDevice);        

    vector_add<<<1,1>>>(d_out, d_a, d_b, N);
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);  
    for(int i = 0; i < N; i++){
        printf("%f \n",out[i]);
    }
    // Cleanup after kernel execution
    cudaFree(d_a);
    free(a);
    cudaFree(d_b);
    free(b);
    cudaFree(d_out);
    free(out);

    return 0;
}