#include "nnkernels.cuh"

__global__ 
void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__
void hadamard(double *a, double *b)
{
    int i = threadIdx.x;
    a[i] = a[i] * b[i];
}

__global__
void sigmoid(double *zVec, double *activations, double *sps)
{
    int i = threadIdx.x;

    double sig = 1.0 / (1.0 + exp(-zVec[i]));
    activations[i] = sig;
    sps[i] = sig * (1.0 - sig);
}

void had(double *a, double *b, int numElements)
{
    hadamard <<< 1, numElements >>> (a, b);
    return;
}

void sigmoids(double *zVec, double *activations, double *sps, int numElements)
{
    sigmoid <<< 1, numElements >>> (zVec, activations, sps);
    return;
}

void addCUDA(
    vector<int> &c,
    vector<int> &a,
    vector<int> &b
)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    
    size_t size = c.size();
    
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));

    cudaStatus = cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    addKernel <<< 1, size >>>(dev_c, dev_a, dev_b);
    cudaStatus = cudaDeviceSynchronize();

    cudaStatus = cudaMemcpy(c.data(), dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}