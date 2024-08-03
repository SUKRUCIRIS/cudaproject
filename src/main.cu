#include <stdio.h>

__global__ void add(float *a, float *b, float *c, int count)
{
    int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        c[tindex] = a[tindex] + b[tindex];
    }
}

__managed__ float vectora[32], vectorb[32], vectorc[32];

int main()
{
    for (int i = 0; i < 32; i++)
    {
        vectora[i] = i;
        vectorb[i] = 32 - i;
    }

    add<<<1, 32>>>(vectora, vectorb, vectorc, 32);

    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        printf("CUDA Error.\n");

        return -1;
    }

    for (int i = 0; i < 32; i++)
    {
        printf("%f\n", vectorc[i]);
    }

    return 0;
}