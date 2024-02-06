#include "kernel.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

__device__ void printDev(float* mem, int length) {
  printf("threadIdx.x: %d\n", threadIdx.x);
  printf("blockDim.x: %d\n", blockDim.x);
  printf("threadIdx.y: %d\n", threadIdx.y);

  int global_index = threadIdx.x + blockDim.x * threadIdx.y;
  
   printf("global index: %d\n", global_index);
  for (int i = 0; i < length; ++i)
  {
    printf("mem %d = %f\n", i, mem[i]);
  }
}

__global__ void printDevMem(float *mem, int length)
{
  printDev(mem, length);
}

int main()
{

  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
  }

  thrust::device_vector<float> D{1, 2, 3};

  float *ptr = thrust::raw_pointer_cast(D.data());

  printDevMem<<<2, 4>>>(ptr, 1);

  wrap_test_print();

  cudaDeviceSynchronize();
  return 0;
}