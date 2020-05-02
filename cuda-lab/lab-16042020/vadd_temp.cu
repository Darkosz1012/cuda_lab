// example of using CUDA streams

#include <stdio.h>

__global__
void initWith(float num, float *a, int N)
{

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;
	float* d;
	float* e;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);
	cudaMallocManaged(&d, size);
	cudaMallocManaged(&e, size);


  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(c, size, deviceId);
	cudaMemPrefetchAsync(d, size, deviceId);
	cudaMemPrefetchAsync(e, size, deviceId);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = 32 * numberOfSMs;

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  /*
   * Create 3 streams to run initialize the 3 data vectors in parallel.
   */

  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);

  /*
   * Give each `initWith` launch its own non-standard stream.
   */

  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(3, a, N);
  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(4, b, N);
  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream3>>>(0, c, N);
  //initWith<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);
  //initWith<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);
  //initWith<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);
  
	
  //addVectorsInto<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(c, a, b, N/3);
  //addVectorsInto<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(c+N/3,a+N/3, b+N/3, N/3 + 1);
  //addVectorsInto<<<numberOfBlocks, threadsPerBlock, 0, stream3>>>(c+2*N/3, a+2*N/3, b+2*N/3, N/3 + 1); 
	addVectorsInto<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(c, a, b, N);
	addVectorsInto<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(d, a, b, N);
	addVectorsInto<<<numberOfBlocks, threadsPerBlock, 0, stream3>>>(e, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);

  checkElementsAre(7, c, N);

  /*
   * Destroy streams when they are no longer needed.
   */

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream3);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

