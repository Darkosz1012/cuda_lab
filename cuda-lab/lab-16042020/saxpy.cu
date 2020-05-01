#include <stdio.h>
#include <chrono>
#include <fstream>

#define N 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively
 * and use profiler to check your progress
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 25us.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

__global__ void saxpy(float * a, float * b, float * c)
{
    int tid = blockIdx.x * blockDim.x * threadIdx.x;

    if ( tid < N )
        c[tid] = 2 * a[tid] + b[tid];
}

__global__ void saxpy_second(float* a, float* b, float* c) {
	
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;  i += stride) {

          c[i] = 2 * a[i] + b[i];
      }
}



__global__
void initWith(float num, float *a, int n)
{

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < n; i += stride)
  {
    a[i] = num;
  }
}



int main()
{
    float *a, *b, *c;

    int size = N * sizeof (int); // The total number of bytes per vector

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // Initialize memory
    for( int i = 0; i < N; ++i )
    {
        a[i] = 2;
        b[i] = 1;
        c[i] = 0;
    }

    int threads_per_block = 128;
    int number_of_blocks = (N / threads_per_block) + 1;

    saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c );
    
    // Print out the first and last 5 values of c for a quality check
    printf("The first kernel test:\n");
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    //second kernel with grid optimizations
    int SMs;
    cudaDeviceGetAttribute(&SMs, cudaDevAttrMultiProcessorCount, 0);

////Some code for testing only
//	std::ofstream save;
//	save.open("time_data.txt");
//
////
   for(int i = 1; i <= 10; i++ ){
	
//	auto start = std::chrono::high_resolution_clock::now();
        
//	saxpy_second<<<32*SMs,256>>>(a, b, c);
//	saxpy<<<128, 512>>>(a, b, c);
        saxpy<<<number_of_blocks,threads_per_block>>>(a, b, c);
	
//	initWith<<<128, 512>>>(0, c, N);

//	cudaDeviceSynchronize();
	
//	auto stop = std::chrono::high_resolution_clock::now();
	
//	save << i << "\t" << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << std::endl;
	
	}

//	save.close();
    
    printf("\nOptimised code quality check: \n");   
    //Check values, like with the first kernel
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");


    cudaFree( a ); cudaFree( b ); cudaFree( c );
}

