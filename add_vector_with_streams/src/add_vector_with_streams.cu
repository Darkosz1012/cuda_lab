/*
 ============================================================================
 Name        : add_vector_with_streams.cu
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <assert.h>
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda.h>
#include <cuda_runtime.h>

#define SIZE 2<<23
#define NOVEC 32
inline cudaError_t checkCUDA(cudaError_t result){

	if(result != cudaSuccess){

		fprintf(stderr, "CUDA Runtime error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}

	return result;
}

__global__ void addVector(float* a, float* b, float* c, int N) {

    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;  i += stride) {

          c[i] = a[i] + b[i];
      }
}



__global__
void init( float *a, int n)
{

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < n; i += stride)
  {
    a[i]=static_cast <float> (i);
  }
}

struct Average{
	double sum = 0;
	int n = 0;
	void operator ()(double x){
		sum+=x;
		n++;
	}
	double get(){
		return n>0?sum/n:0;
	}
};

double addWithStreams(long N){
	std::chrono::system_clock::time_point start;
	std::chrono::system_clock::time_point stop;
	std::chrono::duration<double> elapsed_time;
	int deviceId;
	int numberOfSMs;
	
	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	//const long N = SIZE;
	size_t big = N * 3*NOVEC* sizeof(float);
	float *data;
	cudaMallocManaged(&data, big);
	cudaMemPrefetchAsync(data, big, deviceId);
	cudaStream_t streams[NOVEC];
	start = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < NOVEC; i++){
		size_t threadsPerBlock = 256;
		size_t numberOfBlocks = 32*2;
		cudaStreamCreate(&streams[i]);
		init<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(&data[i*N], N);
		init<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(&data[i*N+N*NOVEC], N);
		addVector<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(&data[i*N],&data[i*N+N*NOVEC],&data[i*N+N*2*NOVEC], N);
		checkCUDA(cudaGetLastError());
	}
	checkCUDA(cudaDeviceSynchronize());
	stop = std::chrono::high_resolution_clock::now();
	elapsed_time = stop - start;
	//std::cout << "with: "<< elapsed_time.count() << std::endl;
	cudaDeviceReset();
	return elapsed_time.count();
}
double addWithoutStreams(long N){
	std::chrono::system_clock::time_point start;
	std::chrono::system_clock::time_point stop;
	std::chrono::duration<double> elapsed_time;
	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	//const long N = SIZE;
	size_t big = N * 3*NOVEC* sizeof(float);
	float *data;
	cudaMallocManaged(&data, big);
	cudaMemPrefetchAsync(data, big, deviceId);
	start = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < NOVEC; i++){
		size_t threadsPerBlock = 256;
		size_t numberOfBlocks = numberOfSMs * 32;
		init<<<numberOfBlocks, threadsPerBlock>>>(&data[i*N], N);
		init<<<numberOfBlocks, threadsPerBlock>>>(&data[i*N+N*NOVEC], N);
		cudaDeviceSynchronize();
		addVector<<<numberOfBlocks, threadsPerBlock>>>(&data[i*N],&data[i*N+N*NOVEC],&data[i*N+N*2*NOVEC], N);
	}
	cudaDeviceSynchronize();
	stop = std::chrono::high_resolution_clock::now();
	elapsed_time = stop - start;
	//std::cout << "without: "<<elapsed_time.count() << std::endl;
	cudaFree(data);
	return elapsed_time.count();
}
double addOneLongVector(long N){
	std::chrono::system_clock::time_point start;
	std::chrono::system_clock::time_point stop;
	std::chrono::duration<double> elapsed_time;
	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	//const long N = SIZE;
	size_t big = N * 3*NOVEC* sizeof(float);
	float *data;
	cudaMallocManaged(&data, big);
	cudaMemPrefetchAsync(data, big, deviceId);
	start = std::chrono::high_resolution_clock::now();

	size_t threadsPerBlock = 256;
	size_t numberOfBlocks = numberOfSMs * 32;
	init<<<numberOfBlocks, threadsPerBlock>>>(data, N*NOVEC);
	init<<<numberOfBlocks, threadsPerBlock>>>(&data[N*NOVEC], N*NOVEC);
	cudaDeviceSynchronize();
	addVector<<<numberOfBlocks, threadsPerBlock>>>(data,&data[N*NOVEC],&data[N*2*NOVEC], N*NOVEC);

	cudaDeviceSynchronize();
	stop = std::chrono::high_resolution_clock::now();
	elapsed_time = stop - start;
	//std::cout << "one long: "<<elapsed_time.count() << std::endl;
	cudaFree(data);
	return elapsed_time.count();
}

int main()
{
	std::cout << "Work"<< std::endl;
	std::ofstream WithFile("with.txt"), WithoutFile("without.txt"), OneFile("one.txt");
	for(long N = 1000000;N <= 28000000;N+=1000000){
		Average with, without, one;
		for(int i = 0 ; i < 3; i++){

			with(addWithStreams(N));
			without(addWithoutStreams(N));
			one(addOneLongVector(N));
		}
		std::cout << "Average with for N = "<< N<<": "<<with.get() << std::endl;
		std::cout << "Average without for N = "<< N<<": "<<without.get() << std::endl;
		std::cout << "Average one for N = "<< N<<": "<<one.get() << std::endl;
		WithFile << N<<"\t"<<with.get() << std::endl;
		WithoutFile << N<<"\t"<<without.get() << std::endl;
		OneFile << N<<"\t"<<one.get() << std::endl;
	}
	WithFile.close();
	WithoutFile.close();
	OneFile.close();
	std::cout << "End"<< std::endl;
}
