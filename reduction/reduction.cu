#include "cuda_runtime.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>


__global__ void reduction(const int* data, const int size, int* output) {

	extern __shared__ int shared_data[];

	int indx = blockIdx.x * blockDim.x + threadIdx.x;
	shared_data[threadIdx.x] = data[indx];
	__syncthreads();
	
	for(int i = 1; i < blockDim.x; i *= 2) {

		if (threadIdx.x % (2*i) == 0) 
			shared_data[threadIdx.x] += shared_data[threadIdx.x + i];
	
		__syncthreads();
	}
	

	if (threadIdx.x == 0)
		output[blockIdx.x] = shared_data[0];
}


__global__ void optimized_reduction(const int* data, const int size, int* output) {

	extern __shared__ int shared_data[];

	int indx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	shared_data[threadIdx.x] = data[indx] + data[indx + blockDim.x * 2];
	__syncthreads();
	
	for(int i = blockDim.x/2; i > 0; i /= 2) {

		if (threadIdx.x < i) 
			shared_data[threadIdx.x] += shared_data[threadIdx.x + i];
	
		__syncthreads();
	}
	

	if (threadIdx.x == 0)
		output[blockIdx.x] = shared_data[0];

}


///////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initialize(int* data, const int size) {

	int indx = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = indx; i < size; i += stride)
		data[i] = 1;
}

//////////////////////////////////////////////////////////////////////////////////////////////


int main() {

	int* data;
	int* result;
	const int size = 1024 * std::pow(2, 16);
	size_t data_size = size * sizeof(int);
	int ID;
	int SM;
	cudaGetDevice(&ID);
	cudaDeviceGetAttribute(&SM, cudaDevAttrMultiProcessorCount, ID);
	cudaMallocManaged(&data, data_size);
	cudaMallocManaged(&result, 32*SM*sizeof(int));
	cudaMemPrefetchAsync(data, data_size, ID);
	cudaMemPrefetchAsync(result, 32*SM*sizeof(int), ID);

	initialize<<<32*SM, 256>>>(data, size);

	auto start_first = std::chrono::high_resolution_clock::now();
	reduction<<<SM * 32, 256, SM*64>>>(data, size, result);
	cudaDeviceSynchronize();
	auto stop_first = std::chrono::high_resolution_clock::now();

	auto start_optimized = std::chrono::high_resolution_clock::now();
	reduction<<<SM * 32, 256>>>(data, size, result);
	cudaDeviceSynchronize();
	auto stop_optimized = std::chrono::high_resolution_clock::now();

	int sum = 0;
	int control = 0;
	for(int i = 0; i < size; i++)
		control += data[i];

	for(int i = 0; i < 32*SM; i++)
		sum += result[i];

	if( sum == size )
		std::cout << "result good" << std::endl;
	std::cout << sum << std::endl;
	std::cout << control << std::endl;
	
	std::ofstream save;
	save.open("time.txt");
//	save << std::chrono::duration_cast<std::chrono::nanoseconds>(stop_first-start_first).count()
//		 << std::endl << std::chrono::duration_cast<std::chrono::nanoseconds>(stop_optimized-start_optimized).count();
	save.close();

	return 0;
}
