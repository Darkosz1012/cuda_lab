#include "cuda_runtime.h"

#include <iostream>
#include <fstream>
#include <array>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <functional>


__global__ void main_histogram(const float* input, const long size, int* histogram, int binsNumber) {

	int indx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	__shared__ int shared_bins[2048];

	//for(int i = indx; i < binsNumber; i += blockDim.x)
	//	shared_bins[i] = 0;
	shared_bins[threadIdx.x] = 0;
	__syncthreads();

	for(int i = indx; i < size; i += stride)
		atomicAdd(&shared_bins[ static_cast<int>(input[i]) ], 1);
	__syncthreads();

	//for(int i = indx; i < binsNumber; i += blockDim.x)
	//	atomicAdd(&histogram[i], shared_bins[i]);
	atomicAdd(&histogram[threadIdx.x], shared_bins[threadIdx.x]);
}


__global__ void saturation(int* histogram, const int binsNumber, const int max_value) {

	int indx = blockDim.x * blockIdx.x + threadIdx.x;

	if( indx < binsNumber )
		if(histogram[indx] > max_value)
			histogram[indx] = max_value;
}


long CPU_histogram(const float* data, const long size, const int binsNumber, std::string name) {

	std::vector<int> histogram;
	histogram.resize(binsNumber);

	auto start = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < size; i++)
		histogram[static_cast<int>(data[i])]++;
	auto stop = std::chrono::high_resolution_clock::now();

	std::ofstream save;
	save.open("CPU_histogram"+name+".txt");

	for(int i = 0; i < binsNumber; i++)
		save << i << "\t" << histogram[i] << std::endl;

	save.close();

	return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
}

//////////////////////////////////////////////////////////////////////////////////////////////


int main() {

	std::random_device rd;
	std::mt19937 generator{ rd() };
	std::uniform_real_distribution<float> uniform_distribution{ 0.0, 2047.0 };

	float* data;
	const long size = 10e5;
	const int binsNumber = 2048;
	cudaMallocManaged(&data, size * sizeof(float));
	int ID;
	int SM;
	cudaGetDevice(&ID);
	cudaDeviceGetAttribute(&SM, cudaDevAttrMultiProcessorCount, ID);

	for(int i = 0; i < size; i++)
		data[i] = uniform_distribution(generator);

	auto CPU_time = CPU_histogram(data, size, binsNumber, "uniform");

	cudaMemPrefetchAsync(data, size * sizeof(float), ID);

	int* device_histogram;
	cudaMallocManaged(&device_histogram, binsNumber * sizeof(int));
	
	for(int i = 0; i < binsNumber; i++)
		device_histogram[i] = 0;	

	cudaMemPrefetchAsync(device_histogram, binsNumber * sizeof(int), ID);

	auto start = std::chrono::high_resolution_clock::now();
	main_histogram<<<SM * 32, 256>>>(data, size, device_histogram, binsNumber);
	cudaDeviceSynchronize();
	auto stop = std::chrono::high_resolution_clock::now();

	std::ofstream save;
	save.open("GPU_histogram_uniform_.txt");

	for(int i = 0; i < binsNumber; i++)
		save << i << "\t" << device_histogram[i] << std::endl;

	save.close();
	save.open("time_uniform.txt");
	save << CPU_time << std::endl << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
	save.close();

	return 0;
}
