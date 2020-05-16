/*
 ============================================================================
 Name        : histogram_nsight.cu
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include <helper_cuda.h>
//#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <fstream>
#include <array>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <functional>
#include <assert.h>


#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
//#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
//    printf("Error at %s:%d\n",__FILE__,__LINE__);\
//    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void main_histogram(const float* input, const long size,int* histogram, const int binsNumber) {

        int indx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        //extern __shared__  int shared_bins[];
        __shared__  int shared_bins[2048];
        for(int i = threadIdx.x; i < binsNumber; i += blockDim.x)
                shared_bins[i] = 0;

        __syncthreads();

        for(int i = indx; i < size; i += stride)
                atomicAdd(&shared_bins[ static_cast<int>(input[i]) ], 1);
        __syncthreads();

        for(int i = threadIdx.x; i < binsNumber; i += blockDim.x)
                atomicAdd(&histogram[i], shared_bins[i]);
}


__global__ void saturation(int* histogram, const int binsNumber, const int max_value) {

        int indx = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for(int i = indx; i < binsNumber; i += stride)
                if(histogram[i] > max_value)
                        histogram[i] = max_value;
}
void saturationCPU(int* histogram, const int binsNumber, const int max_value) {

        for(int i = 0; i < binsNumber; i++)
                if(histogram[i] > max_value)
                        histogram[i] = max_value;
}

__global__ void band(float* data, long N, float low, float high){
	int indx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for(int i = indx; i < N; i += stride){
		data[i]*=high-low;
		data[i]+=low;
	}
}
__global__ void band2(float* data, long N, float low, float high){
	int indx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for(int i = indx; i < N; i += stride){
		if(data[i]<low){
			data[i]=low;
		}
		if(data[i]>=high){
			data[i]=high-1;
		}
	}
}
long CPU_histogram(int * histogram, const float* data, const long size, const int binsNumber) {

        auto start = std::chrono::high_resolution_clock::now();
        for(long i = 0; i < size; i++)
                histogram[static_cast<int>(data[i])]++;
        auto stop = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
}


long GPU_histogram(int* device_histogram, const int SM, const int ID, const float* data, const long size, const int binsNumber) {

        cudaMemPrefetchAsync(data, size * sizeof(float), ID);

        auto start = std::chrono::high_resolution_clock::now();
        main_histogram<<<SM * 32, 256/*,binsNumber* sizeof(int)*/>>>(data, size, device_histogram, binsNumber);
        cudaCheck(cudaPeekAtLastError() );
        cudaDeviceSynchronize();
        auto stop = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
}

//////////////////////////////////////////////////////////////////////////////////////////////

void saveResult(int* device_histogram,const int binsNumber, std::string name){
	std::ofstream save;
	save.open(name);

	for(int i = 0; i < binsNumber; i++)
			save << i << "\t" << device_histogram[i] << std::endl;

	save.close();
}

int main() {

		std::cout << "start"<<std::endl;
		curandGenerator_t cugen;
		CURAND_CALL(curandCreateGenerator ( &cugen, CURAND_RNG_PSEUDO_MT19937));

        float* data;
        const long size = 10e7;
        const int binsNumber = 2048;
        cudaMallocManaged(&data, size * sizeof(float));
        int ID;
        int SM;
        cudaGetDevice(&ID);
        cudaDeviceGetAttribute(&SM, cudaDevAttrMultiProcessorCount, ID);

        std::vector<int> histogram(binsNumber,0);

        int* device_histogram;
		cudaMallocManaged(&device_histogram, binsNumber * sizeof(int));

		for(int i = 0; i < binsNumber; i++)
				device_histogram[i] = 0;

		cudaMemPrefetchAsync(device_histogram, binsNumber * sizeof(int), ID);

		long CPU_time = 0, GPU_time = 0;
		for(int k = 0; k < 200; k++){
			CURAND_CALL(curandGenerateUniform(cugen, data, size));
			band<<<SM * 32, 256>>>(data, size, 0.0, 2047.0);
			cudaCheck(cudaPeekAtLastError() );
			cudaDeviceSynchronize();
			std::cout<<k<<std::endl;
			CPU_time += CPU_histogram(histogram.data() ,data, size, binsNumber);
			GPU_time += GPU_histogram(device_histogram,SM, ID, data, size, binsNumber);
		}
		saturation<<<SM * 4, 256>>>(device_histogram, binsNumber, 65536);
		cudaCheck(cudaPeekAtLastError() );
		cudaDeviceSynchronize();
		cudaCheck(cudaPeekAtLastError() );
		saturationCPU(histogram.data(), binsNumber, 65536);
		saveResult(histogram.data(), binsNumber, "CPU_histogram_uniform.txt");
		saveResult(device_histogram, binsNumber, "GPU_histogram_uniform.txt");

        std::ofstream save;
        save.open("time.txt");
        save << CPU_time << std::endl << GPU_time;
//        std::default_random_engine generator1;
//        std::normal_distribution<float> normal_distribution{ 1024.0, 256.0 };


        for(int i = 0; i < binsNumber; i++){
        	histogram[i] = 0;
        	device_histogram[i] = 0;
        }

        CPU_time = 0;
        GPU_time = 0;
        for(int k = 0; k < 200; k++){
        	CURAND_CALL(curandGenerateNormal(cugen, data, size , 1024.0, 256.0));
//        	for(int i = 0 ; i < size; i++){
//        		if(data[i]<0 ){
//        			std::cout<<"Error: "<< data[i]<<std::endl;
//        		}
//        	}
        	band2<<<SM * 32, 256>>>(data, size, 0.0, 2047.0);
			cudaCheck(cudaPeekAtLastError() );
			cudaDeviceSynchronize();
        	//std::cout<<data[100]<< " "<<data[size/2]<< " "<<data[size-100]<<std::endl;
			//band<<<SM * 32, 256>>>(data, size, 0.0, 2047.0);
			//cudaCheck(cudaPeekAtLastError() );
			//cudaDeviceSynchronize();
			std::cout<<k<<std::endl;
			CPU_time += CPU_histogram(histogram.data() ,data, size, binsNumber);
			GPU_time += GPU_histogram(device_histogram,SM, ID, data, size, binsNumber);
		}
        save << std::endl << std::endl << "normal" << std::endl << CPU_time << std::endl << GPU_time;

        save.close();

        saturation<<<SM * 4, 256>>>(device_histogram, binsNumber, 65536);
		cudaCheck(cudaPeekAtLastError() );
		cudaDeviceSynchronize();
		cudaCheck(cudaPeekAtLastError() );
		saturationCPU(histogram.data(), binsNumber, 65536);
        saveResult(histogram.data(), binsNumber, "CPU_histogram_normal.txt");
        saveResult(device_histogram, binsNumber, "GPU_histogram_normal.txt");

        cudaFree(device_histogram);
        cudaFree(data);
        std::cout<<"end"<<std::endl;
        return 0;
}
