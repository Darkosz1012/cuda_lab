/*
 ============================================================================
 Name        : reduction.cu
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
#include <cuda.h>
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


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)



__global__ void addNaive(float* data, int N){
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x+1;
	if (idx < N)
		atomicAdd(data, data[idx]);
}

__global__ void addReduction(float* data, int N){
	__shared__  int partialSum[2* 256];

	unsigned int t = threadIdx.x;
	unsigned int start = 2*blockIdx.x*blockDim.x;
	partialSum[t] = start+t<N?data[start+t] : 0;
	partialSum[blockDim.x+t] = start + blockDim.x + t < N ? data[start + blockDim.x + t] : 0;

	for (unsigned int stride = 1; stride <=blockDim.x;stride *=2){
		__syncthreads();
		if(t%stride==0){
			atomicAdd(&partialSum[2*t],partialSum[2*t+stride]);
		}
	}
	data[blockIdx.x]=partialSum[0];
}

double setUpReduction(int N,const int SM, const int ID){
	float* data;
	CUDA_CHECK_RETURN(cudaMallocManaged(&data, N * sizeof(float)));
	for(int i = 0; i < N; i++)
		data[i] = 1;

	CUDA_CHECK_RETURN(cudaMemPrefetchAsync(data, N * sizeof(float), ID));
	
	auto start = std::chrono::high_resolution_clock::now();
	addReduction<<<N/256+1, 256>>>(data, N);
	CUDA_CHECK_RETURN(cudaPeekAtLastError() );
	cudaDeviceSynchronize();
	int result = 0;
	for(int i = 0; i < N/256+1;i++){
		result += data[i];
		//std::cout << data[i]<<std::endl;
	}
	auto stop = std::chrono::high_resolution_clock::now();
	//std::cout << "Reduction: "<< result<< " "<< N << std::endl;
	CUDA_CHECK_RETURN(cudaFree(data));
	return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
}

double setUpNaive(int N,const int SM, const int ID){
	float* data;
	CUDA_CHECK_RETURN(cudaMallocManaged(&data, N * sizeof(float)));
	for(int i = 0; i < N; i++)
		data[i] = 1;

	CUDA_CHECK_RETURN(cudaMemPrefetchAsync(data, N * sizeof(float), ID));

	auto start = std::chrono::high_resolution_clock::now();
	addNaive<<<N/256+1, 256>>>(data, N);
	CUDA_CHECK_RETURN(cudaPeekAtLastError() );
	cudaDeviceSynchronize();
	auto stop = std::chrono::high_resolution_clock::now();
	//std::cout << "Naive: "<< data[0]<< " "<< N << std::endl;
	CUDA_CHECK_RETURN(cudaFree(data));
	return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
}



int main(void)
{
	int ID;
	int SM;
	cudaGetDevice(&ID);
	cudaDeviceGetAttribute(&SM, cudaDevAttrMultiProcessorCount, ID);
	std::ofstream save1, save2;
	save1.open("naive.txt");
	save2.open("reduction.txt");

	for(int N = 50000; N <=50000000; N+=50000){
		int sum1=0, sum2=0;
		int iter = 20;
		for(int i = 0; i < iter;i++){
			sum1+=setUpNaive(N,SM, ID);
			sum2+=setUpReduction(N,SM, ID);
		}
		save1 << N << "\t"<< sum1/iter<<std::endl;
		save2 << N << "\t"<<  sum2/iter<<std::endl;
		std::cout << "N: "<<N << "\t"<< sum1/iter<<std::endl;
		std::cout << "R: "<<N << "\t"<<  sum2/iter<<std::endl;
	}
	save1.close();
	save2.close();
	return 0;
}

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

