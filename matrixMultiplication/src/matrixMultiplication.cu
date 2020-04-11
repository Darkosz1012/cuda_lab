#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <fstream>
#include <chrono>
#include <iostream>



__global__ void matrixMultiplication2D(const float *A, const float *B, float *C, int size) {
	
	int rowIdx = blockIdx.y * size + threadIdx.y;
	int colIdx = blockIdx.x * size + threadIdx.x;
	
	if(rowIdx < size && colIdx < size){
		
		float product = 0;

		for(int i = 0; i < size; i++){

			product += A[rowIdx * size + i] * B[i * size + colIddx];
		}
		
		C[rowIdx * size + colIdx] = product;
	}
}

__global__ void matrixMultiplication3D(const float *A, const float *B, float *C, int size) {

	int rowIdx = blockIdx.y * size + threadIdx.y;
	int colIdx = blockIdx.x * size + threadIdx.x;
	int deepIdx = blockIdx.x * size + threadIdx.x;
	if(rowIdx < size && colIdx < size && deepIdx < size){
		C[rowIdx * size + colIdx] += A[rowIdx * size + deepIdx] * B[deepIdx * size + colIddx];
	}
}

void checkMatrixMul( int * a, int * b, int * c )
{
    int val = 0;

    for( int row = 0; row < N; ++row )
        for( int col = 0; col < N; ++col )
        {
            val = 0;
            for ( int k = 0; k < N; ++k )
                val += a[row * N + k] * b[k * N + col];
            if(c[row * N + col] != val)
                std::cout<<"Error: Result"<<std::endl;
        }
}

inline cudaError_t checkCUDA(cudaError_t result){

	if(result != cudaSuccess){
	
	fprintf(stderr, "CUDA Runtime error: %s\n", cudaGetErrorString(result));
	assert(result == cudaSuccess);
	}	
	
	return result;
}

void allocWithCPU(float* h_A, float* h_B, float* h_result, float* d_A, float* d_B, float* d_result,size_t size,int numberOfElemets){
    //classic mallocs
    h_A = static_cast<float*>(malloc(size));
    h_B = static_cast<float*>(malloc(size));
    h_result = static_cast<float*>(malloc(size));

    for(int j = 0; j < numberOfElements; j++){
        h_A[j] = static_cast<float>(rand())/RAND_MAX;
        h_B[j] = static_cast<float>(rand())/RAND_MAX;
        h_result[j]=0;
    }

    checkCUDA(cudaMalloc((void**)&d_A, size));
    checkCUDA(cudaMalloc((void**)&d_B, size));
    checkCUDA(cudaMalloc((void**)&d_result, size));

    checkCUDA(cudaMemcpy(d_A, h_A, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(d_B, h_B, cudaMemcpyHostToDevice));
}
void allocWithGPU(float* A, float* B, float* result,size_t size,int numberOfElemets){
    checkCUDA(cudaMallocManaged(&A, size));
    checkCUDA(cudaMallocManaged(&B, size));
    checkCUDA(cudaMallocManaged(&result, size));

    for(int j = 0; j < numberOfElements; j++){
        A[j] = static_cast<float>(rand())/RAND_MAX;
        B[j] = static_cast<float>(rand())/RAND_MAX;
        result[j] = 0;
    }

}
int main() {
	
	float* h_A, h_B, h_result;
	float* d_A, d_B, d_result;

	float* A, B, result;

	int numberOfElementsInDim = 10;
	int numberOfElemets = numberOElementsInDim*NumberOfElementsInDim;
	size_t size = numberOfElements * sizeof(float);

	std::ofstream save;

	std::chrono::high_resolution_clock start;
	std::chrono::high_resolution_clock stop;
	std::chronoduration<double> elapsed_time;

    int jump = 100;
    int numberOfResult = 100;

    save.open("classic_2D.txt");
	for(int i = 0; i < numberOfResult; i++){
        double average = 0;
		for(int k = 0; k < 10; k++){
			start = std::chrono::high_resolution_clock::now();

			allocWithCPU(h_A, h_B, h_result, d_A, d_B, d_result, size, numberOfElements);

            dim3 threads_per_block (16, 16, 1); // A 16 x 16 block threads
            dim3 number_of_blocks ((numberOfElementsInDim / threads_per_block.x) + 1, (numberOfElementsInDim / threads_per_block.y) + 1, 1);
			matrixMultiplication2D<<<number_of_blocks, threads_per_block>>>(d_A, d_B, d_result, numberOfElementsInDim);

            cudaDeviceSynchronize();

			stop = std::chrono::high_resolution_clock::now();
			elapsed_time = stop - start;
            average = (average*k+elapsed_time)/k+1;

            checkCUDA(cudaMemcpy(d_result, h_result, cudaMemcpyDeviceToHost));
            
            checkMatrixMul(h_A,h_B,h_result);

			checkCUDA(cudaFree(d_A));
			checkCUDA(cudaFree(d_B));
			checkCUDA(cudaFree(d_result));
			free(h_A);
			free(h_B);
			free(h_result);

		}
        save << numberOfElements <<"\t"<< numberOfElemetnsInDim <<"\t" << average << std::endl;
		
        
		numberOfElemetnsInDim += jump;
		numberOfElements = numberOfElemetnsInDim * numberOfElemetnsInDim;
		size = numberOfElemetns * sizeof(float);
	}
    save.close();

    save.open("managed_2D.txt");
	for(int i = 0; i < numberOfResult; i++){
        double average = 0;
		for(int k = 0; k < 10; k++){
			start = std::chrono::high_resolution_clock::now();

			allocWithGPU(A, B, result, size, numberOfElements);

            dim3 threads_per_block (16, 16, 1); // A 16 x 16 block threads
            dim3 number_of_blocks ((numberOfElementsInDim / threads_per_block.x) + 1, (numberOfElementsInDim / threads_per_block.y) + 1, 1);
			matrixMultiplication2D<<<number_of_blocks, threads_per_block>>>(A, B, result, numberOfElementsInDim);

            cudaDeviceSynchronize();

			stop = std::chrono::high_resolution_clock::now();
			elapsed_time = stop - start;
            average = (average*k+elapsed_time)/k+1;
            
            checkMatrixMul(A,B,result);

            checkCUDA(cudaFree(A));
            checkCUDA(cudaFree(B));
            checkCUDA(cudaFree(result));

		}
        save << numberOfElements <<"\t"<< numberOfElemetnsInDim <<"\t" << average << std::endl;
		
        
		numberOfElemetnsInDim += jump;
		numberOfElements = numberOfElemetnsInDim * numberOfElemetnsInDim;
		size = numberOfElemetns * sizeof(float);
	}
    save.close();

   

	return 0;
}

