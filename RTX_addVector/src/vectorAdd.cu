/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
		int numElements) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {
		C[i] = A[i] + B[i];
	}
float timedifference_msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}

/**
 * Host main routine
 */
int main(void) {
	// Error code to check return values for CUDA call
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = 10;

	clock_t start_t, end_t, total_t;
	struct timeval t0;
   struct timeval t1;
   float elapsed;

	//We are using this loop to check how long vectors we can process
	for (int out = 0; out < 50; out++) {
		char bigEnough[64] = "";
		sprintf(bigEnough, "%d", numElements);
		//strcat(bigEnough, numElements);
		strcat(bigEnough, ".txt");
		FILE* execution_time;
		execution_time = fopen(bigEnough, "w");

		size_t size = numElements * sizeof(float);
		printf("[Vector addition of %d elements]\n", numElements);
		fprintf(execution_time, "Time for add vectors of %d elements:\n",numElements);
		// Allocate the host operating vectors
		float *h_A = (float *) malloc(size);
		float *h_B = (float *) malloc(size);
		float *h_C = (float *) malloc(size);

		// Verify that allocations succeeded
		if (h_A == NULL || h_B == NULL || h_C == NULL) {
			fprintf(stderr, "Failed to allocate host vectors!\n");
			exit(EXIT_FAILURE);
		}

		// Initialize the host input vectors
		for (int i = 0; i < numElements; ++i) {
			h_A[i] = rand() / (float) RAND_MAX;
			h_B[i] = rand() / (float) RAND_MAX;
		}
		for (int in = 0; in < 10; in++) {
			gettimeofday(&t0, 0);
			// Allocate the device input vectors A and B
			float *d_A = NULL;
			err = cudaMalloc((void **) &d_A, size);

			if (err != cudaSuccess) {
				fprintf(stderr,
						"Failed to allocate device vector A (error code %s)!\n",
						cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}

			float *d_B = NULL;
			err = cudaMalloc((void **) &d_B, size);

			if (err != cudaSuccess) {
				fprintf(stderr,
						"Failed to allocate device vector B (error code %s)!\n",
						cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}

			// Allocate the device output vector C
			float *d_C = NULL;
			err = cudaMalloc((void **) &d_C, size);

			if (err != cudaSuccess) {
				fprintf(stderr,
						"Failed to allocate device vector C (error code %s)!\n",
						cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}

			// Copy the host input vectors A and B in host memory to the device input vectors in
			// device memory
			printf("Copy input data from the host memory to the CUDA device\n");
			err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

			if (err != cudaSuccess) {
				fprintf(stderr,
						"Failed to copy vector A from host to device (error code %s)!\n",
						cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}

			err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

			if (err != cudaSuccess) {
				fprintf(stderr,
						"Failed to copy vector B from host to device (error code %s)!\n",
						cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}



			// Launch the Vector Add CUDA Kernel
			int threads = 1024;
			int threadsPerBlock = 256;
			int blocksPerGrid = numElements/threadsPerBlock+threadsPerBlock;
			printf("CUDA kernel launch with %d blocks of %d threads\n",
					blocksPerGrid, threadsPerBlock);

//			cudaEvent_t start, stop;
//			cudaEventCreate(&start);
//			cudaEventCreate(&stop);
//
//			cudaEventRecord(start);
			vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,
					numElements);
//			cudaEventRecord(stop);

//			gettimeofday(&t0, 0);
//			vectorAddCpu(h_A, h_B, h_C, numElements);
//			gettimeofday(&t1, 0);

			err = cudaGetLastError();

			if (err != cudaSuccess) {
				fprintf(stderr,
						"Failed to launch vectorAdd kernel (error code %s)!\n",
						cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}

//			float milliseconds;
//			cudaEventSynchronize(stop);
//			cudaEventElapsedTime(&milliseconds, start, stop);





			// Copy the device result vector in device memory to the host result vector
			// in host memory.
			printf("Copy output data from the CUDA device to the host memory\n");
			err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

			if (err != cudaSuccess) {
				fprintf(stderr,
						"Failed to copy vector C from device to host (error code %s)!\n",
						cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
			gettimeofday(&t1, 0);
			elapsed = timedifference_msec(t0, t1);
			fprintf(execution_time, "%f\n",elapsed);

			// Verify that the result vector is correct
			for (int i = 0; i < numElements; ++i) {
				if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
					fprintf(stderr, "Result verification failed at element %d!\n",
							i);
					exit(EXIT_FAILURE);
				}
			}

			printf("Test PASSED\n");

			// Free device global memory
			err = cudaFree(d_A);

			if (err != cudaSuccess) {
				fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
						cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}

			err = cudaFree(d_B);

			if (err != cudaSuccess) {
				fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
						cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}

			err = cudaFree(d_C);

			if (err != cudaSuccess) {
				fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
						cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
		}

		// Free host memory
		free(h_A);
		free(h_B);
		free(h_C);

		fprintf(execution_time, "\n\n\n");

		numElements = numElements * 10;
		fclose(execution_time);
	}

	printf("Done\n");
	return 0;
}
