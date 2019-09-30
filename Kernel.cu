#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;

constexpr auto matSize = 16;

cudaError_t imageRectificationWithCuda(string& inputImg, string& outImg, int& numOfThreads);

__global__ void imgRectificationKernel(int* matrix)
{
	int i = blockIdx.x * 4 + threadIdx.x;
	if (matrix[i] < 127) {
		matrix[i] = 127;
	}
}

int main(int argc, char* argv[])
{
	string inputImgName = "";
	string outImgName = "";
	int numOfThreads = 0;

	if (argc != 4 || argv[1] == NULL || argv[2] == NULL || argv[3] == NULL ||
		argv[1] == "-h" || argv[1] == "--help" || argv[1] == "--h") {
		cout << "Assignment1.exe <name of input png> <name of output png> < # threads>" << endl;
	}
	else {
		if (argv[1] != NULL) {
			inputImgName = argv[1];
		}
		if (argv[2] != NULL) {
			outImgName = argv[2];
		}
		if (argv[3] != NULL) {
			numOfThreads = stoi(argv[3]);
		}
	}

	std::cout << "Name of Input Image File: " << inputImgName << std::endl;
	std::cout << "Name of Output Image File: " << outImgName << std::endl;
	std::cout << "Name of Output Image File: " << numOfThreads << std::endl;

	cudaError_t status = imageRectificationWithCuda(inputImgName, outImgName, numOfThreads);
	return 0;
}

cudaError_t imageRectificationWithCuda(string &inputImg, string &outImg, int &threads)
{
	int sizeOfMat = 16;
	int matrix[4][4] = { {126, 128, 122, 100}, {129, 120, 350, 300}, {122, 135, 127, 129}, {140, 145, 190, 195} };

	int* dev_Mat = nullptr;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)& dev_Mat, 16 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_Mat, &matrix, 16 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host to dev failed!");
		goto Error;
	}

	int numOfThreadsPerBlock = 4;
	// Launch kernel on the GPU with one thread for each element.
	imgRectificationKernel <<<(sizeOfMat + (numOfThreadsPerBlock - 1)) / numOfThreadsPerBlock, numOfThreadsPerBlock >> > (dev_Mat);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "imgRectificationKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching imgRectificationKernel!\n", cudaStatus);
		goto Error;
	}

	int* output[4][4] = { 0 };
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(output, dev_Mat, sizeOfMat * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	for (int i = 0; i < 16; i++) {
		printf("%i", output[i / 4][i % 4]);
	}

Error:
	cudaFree(dev_Mat);

	return cudaStatus;
}
