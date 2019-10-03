#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>

#define LODEPNG_COMPILE_DECODER

using namespace std;
using namespace lodepng;

//constexpr auto matSize = 16;

cudaError_t imageRectificationWithCuda(std::vector<unsigned char>& inputImage, int& numOfThreads, unsigned char* outputImgPtr);

__global__ void imgRectificationKernel(unsigned char* matrix)
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

	if (argc != 5 || argv[1] == NULL || argv[2] == NULL || argv[3] == NULL || argv[4] == NULL ||
		argv[1] == "-h" || argv[1] == "--help" || argv[1] == "--h") {
		cout << "Assignment1.exe <Command> <name of input png> <name of output png> < # threads>" << endl;
	}
	else {
		if (argv[2] != NULL) {
			inputImgName = argv[2];
		}
		if (argv[3] != NULL) {
			outImgName = argv[3];
		}
		if (argv[4] != NULL) {
			numOfThreads = stoi(argv[4]);
		}
	}

	std::vector<unsigned char> inputImage;
	unsigned width = 0;
	unsigned height = 0;

	if (decode(inputImage, width, height, inputImgName) != 0) {
		cout << "You F**ed up decoding the image" << endl;
		return 0;
	}

	unsigned char* outputImgPtr = nullptr;

	if (argv[1] != NULL && argv[1] == "rectify") {
		cudaError_t status = imageRectificationWithCuda(inputImage, numOfThreads, outputImgPtr);
		cout << status << endl;
	}

	//else if (argv[1] != NULL && argv[1] == "pool") {
		//TODO::pool function
	//}
	//scout << *outputImgPtr << endl;

	/*if (encode(outImgName, outputImgPtr, width, height, LodePNGColorType::LCT_RGBA, 8) != 0) {
		cout << "You f**ed up encoding the image" << endl;
		return 0;
	}*/
		
	std::cout << "Name of Input Image File: " << inputImgName << std::endl;
	std::cout << "Name of Output Image File: " << outImgName << std::endl;
	std::cout << "Name of Output Image File: " << numOfThreads << std::endl;

	//cudaError_t status = imageRectificationWithCuda(inputImgName, outImgName, numOfThreads);
	return 0;
}

cudaError_t imageRectificationWithCuda(vector<unsigned char> &inputImage, int& numOfThreads, unsigned char* outputImgPtr)
{
	int sizeOfMat = inputImage.size();

	unsigned char* dev_Mat = nullptr;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)& dev_Mat, sizeOfMat * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_Mat, &inputImage, sizeOfMat * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host to dev failed!");
		goto Error;
	}

	int numOfThreadsPerBlock = 1024;
	// Launch kernel on the GPU with one thread for each element.
	imgRectificationKernel <<<(numOfThreads + (numOfThreadsPerBlock - 1)) / numOfThreadsPerBlock, (numOfThreads % numOfThreadsPerBlock) >> > (dev_Mat);

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

	unsigned char* output = nullptr;
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(output, dev_Mat, sizeOfMat * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	outputImgPtr = output;
	cout << output << endl;

Error:
	cudaFree(dev_Mat);

	return cudaStatus;
}
