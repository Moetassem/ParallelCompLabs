#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;
using namespace lodepng;

//constexpr auto matSize = 16;

cudaError_t imageRectificationWithCuda(int numOfThreads, char* inputImageName, char* outputImageName);

cudaError_t imagePoolingWithCuda(int numOfThreads, char* inputImageName, char* outputImageName);

void max(unsigned char a, unsigned char b, unsigned char c, unsigned char d, unsigned char o);

__global__ void imgRectificationKernel(unsigned char* inMatrix, int size, int numOfThreads)
{
	for (int i = 0; i < size / numOfThreads; i++) {
		int	k = numOfThreads*i+threadIdx.x;
		if (inMatrix[k] < 127) {
			inMatrix[k] = 127;
		}
		else {
			inMatrix[k] = inMatrix[k];
		}
	}
}

__global__ void imgPoolingKernel(unsigned char* rArray, unsigned char* gArray, unsigned char* bArray, unsigned char* aArray, unsigned char* outRArray, unsigned char* outGArray, unsigned char* outBArray, unsigned char* outAArray, unsigned char* outArray, int size, int numOfThreads, int width)
{
	for (int i = 0; i < size/4 ; i = i+2) {
		
		
		int	k = numOfThreads * i + threadIdx.x;
		if (k != 0 && k % width == 0) {
			k = k + width;
		}

		// Pooling R channel
		if (rArray[k] > rArray[k + 1]) {
			outRArray[k/2] = rArray[k];
		}
		else {
			outRArray[i/2] = rArray[i+1];
		}
		if (rArray[i + width] > outRArray[j]) {
			outRArray[i/2] = rArray[i + width];
		}
		if (rArray[i + width + 1] > outRArray[i / 2]) {
			outRArray[i/2] = rArray[i + width + 1];
		}

		// Pooling 
		if (bArray[i] > bArray[i + 1]) {
			outBArray[j] = bArray[i];
		}
		else {
			outBArray[j] = bArray[i + 1];
		}
		if (bArray[i + width] > outBArray[j]) {
			outBArray[j] = bArray[i + width];
		}
		if (bArray[i + width + 1] > outBArray[i / 2]) {
			outBArray[j] = bArray[i + width + 1];
		}

		//max(rArray[i], rArray[i + 1], rArray[i + width], rArray[i + width + 1], outArray[i / 2]);
		//outArray[(i / 2) + 1] = max({ gArray[i], gArray[i + 1], gArray[i + width], gArray[i + width + 1] });

		// printf("%i\n", k);
	}

	//printf("%u\n", outMatrix[k]);

	//}
	//cout << inMatrix[i] << endl;
	//cout << outMatrix[i]
}

//void max(unsigned char a, unsigned char b, unsigned char c, unsigned char d, unsigned char o) {
//	if (a > b) {
//		o = a;
//	}
//	else {
//		o = b;
//	}
//	if (c > o) {
//		o = c;
//	}
//	if (d > o) {
//		o = d;
//	}
//}

int main(int argc, char* argv[])
{
	char* inputImgName = nullptr;
	char* outImgName = nullptr;
	int numOfThreads = 0;

	if (argc != 5 || argv[1] == NULL || argv[2] == NULL || argv[3] == NULL || argv[4] == NULL ||
		argv[1] == "-h" || argv[1] == "--help" || argv[1] == "--h") {
		cout << "Assignment1.exe <Command> <name of input png> <name of output png> < # threads>" << endl;
		return 0;
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

	if (argv[1] != NULL /*&& argv[1] == cmd.c_str()*/) {
		cudaError_t status = imageRectificationWithCuda(numOfThreads, inputImgName, outImgName);
	}

	//else if (argv[1] != NULL && argv[1] == "pool") {
		//TODO::pool function
	//}
	//scout << *outputImgPtr << endl;
		
	std::cout << "Name of Input Image File: " << inputImgName << std::endl;
	std::cout << "Name of Output Image File: " << outImgName << std::endl;
	std::cout << "Name of Output Image File: " << numOfThreads << std::endl;

	/*inputImageVec.clear();
	outputImageVec.clear();
	free(&inputImageVec);
	free(&outputImageVec);*/
	return 0;
}

cudaError_t imageRectificationWithCuda(int numOfThreads, char* inputImageName, char* outputImageName)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorDeviceUninitilialized;

	unsigned char * inputImage = nullptr;

	unsigned width, height = 0;

	int error = lodepng_decode32_file(&inputImage, &width, &height, inputImageName);
	if (error != 0) {
		cout << "You F**ed up decoding the image" << endl;
		cudaStatus = cudaError_t::cudaErrorAssert;
		goto Error;
	}

	int sizeOfMat = width * height * 4;

	unsigned char* dev_inMat;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)&dev_inMat, sizeOfMat * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	for (int i = 0; i < sizeOfMat; i++) {
		dev_inMat[i] = inputImage[i];
	}

	//int numOfThreadsPerBlock = 1024;
	// Launch kernel on the GPU with one thread for each element.
	imgRectificationKernel <<<1, numOfThreads>> > (dev_inMat, sizeOfMat, numOfThreads);
		
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

	error = lodepng_encode32_file(outputImageName, dev_inMat, width, height);
	if (error != 0) {
		cout << "You f**ed up encoding the image" << endl;
		cudaStatus = cudaError_t::cudaErrorAssert;
		goto Error;
	}
	
	free(inputImage);

Error:
	cudaFree(dev_inMat);

	return cudaStatus;
}

cudaError_t imagePoolingWithCuda(int numOfThreads, char* inputImageName, char* outputImageName)
{
	cudaError_t cudaStatus = cudaError_t::cudaErrorDeviceUninitilialized;

	unsigned char* inputImage = nullptr;

	unsigned width, height = 0;

	int error = lodepng_decode32_file(&inputImage, &width, &height, inputImageName);
	if (error != 0) {
		cout << "You F**ed up decoding the image" << endl;
		cudaStatus = cudaError_t::cudaErrorAssert;
		goto Error;
	}

	int sizeOfArray = width * height * 4;

	unsigned char *dev_RArray, *dev_GArray, *dev_BArray, *dev_AArray, *dev_outRArray, *dev_outGArray, *dev_outBArray, *dev_outAArray, *dev_outArray;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// To make our life easier, we're going to split the RGBA values into separate arrays - let's start by mallocing them
	cudaStatus = cudaMallocManaged((void**)& dev_RArray, (sizeOfArray /4) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_GArray, (sizeOfArray /4) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_BArray, (sizeOfArray /4) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_AArray, (sizeOfArray / 4) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outRArray, (sizeOfArray / 16) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outGArray, (sizeOfArray / 16) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outBArray, (sizeOfArray / 16) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outAArray, (sizeOfArray / 16) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_outArray, (sizeOfArray / 4) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Filter out the RBG channels into seperate arrays
	int filterIndex = 0;
	for (int i = 0; i < sizeOfArray; i++) {
		switch (i % 4)
		{
			case (0): dev_RArray[filterIndex] = inputImage[i];
			case (1): dev_GArray[filterIndex] = inputImage[i];
			case (2): dev_BArray[filterIndex] = inputImage[i];
				// We are simply going to ignore the alpha index, and increment the filter index
			case (3): {
				dev_AArray[filterIndex] = inputImage[i];
				filterIndex++;
			}
		}
	}

	//int numOfThreadsPerBlock = 1024;
	// Launch kernel on the GPU with one thread for each element.
	imgPoolingKernel << <1, numOfThreads >> > (dev_RArray, dev_GArray, dev_BArray, dev_AArray, dev_outRArray, dev_outGArray, dev_outBArray, dev_outAArray, dev_outArray, sizeOfArray, numOfThreads, width);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "imgPoolingKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching imgPoolingKernel!\n", cudaStatus);
		goto Error;
	}

	error = lodepng_encode32_file(outputImageName, dev_outArray, width/2, height/2);
	if (error != 0) {
		cout << "You f**ed up encoding the image" << endl;
		cudaStatus = cudaError_t::cudaErrorAssert;
		goto Error;
	}

	free(inputImage);

Error:
	cudaFree(dev_outArray);
	cudaFree(dev_RArray);
	cudaFree(dev_GArray);
	cudaFree(dev_BArray);

	return cudaStatus;
}
