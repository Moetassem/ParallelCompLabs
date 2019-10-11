#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "gputimer.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;
using namespace lodepng;

// The maximum number of threads per block is set by the GPU. It was determined via trial and error
constexpr auto MAX_NUMBER_THREADS = 1024;

cudaError_t imageRectificationWithCuda(int numOfThreads, char* inputImageName, char* outputImageName);

cudaError_t imagePoolingWithCuda(int numOfThreads, char* inputImageName, char* outputImageName);

/*
*
* imgRectificationKernel takes an input array and sets any RGBA values less than 127 to 127, otherwise keeps them as is.
* It uses the size of the input array and the numberOfThreads requested to figure out the index used to access the array (k)
*
*/
__global__ void imgRectificationKernel(unsigned char* inMatrix, int size, int numOfThreads)
{
	for (int i = 0; i < size / numOfThreads; i++) {
		int	k = (numOfThreads * i + threadIdx.x) + (blockIdx.x * numOfThreads);
		if (inMatrix[k] < 127) {
			inMatrix[k] = 127;
		}
		else {
			inMatrix[k] = inMatrix[k];
		}
	}
}

/*
*
* pixelSplitIntoQuarters takes an RGBA array splits it into seperate arrays, 
* in order to make it easier to run the pooling algorithm. This process happens in parrallel
*
*/
__global__ void pixelsSplitIntoQuarters(unsigned char* rgbaArray, unsigned char* rArray, unsigned char* gArray, unsigned char* bArray, unsigned char* aArray,
										int sizeofQuarterPixels, int numOfThreads)
{
	for (int i = 0; i < (sizeofQuarterPixels) / numOfThreads; i++) {
		int j = (threadIdx.x + numOfThreads * i) + (blockIdx.x * numOfThreads);
		int k = j * 4;

		rArray[j] = rgbaArray[k];
		gArray[j] = rgbaArray[k + 1];
		bArray[j] = rgbaArray[k + 2];
		aArray[j] = rgbaArray[k + 3];
	}	
}

/*
*
* arrayMaxPerQuarterPixelKernel takes a single RGBA channel array and runs the pooling algorithm for 2x2 squares - in parallel
*
*/
__global__ void arrayMaxPerQuarterPixelKernel(unsigned char* inArray, unsigned char* outArray, int sizeofQuarterPixels, int numOfThreads, int width)
{
	for (int i = 0; i < ((sizeofQuarterPixels / 4) / numOfThreads); i++) {
		int j = (threadIdx.x + numOfThreads * i) + (blockIdx.x * numOfThreads);
		int k = 2 * j + width * (j / (width / 2));

		if (inArray[k] > inArray[k + 1]) {
			outArray[j] = inArray[k];
		}
		else {
			outArray[j] = inArray[k + 1];
		}
		if (inArray[k + width] > outArray[j]) {
			outArray[j] = inArray[k + width];
		}
		if (inArray[k + width + 1] > outArray[j]) {
			outArray[j] = inArray[k + width + 1];
		}
	}
}

/*
*
* pixelsMerge takes the seprated R G B A arrays and copies them back into a single RGBA array so the image can be outputted
*
*/
__global__ void pixelsMerge(unsigned char* outrArray, unsigned char* outgArray, unsigned char* outbArray, unsigned char* outaArray, unsigned char* outfinalArray,
							int sizeofQuarterPixels, int numOfThreads) {
	for (int i = 0; i < ((sizeofQuarterPixels/4) / numOfThreads); i++) {
		int j = (threadIdx.x + numOfThreads * i) + (blockIdx.x * numOfThreads);
		int k = 4 * j;

		outfinalArray[k] = outrArray[j];
		outfinalArray[k + 1] = outgArray[j];
		outfinalArray[k + 2] = outbArray[j];
		outfinalArray[k + 3] = outaArray[j];
	}
}

/*
* The main function is run via the commandline 
*/
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

	if (argv[1] != NULL && !strcmp(argv[1],"rectify")) {
		cout << "Rectifing" << endl;
		cudaError_t status = imageRectificationWithCuda(numOfThreads, inputImgName, outImgName);
	}

	if (argv[1] != NULL && !strcmp(argv[1], "pool")) {
		cout << "Pooling" << endl;
		cudaError_t status = imagePoolingWithCuda(numOfThreads, inputImgName, outImgName);
	}
		
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
	GpuTimer gpuTimer; // Struct for timing the GPU
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

	// Allocated memory on the GPU for the input image
	cudaStatus = cudaMallocManaged((void**)&dev_inMat, sizeOfMat * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy the input image into the GPU
	for (int i = 0; i < sizeOfMat; i++) {
		dev_inMat[i] = inputImage[i];
	}

	// Launch kernel on the GPU with one thread for each element.
	int numBlocks = ((numOfThreads + (MAX_NUMBER_THREADS - 1)) / MAX_NUMBER_THREADS);
	int threadsPerBlock = ((numOfThreads + (numBlocks - 1)) / numBlocks);

	/************ Parrallel Part of Execution ************/
	gpuTimer.Start();
	imgRectificationKernel <<<numBlocks, threadsPerBlock>> > (dev_inMat, sizeOfMat, threadsPerBlock);
	gpuTimer.Stop();
	/****************************************************/
	printf("-- Number of Threads: %d -- Execution Time (ms): %g \n", numOfThreads, gpuTimer.Elapsed());

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
	GpuTimer gpuTimer; // Struct for timing the GPU
	unsigned char* inputImage = nullptr;
	unsigned width, height = 0;

	int error = lodepng_decode32_file(&inputImage, &width, &height, inputImageName);
	if (error != 0) {
		cout << "You f**ed up decoding the image" << endl;
		cudaStatus = cudaError_t::cudaErrorAssert;
		goto Error;
	}

	int sizeOfArray = width * height * 4;

	unsigned char *dev_RGBAArray, *dev_RArray, *dev_GArray, *dev_BArray, *dev_AArray, *dev_outRArray, *dev_outGArray, *dev_outBArray, *dev_outAArray, *dev_outArray;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Declare memory for the input array on the GPU
	cudaStatus = cudaMallocManaged((void**)& dev_RGBAArray, sizeOfArray * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	for (int i = 0; i < sizeOfArray; i++) {
		dev_RGBAArray[i] = inputImage[i];
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

	int numBlocks = ((numOfThreads + (MAX_NUMBER_THREADS - 1)) / MAX_NUMBER_THREADS);
	int threadsPerBlock = ((numOfThreads + (numBlocks - 1)) / numBlocks);

	/************ Parrallel Part of Execution ************/
	gpuTimer.Start();
	pixelsSplitIntoQuarters << <numBlocks, threadsPerBlock >> > (dev_RGBAArray, dev_RArray, dev_GArray, dev_BArray, dev_AArray, sizeOfArray/4, threadsPerBlock);

	//int numOfThreadsPerBlock = 1024;
	// Launch kernel on the GPU with one thread for each element.
	arrayMaxPerQuarterPixelKernel <<<numBlocks, threadsPerBlock >> > (dev_RArray, dev_outRArray, sizeOfArray/4, threadsPerBlock, width);

	arrayMaxPerQuarterPixelKernel <<<numBlocks, threadsPerBlock >> > (dev_GArray, dev_outGArray, sizeOfArray/4, threadsPerBlock, width);

	arrayMaxPerQuarterPixelKernel <<<numBlocks, threadsPerBlock >> > (dev_BArray, dev_outBArray, sizeOfArray/4, threadsPerBlock, width);

	arrayMaxPerQuarterPixelKernel <<<numBlocks, threadsPerBlock >> > (dev_AArray, dev_outAArray, sizeOfArray/4, threadsPerBlock, width);

	pixelsMerge <<<numBlocks, threadsPerBlock >> > (dev_outRArray, dev_outGArray, dev_outBArray, dev_outAArray, dev_outArray, sizeOfArray/4, threadsPerBlock);
	gpuTimer.Stop();
	/****************************************************/
	printf("-- Number of Threads: %d -- Execution Time (ms): %g \n", numOfThreads, gpuTimer.Elapsed());

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
	// BE FREE MY LOVLIES
	cudaFree(dev_RGBAArray);
	cudaFree(dev_RArray);
	cudaFree(dev_GArray);
	cudaFree(dev_BArray);
	cudaFree(dev_AArray);
	cudaFree(dev_outRArray);
	cudaFree(dev_outGArray);
	cudaFree(dev_outBArray);
	cudaFree(dev_outAArray);
	cudaFree(dev_outArray);

	return cudaStatus;
}
