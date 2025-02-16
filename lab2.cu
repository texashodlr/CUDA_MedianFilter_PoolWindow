/**
 * @file lab2.cpp
 * @author Abhishek Bhaumick (abhaumic@purdue.edu)
 * @brief 
 * @version 0.1
 * @date 2021-01-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include "lab2.cuh"
#include "cpuLib.h"
#include "cudaLib.cuh"

int main(int argc, char** argv) {
	std::string str;
	int choice;

	std::cout << "ECE 695 - Lab 2 \n";
	std::cout << "Select application: \n";
	std::cout << "  1 - CPU SAXPY \n";
	std::cout << "  2 - GPU SAXPY \n";
	std::cout << "  3 - CPU Monte-Carlo Pi \n";
	std::cout << "  4 - GPU Monte-Carlo Pi \n";
	std::cout << "  5 - Bytes-Image File Test \n";
	std::cout << "  6 - Median Filter CPU \n";
	std::cout << "  7 - Median Filter GPU \n";
	std::cout << "  8 - Pool CPU \n";
	std::cout << "  9 - Pool GPU \n";

	std::cin >> choice;

	std::cout << "\n";
	std::cout << "Choice selected - " << choice << "\n\n";

	PoolLayerArgs poolArgs;
	MedianFilterArgs filArgs;
	TensorShape inShape;

	switch (choice) {
		//  CPU only SAXPY
		case 1:
			std::cout << "Running SAXPY CPU! \n\n";
			runCpuSaxpy(VECTOR_SIZE);
			std::cout << "\n\n ... Done!\n";
			break;

		//  CUDA + GPU SAXPY
		case 2:
			std::cout << "Running SAXPY GPU! \n\n";
			runGpuSaxpy(VECTOR_SIZE);
			std::cout << "\n\n ... Done!\n";
			break;

		case 3:
			std::cout << "Running Monte-Carlo Pi Estimation on CPU! \n\n";
			runCpuMCPi(MC_ITER_COUNT, MC_SAMPLE_SIZE);
			std::cout << "\n\n ... Done!\n";
			break;

		case 4:
			std::cout << "Running Monte-Carlo Pi Estimation on GPU! \n\n";
			runGpuMCPi(GENERATE_BLOCKS, SAMPLE_SIZE, REDUCE_BLOCKS, REDUCE_SIZE);
			std::cout << "\n\n ... Done!\n";
			break;

		case 5:
			std::cout << "Running BytesImage File Test on CPU! \n\n";
			testLoadBytesImage("./resources/lena512color.tiff.bytes");
			std::cout << "\n\n ... Done!\n";
			break;

		case 6:
			std::cout << "Running Median Filter on CPU! \n\n";
			filArgs = { 4, 4 };
			runCpuMedianFilter("./resources/lena512color.tiff.bytes",
				"./resources/lena512color_fil.bytes", filArgs);
			std::cout << "\n\n ... Done!\n";

		case 7:
			std::cout << "Running Median Filter on GPU! \n\n";
			filArgs = { 4, 4 };
			runGpuMedianFilter("./resources/lena512color.tiff.bytes",
				"./resources/lena512color_fil.bytes", filArgs);
			std::cout << "\n\n ... Done!\n";
			break;
			
		case 8:
			std::cout << "Running Pool CPU! \n\n";
			inShape = { 32, 32 };
			poolArgs = { PoolOp::MaxPool, 4, 4, 1, 1 };
			runCpuPool(inShape, poolArgs);
			std::cout << "\n\n ... Done!\n";
			break;
			
		case 9:
			std::cout << "Running Pool GPU! \n\n";
			//	STUDENT: Call runGpuPool here
			inShape = { 32, 32 };
			poolArgs = { PoolOp::MaxPool, 4, 4, 1, 1 };
			runGpuPool(inShape, poolArgs);
			std::cout << "\n\n ... Done!\n";
			break;

		default:
			std::cout << "Hmm ... Devious, you are!\n";
			std::cout << " Choose correctly, you must.\n";
			break;
	}

	return 0;
}

int testLoadBytesImage(std::string filePath) {
	ImageDim imgDim;
	uint8_t * imgData;
	int bytesRead = loadBytesImage(filePath, imgDim, &imgData);
	int bytesExpected = imgDim.height * imgDim.width * imgDim.channels * imgDim.pixelSize;
	if (bytesRead != bytesExpected) {
		std::cout << "Read Failed - Insufficient Bytes - " << bytesRead 
			<< " / "  << bytesExpected << " \n";
		return -1;
	}
	std::cout << "Read Success - " << bytesRead << " Bytes \n"; 
	return 0;
}


