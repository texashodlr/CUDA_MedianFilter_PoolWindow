
#include <iostream>
#include "cudaLib.cuh"
#include "cpuLib.h"
#include "cuda_runtime_api.h"
#define TILE_SIZE 16
#define POOL_SIZE 16

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	return 0;
}

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 3.14159f;

	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}

int runGpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args) {
	
	//Run CPU image import code START//
	ImageDim h_imgDim;
	uint8_t* h_imgData;

	int bytesRead = loadBytesImage(imgPath, h_imgDim, &h_imgData);
	int img_size = h_imgDim.height * h_imgDim.width * h_imgDim.channels * h_imgDim.pixelSize;

	std::cout << "Size = " << img_size << "\n";
	uint8_t* h_outData = (uint8_t*)malloc(img_size * sizeof(uint8_t));
	//Run CPU image import code END//


	//Now allocating memory for matrices on the device//
	uint8_t* d_imgData, *d_outData;

	//Safe  CUDA mem allocations for device matrices // 
	cudaError_t err;
	err = cudaMalloc((void**)&d_imgData, img_size * sizeof(uint8_t));
	if (err != cudaSuccess) {
		std::cerr << "CUDA malloc failed for d_imgData: " << cudaGetErrorString(err) << "\n";
		return -1;
	}

	err = cudaMalloc((void**)&d_outData, img_size * sizeof(uint8_t));
	if (err != cudaSuccess) {
		std::cerr << "CUDA malloc failed for d_outData: " << cudaGetErrorString(err) << "\n";
		cudaFree(d_imgData);  // Free already allocated memory
		return -1;
	}

	//Now copying the input matrices M and N from H to D (safely)//
	//cudaMemcpy(d_imgData, h_imgData, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice); 
	
	err = cudaMemcpy(d_imgData, h_imgData, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "CUDA memcpy (Host to Device) failed: " << cudaGetErrorString(err) << "\n";
		cudaFree(d_imgData);
		cudaFree(d_outData);
		return -1;
	}
	
	//TILE_SIZE=16 use this for TBs and Grid//
	dim3 blockSize(TILE_SIZE, TILE_SIZE, 1); // 4x4 TBs
	dim3 gridSize((h_imgDim.width + (TILE_SIZE-1)) / TILE_SIZE, (h_imgDim.height + (TILE_SIZE-1)) / TILE_SIZE, 1);
	
	//Setting Window Array size for the shared memory option//
	int windowSize = (TILE_SIZE + args.filterH - 1) * (TILE_SIZE + args.filterW - 1) * h_imgDim.channels;
	size_t sharedMemSize = windowSize * sizeof(uint8_t);

	//Diagnostics just to see memory consumption//
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("Before Kernel Launch: Free memory: %lu bytes | Total memory: %lu bytes\n", freeMem, totalMem);


	//Switch statement to select the various filter options//
	int choice;
	std::cout << "Which GPU Median Filter should we run?\n";
	std::cout << "  1 - Naive Global Memory GPU Median Filter\n";
	std::cout << "  2 - Insertion Sort Global Memory GPU Median Filter\n";
	std::cout << "  3 - Quick Select Global Memory GPU Median Filter\n";
	std::cout << "  4 - Insertion Sort Shared Memory GPU Median Filter\n";

	std::cin >> choice;

	std::cout << "\n";
	std::cout << "Choice selected - " << choice << "\n\n";

	switch (choice) {

		case 1:
			std::cout << "Running Naive Global Memory GPU Median Filter \n\n";
			medianFilter_gpu << <gridSize, blockSize >> > (d_imgData, d_outData,
				h_imgDim.height, h_imgDim.width, h_imgDim.channels, args.filterH, args.filterW);
			std::cout << "\n\n ... Done!\n";
			break;

		case 2:
			std::cout << "Running Insertion Sort Global Memory GPU Median Filter \n\n";
			medianFilter_gpu2 << <gridSize, blockSize >> > (d_imgData, d_outData,
				h_imgDim.height, h_imgDim.width, h_imgDim.channels, args.filterH, args.filterW);
			std::cout << "\n\n ... Done!\n";
			break;

		case 3:
			std::cout << "Running Quick Select Global Memory GPU Median Filter \n\n";
			medianFilter_gpu3 << <gridSize, blockSize >> > (d_imgData, d_outData,
				h_imgDim.height, h_imgDim.width, h_imgDim.channels, args.filterH, args.filterW);
			std::cout << "\n\n ... Done!\n";
			break;

		case 4:
			std::cout << "Running Insertion Sort Shared Memory GPU Median Filter \n\n";
			medianFilter_gpu4 << <gridSize, blockSize, sharedMemSize >> > (d_imgData, d_outData,
				h_imgDim.height, h_imgDim.width, h_imgDim.channels, args.filterH, args.filterW);
			std::cout << "\n\n ... Done!\n";
			break;
		
		default:
			std::cout << "Hmm ... Devious, you are!\n";
			std::cout << " Choose correctly, you must.\n";
			break;
	}

	cudaMemGetInfo(&freeMem, &totalMem);
	printf("After Kernel Launch: Free memory: %lu bytes | Total memory: %lu bytes\n", freeMem, totalMem);

	//Device sync safety//
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Kernel execution failed: " << cudaGetErrorString(err) << "\n";
		cudaFree(d_imgData);
		cudaFree(d_outData);
		return -1;
	}

	//Device image to host image copy//
	cudaMemcpy(h_outData, d_outData, img_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	//Modified CPU version//
	writeBytesImage(outPath, h_imgDim, h_outData);

	cudaFree(d_imgData);
	cudaFree(d_outData);

	std::cout << "Lazy, you are! ... ";
	std::cout << "Filter pixels, you must! ... ";

	return 0;
}

__global__
void medianFilter_gpu(uint8_t* inPixels, uint8_t* outPixels,
	int height, int width, int channels, int filterH, int filterW) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < width && row < height) {
		uint8_t window[81]; // For filter max-width 9,9
		for (int channel = 0; channel < channels; ++channel) {
			int pixels = 0;
			for (int medianRow = ((-filterH) / 2); medianRow <= (filterH / 2); ++medianRow) {
				for (int medianCol = ((-filterW) / 2); medianCol <= (filterW / 2); ++medianCol) {
					int curRow = row + medianRow;
					int curCol = col + medianCol;
					if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
						window[pixels++] = inPixels[(curRow * width + curCol) * channels + channel];
					}
				}
			}
			__syncthreads();
			// Slow and Steady Swap Sort (Naive)! //
			for (int i = 0; i < pixels - 1; ++i) {
				for (int j = i + 1; j < pixels; ++j) {
					if (window[i] > window[j]) {
						uint8_t tmp = window[i];
						window[i] = window[j];
						window[j] = tmp;
					}
				}
			}
			__syncthreads();
			outPixels[(row * width + col) * channels + channel] = window[(pixels) / 2];
		}
	}
}

//Faster Insertion Sort
__global__
void medianFilter_gpu2(uint8_t* inPixels, uint8_t* outPixels,
	int height, int width, int channels, int filterH, int filterW) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < width && row < height) {
		uint8_t window[81];
		for (int channel = 0; channel < channels; ++channel) {
			int pixels = 0;
			for (int medianRow = ((-filterH) / 2); medianRow <= (filterH / 2); ++medianRow) {
				for (int medianCol = ((-filterW) / 2); medianCol <= (filterW / 2); ++medianCol) {
					int curRow = row + medianRow;
					int curCol = col + medianCol;
					if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
						window[pixels++] = inPixels[(curRow * width + curCol) * channels + channel];
					}
				}
			}
			__syncthreads();
			/* Insertion Sort */
			for (int i = 1; i < pixels; i++) {
				uint8_t tmp = window[i];
				int j = i - 1;
				while (j >= 0 && window[j] > tmp) {
					window[j + 1] = window[j];
					j--;
				}
				window[j + 1] = tmp;
			}
			__syncthreads();
			outPixels[(row * width + col) * channels + channel] = window[(pixels) / 2];
		}
	}
}

__device__ void swap(uint8_t& a, uint8_t& b) {
	uint8_t temp = a;
	a = b;
	b = temp;
}


__device__ int partition(uint8_t* arr, int left, int right, int pivotIndex) {
	uint8_t pivotValue = arr[pivotIndex];
	swap(arr[pivotIndex], arr[right]); // Move pivot to end
	int storeIndex = left;

	for (int i = left; i < right; i++) {
		if (arr[i] < pivotValue) {
			swap(arr[i], arr[storeIndex]);
			storeIndex++;
		}
	}

	swap(arr[storeIndex], arr[right]); // Move pivot to its final place
	return storeIndex;
}

__device__ uint8_t quickselect(uint8_t* arr, int left, int right, int k) {
	while (left <= right) {
		int pivotIndex = left + (right - left) / 2; // Use middle element as pivot
		pivotIndex = partition(arr, left, right, pivotIndex);

		if (pivotIndex == k) {
			return arr[k]; // Found median
		}
		else if (pivotIndex < k) {
			left = pivotIndex + 1;
		}
		else {
			right = pivotIndex - 1;
		}
	}
	return arr[left]; // Edge case
}

//QuickSelect SPEED
__global__
void medianFilter_gpu3(uint8_t* inPixels, uint8_t* outPixels,
	int height, int width, int channels, int filterH, int filterW) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < width && row < height) {
		uint8_t window[81];
		for (int channel = 0; channel < channels; ++channel) {
			int pixels = 0;
			for (int medianRow = ((-filterH) / 2); medianRow <= (filterH / 2); ++medianRow) {
				for (int medianCol = ((-filterW) / 2); medianCol <= (filterW / 2); ++medianCol) {
					int curRow = row + medianRow;
					int curCol = col + medianCol;
					if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
						window[pixels++] = inPixels[(curRow * width + curCol) * channels + channel];
					}
				}
			}
			__syncthreads();
			int medianIndex = pixels / 2;
			uint8_t medianValue = quickselect(window, 0, pixels - 1, medianIndex);
			__syncthreads();
			outPixels[(row * width + col) * channels + channel] = medianValue;
		}
	}
}

//Shared Memory GPU Filter, with Insertion Sort //

__global__
void medianFilter_gpu4(uint8_t* inPixels, uint8_t* outPixels,
	int height, int width, int channels, int filterH, int filterW) {

	extern __shared__ uint8_t sharedMem[];

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int localCol = threadIdx.x;
	int localRow = threadIdx.y;

	int sharedW = blockDim.x + (2 * (filterW / 2));
	int sharedH = blockDim.y + (2 * (filterH / 2));

	int globalIndex = (row * width + col) * channels;

	uint8_t* sharedPixels = sharedMem;

	for (int channel = 0; channel < channels; ++channel) {
		int sharedIdx = (((localRow + filterH / 2) * sharedW + (localCol + filterW / 2)) * channels + channel);

		// Clamp to valid bounds
		int loadRow = min(max(row, 0), height - 1);
		int loadCol = min(max(col, 0), width - 1);
		int loadIdx = (loadRow * width + loadCol) * channels + channel;

		sharedPixels[sharedIdx] = inPixels[loadIdx];

		if (localCol < (filterW / 2) && (localRow + filterH / 2) < sharedH) {
			int leftCol = max(col - (filterW / 2), 0);
			sharedPixels[((localRow + filterH / 2) * sharedW + localCol) * channels + channel] =
				inPixels[(loadRow * width + leftCol) * channels + channel];
		}
		if (localCol >= blockDim.x - (filterW / 2) && (localRow + filterH / 2) < sharedH) {
			int rightCol = min(col + (filterW / 2), width - 1);
			sharedPixels[((localRow + filterH / 2) * sharedW + (localCol + filterW)) * channels + channel] =
				inPixels[(loadRow * width + rightCol) * channels + channel];
		}
		if (localRow < (filterH / 2) && (localCol + filterW / 2) < sharedW) {
			int topRow = max(row - (filterH / 2), 0);
			sharedPixels[(localRow * sharedW + (localCol + filterW / 2)) * channels + channel] =
				inPixels[(topRow * width + loadCol) * channels + channel];
		}
		if (localRow >= blockDim.y - (filterH / 2) && (localCol + filterW / 2) < sharedW) {
			int bottomRow = min(row + (filterH / 2), height - 1);
			sharedPixels[((localRow + filterH) * sharedW + (localCol + filterW / 2)) * channels + channel] =
				inPixels[(bottomRow * width + loadCol) * channels + channel];
		}

	}

	__syncthreads();

	if (col < width && row < height) {
		uint8_t window[81];
		int pixels = 0;

		for (int channel = 0; channel < channels; ++channel) {
			pixels = 0;
			for (int medianRow = -filterH / 2; medianRow <= filterH / 2; ++medianRow) {
				int curRow = localRow + filterH / 2 + medianRow;
				for (int medianCol = -filterW / 2; medianCol <= filterW / 2; ++medianCol) {
					int curCol = localCol + filterW / 2 + medianCol;
					if (curRow >= 0 && curRow < sharedH && curCol >= 0 && curCol < sharedW) {
						window[pixels++] = sharedPixels[((curRow * sharedW + curCol) * channels + channel)];
					}
				}
			}

			//Insertion Sort Again//
			for (int i = 1; i < pixels; i++) {
				uint8_t tmp = window[i];
				int j = i - 1;
				while (j >= 0 && window[j] > tmp) {
					window[j + 1] = window[j];
					j--;
				}
				window[j + 1] = tmp;
			}
			outPixels[globalIndex + channel] = window[(pixels) / 2];
		}
	}
}

int runGpuPool(TensorShape inShape, PoolLayerArgs poolArgs) {

		//Initial output sizing//
		uint32_t outputH = ((inShape.height - poolArgs.poolH) / (poolArgs.strideH)) + 1;
		uint32_t outputW = ((inShape.width - poolArgs.poolW) / (poolArgs.strideW)) + 1;

		TensorShape outShape = { outputH, outputW };
				
		//Host matrix mallocs
		size_t inMatrixSize = inShape.height * inShape.width * sizeof(float);
		size_t outMatrixSize = outShape.height * outShape.width * sizeof(float);

		float* h_inMatrix = (float*)malloc(inShape.height * inShape.width * sizeof(float));
		float* h_inMatrix_cpu = (float*)malloc(inShape.height * inShape.width * sizeof(float));
		float* h_outMatrix = (float*)malloc(outShape.height * outShape.width * sizeof(float));
		float* h_outMatrix_cpu = (float*)malloc(outShape.height * outShape.width * sizeof(float));

		//Now allocating memory for matrices on the device (safely)//
		float* d_inMatrix, * d_outMatrix;

		cudaError_t err;
		err = cudaMalloc((void**)&d_inMatrix, inMatrixSize);
		if (err != cudaSuccess) {
			std::cerr << "CUDA malloc failed for d_inMatrix: " << cudaGetErrorString(err) << "\n";
			return -1;
		}

		err = cudaMalloc((void**)&d_outMatrix, outMatrixSize);
		if (err != cudaSuccess) {
			std::cerr << "CUDA malloc failed for d_outMatrix: " << cudaGetErrorString(err) << "\n";
			cudaFree(d_inMatrix);  // Free already allocated memory
			return -1;
		}

		//Host Code to populate the vectors with random values
		srand(time(NULL));
		std::cout << "Matrix Begin: [ ";
		for (uint32_t r = 0; r < inShape.height; ++r) {
			for (uint32_t c = 0; c < inShape.width; ++c) {
				float randomValue = static_cast<float>(rand()) / RAND_MAX;
				h_inMatrix[r * inShape.width + c] = randomValue;
				h_inMatrix_cpu[r * inShape.width + c] = randomValue;
				std::cout << randomValue << ", ";
			}
			std::cout << "\n ";
		}
		std::cout << "]... \n";
		std::cout << "Set Tensors to stun !!\n";

		//Now copying the input matrices M and N from H to D (safely)//
		//cudaMemcpy(d_imgData, h_imgData, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice); 

		err = cudaMemcpy(d_inMatrix, h_inMatrix, inMatrixSize, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "CUDA memcpy (Host to Device) failed: " << cudaGetErrorString(err) << "\n";
			cudaFree(d_inMatrix);
			cudaFree(d_outMatrix);
			return -1;
		}

		//TILE_SIZE=16 use this for TBs and Grid//
		//Could be modified for dynamics with Wind/Pool/Mat size//
		dim3 blockSize(TILE_SIZE, TILE_SIZE, 1); // 4x4 TBs
		dim3 gridSize((outputW + (TILE_SIZE - 1)) / TILE_SIZE, ((outputH + (TILE_SIZE - 1)) / TILE_SIZE),1);

		//Setting Window Array size for the shared memory option//
		//int windowSize = (TILE_SIZE + args.filterH - 1) * (TILE_SIZE + args.filterW - 1) * h_imgDim.channels;
		//size_t sharedMemSize = windowSize * sizeof(uint8_t);

		//Diagnostics just to see memory consumption//
		size_t freeMem, totalMem;
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("Before Kernel Launch: Free memory: %lu bytes | Total memory: %lu bytes\n", freeMem, totalMem);

		poolLayer_gpu_shared << <gridSize, blockSize >> > (d_inMatrix, inShape, d_outMatrix, outShape, poolArgs);
		poolLayer_cpu(h_inMatrix_cpu, inShape, h_outMatrix_cpu, outShape, poolArgs);

		cudaMemGetInfo(&freeMem, &totalMem);
		printf("After Kernel Launch: Free memory: %lu bytes | Total memory: %lu bytes\n", freeMem, totalMem);

		//Device sync safety//
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "CUDA Kernel execution failed: " << cudaGetErrorString(err) << "\n";
			cudaFree(d_inMatrix);
			cudaFree(d_outMatrix);
			return -1;
		}

		//Device image to host image copy//
		cudaMemcpy(h_outMatrix, d_outMatrix, outMatrixSize, cudaMemcpyDeviceToHost);

		//Modified CPU version//
		//Printing the matrix for validation//
		std::cout << "Printing CPU Output Matrix!\n";
		std::cout << "Matrix Begin: \n[ ";
		for (uint32_t r = 0; r < outShape.height; ++r) {
			for (uint32_t c = 0; c < outShape.width; ++c) {
				float tmp_cpu = h_outMatrix_cpu[r * outShape.width + c];
				std::cout << tmp_cpu << ", ";
			}
			std::cout << "\n ";
		}
		std::cout << "]... \n\n";

		std::cout << "Printing GPU Output Matrix!\n";
		std::cout << "Matrix Begin: \n[ ";
		for (uint32_t r = 0; r < outShape.height; ++r) {
			for (uint32_t c = 0; c < outShape.width; ++c) {
				float tmp_gpu = h_outMatrix[r * outShape.width + c];
				std::cout << tmp_gpu << ", ";
			}
			std::cout << "\n ";
		}
		std::cout << "]... \n";

		cudaFree(d_inMatrix);
		cudaFree(d_outMatrix);
		free(h_inMatrix);
		free(h_outMatrix);
		free(h_inMatrix_cpu);
		free(h_outMatrix_cpu);
		
		std::cout << "Lazy, you are! ... ";
		std::cout << "Filter pixels, you must! ... ";

		return 0;
	}



__global__
void poolLayer_gpu(float* input, TensorShape inShape,
	float* output, TensorShape outShape, PoolLayerArgs args) {
	uint32_t outCol = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t outRow = blockIdx.y * blockDim.y + threadIdx.y;

	//setting row and col to the OUTPUTs as they are converting the input to the output
	//We want to split it up such that each thread takes a window//
	//So we could bounds check first to ensure col+poolW < inShape.W
	//if (outCol < outShape.width && outRow < outShape.height)

	if (outCol >= outShape.width || outRow >= outShape.height) { return; }
	else {
		//switch statement again which seems to be a bit overkill
		float poolPick = 0;

		switch (args.opType) {

		case PoolOp::MaxPool:
			poolPick = -INFINITY;
			break;

		case PoolOp::MinPool:
			poolPick = INFINITY;
			break;

		case PoolOp::AvgPool:
			poolPick = 0;
			break;
		default:
			poolPick = -INFINITY;
			//std::cout << "Defaulting to MAX POOL\n";
			break;
		}

		uint32_t inRow = outRow * args.strideH;
		uint32_t inCol = outCol * args.strideW;
		//if ((inRow + args.poolH) <= inShape.height && (inCol + args.poolW) <= inShape.width)
		if ((inRow + args.poolH) > inShape.height || (inCol + args.poolW) > inShape.width) { return; }
		else {
			for (uint32_t poolRow = 0; poolRow < args.poolH; ++poolRow) {
				for (uint32_t poolCol = 0; poolCol < args.poolW; ++poolCol) {
					//	STUDENT: Calculate row and col of element here
					// Doing the stride with outputH,W
					uint32_t row = inRow + poolRow;
					uint32_t col = inCol + poolCol;
					
					if (row < inShape.height && col < inShape.width) {
						float tmp = input[row * inShape.width + col];
						switch (args.opType) {
							//	STUDENT: Add cases and complete pooling code for all 3 options
						case PoolOp::MaxPool:
							poolPick = fmaxf(poolPick, tmp);
							break;

						case PoolOp::AvgPool:
							poolPick += tmp;
							break;

						case PoolOp::MinPool:
							poolPick = fminf(poolPick, tmp);
							break;

						default:
							//std::cout << "Pick max from pool, you must!\n";
							printf("Pick max from pool, you must!\n");
							//return 0;	//	STUDENT: Remove this as reqd.
							break;
						}
					}
				}
			}
			//If loop which just averages the poolPick value across the elements-size of the pool//
			if (args.opType == PoolOp::AvgPool) {
				poolPick /= (args.poolH * args.poolW);
			}

			output[outRow * outShape.width + outCol] = poolPick;
		}

	}
}

__global__
void poolLayer_gpu_shared(float* input, TensorShape inShape,
	float* output, TensorShape outShape, PoolLayerArgs args) {

	uint32_t outCol = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t outRow = blockIdx.y * blockDim.y + threadIdx.y;
	


	if (outCol >= outShape.width || outRow >= outShape.height) { return; }
	else {
		//switch statement again which seems to be a bit overkill
		float poolPick = 0;

		switch (args.opType) {

		case PoolOp::MaxPool:
			poolPick = -INFINITY;
			break;

		case PoolOp::MinPool:
			poolPick = INFINITY;
			break;

		case PoolOp::AvgPool:
			poolPick = 0;
			break;
		default:
			poolPick = -INFINITY;
			//std::cout << "Defaulting to MAX POOL\n";
			break;
		}
		uint32_t strideH = args.strideH;
		uint32_t strideW = args.strideW;

		uint32_t poolH = args.poolH;
		uint32_t poolW = args.poolW;

		uint32_t inRow = outRow * strideH;
		uint32_t inCol = outCol * strideW;
		
		__shared__ float tile[(TILE_SIZE + POOL_SIZE - 1)*(TILE_SIZE + POOL_SIZE - 1)];

		uint32_t sharedTileWidth = (TILE_SIZE + POOL_SIZE - 1);
		uint32_t sharedRow = threadIdx.y * strideH;
		uint32_t sharedCol = threadIdx.x * strideW;
		uint32_t poolTile = poolH * poolW; //16
		uint32_t poolTileRow = poolTile * outShape.width; //5*16

		//Loading this thread's respective portion of the input matrix into shared memory//
		//Assuming that everything that gets to this is safe//
		for (uint32_t r = 0; r < poolH; ++r) {
			for (uint32_t c = 0; c < poolW ; ++c) {
				uint32_t loadRow = sharedRow + r;
				uint32_t loadCol = sharedCol + c;
				uint32_t sharedTileOffset = (threadIdx.y * poolTileRow + threadIdx.x * poolTile); // Does have redudant memory values // 
				uint32_t sharedIdx = (loadRow * (poolW)+loadCol) + sharedTileOffset;
				uint32_t globalIdx = loadRow * inShape.width + loadCol;
				printf("sharedIdx: %u | globalIdx: %u | LocalRow: %u | LocalCol: %u | globalRow: %u | globalCol: %u | Global Value: %f\n", sharedIdx, globalIdx, r, c, sharedRow, sharedCol, input[loadRow * inShape.width + loadCol]);
				if(loadRow < inShape.height && loadCol < inShape.width){ tile[sharedIdx] = input[loadRow * inShape.width + loadCol]; }
				else {
					//For Max only//
					printf("!OOB TILE!  sharedIdx: %u | globalIdx: %u | LocalRow: %u | LocalCol: %u | globalRow: %u | globalCol: %u | Global Value: %f\n", sharedIdx, globalIdx, r, c, sharedRow, sharedCol, input[loadRow * inShape.width + loadCol]);
					tile[loadRow * poolW + loadCol] = 0;
				}
			}
		}

		__syncthreads();

		for (uint32_t r = 0; r < poolH; ++r) {
			for (uint32_t c = 0; c < poolW; ++c) {
				uint32_t loadRow = sharedRow + r;
				uint32_t loadCol = sharedCol + c;
				uint32_t sharedTileOffset = (threadIdx.y * poolTileRow + threadIdx.x * poolTile); // Does have redudant memory values // 
				uint32_t sharedIdx = (loadRow * (poolW)+loadCol) + sharedTileOffset;
				//printf("sharedIdx: %u | globalIdx: %u | LocalRow: %u | LocalCol: %u | globalRow: %u | globalCol: %u | Global Value: %f\n", sharedIdx, globalIdx, r, c, sharedRow, sharedCol, input[loadRow * inShape.width + loadCol]);
				float tmp = tile[sharedIdx];

				switch (args.opType) {
				case PoolOp::MaxPool:
					poolPick = fmaxf(poolPick, tmp);
					break;

				case PoolOp::AvgPool:
					poolPick += tmp;
					break;

				case PoolOp::MinPool:
					poolPick = fminf(poolPick, tmp);
					break;

				default:
					//std::cout << "Pick max from pool, you must!\n";
					printf("Pick max from pool, you must!\n");
					//return 0;	//	STUDENT: Remove this as reqd.
					break;
				}
			}
		}


		
		//If loop which just averages the poolPick value across the elements-size of the pool//
		if (args.opType == PoolOp::AvgPool) {
			poolPick /= (poolH * poolW);
		}
		printf("Pool pick: %f\n", poolPick);
		output[outRow * outShape.width + outCol] = poolPick;
	}
}

//	STUDENT: Add functions here

/*

for (uint32_t poolRow = 0; poolRow < poolH; ++poolRow) {
			for (uint32_t poolCol = 0; poolCol < poolW; ++poolCol) {
				uint32_t sharedIdx = (sharedRow + poolRow) * sharedTileWidth + (sharedCol + poolCol);
				float tmp = tile[sharedIdx];

				switch (args.opType) {
				case PoolOp::MaxPool:
					poolPick = fmaxf(poolPick, tmp);
					break;

				case PoolOp::AvgPool:
					poolPick += tmp;
					break;

				case PoolOp::MinPool:
					poolPick = fminf(poolPick, tmp);
					break;

				default:
					//std::cout << "Pick max from pool, you must!\n";
					printf("Pick max from pool, you must!\n");
					//return 0;	//	STUDENT: Remove this as reqd.
					break;
				}
			}
		}


*/