
__global__
void medianFilter_gpu(uint8_t* inPixels, ImageDim imgDim,
	uint8_t* outPixels, MedianFilterArgs args) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int w = imgDim.width;
	int h = imgDim.height;

	/* Outer Bounding Check */
	/* Doesn't account for non-square images--yet! */
	if (col < w && row < h) {
		//int pixVal = 0;
		//unsigned char window[(args.filterW * args.filterH)] -- can't do this because dynamic arrays 

		/* Setting the window size & Pixel Count*/
		uint8_t window[16]; // For filter width 4,4
		int pixels = 0;

		for (int medianRow = -args.filterH; medianRow < args.filterH + 1; ++medianRow) {
			for (int medianCol = -args.filterW; medianCol < args.filterW + 1; ++medianCol) {
				int curRow = row + medianRow;
				int curCol = col + medianCol;

				/* Filter bounds check, assuming it's inside the bounds then that pixel is added to the window array
					then incremented
				*/
				if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
					window[pixels++] = inPixels[curRow * w + curCol];
				}
			}
		}

		for (int i = 0; i < pixels - 1; i++) {
			for (int j = i + 1; j < pixels; j++) {
				if (window[i] > window[j]) {
					uint8_t tmp = window[i];
					window[i] = window[j];
					window[j] = tmp;
				}
			}
		}

		outPixels[row * w + col] = window[pixels / 2];
	}
}

//int medianFilter_cpu (uint8_t * inPixels, ImageDim imgDim, uint8_t * outPixels, MedianFilterArgs args)
/*
__global__
void medianFilterKernel(uint8_t* in, uint8_t* out, int w, int h, int filterW, int filterH) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < w && row < h) {
		//int pixVal = 0;
		//unsigned char window[(args.filterW * args.filterH)] -- can't do this because dynamic arrays
		uint8_t window[16]; // For filter width 4,4
		int pixels = 0;

		for (int medianRow = -filterH; medianRow < filterH+1; ++medianRow) {
			for (int medianCol = -filterW; medianCol < filterW+1; ++medianCol) {
				int curRow = row + medianRow;
				int curCol = col + medianCol;

				if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
					pixVal += in[curRow * w + curCol];
					++pixels;
				}
			}
		}
		out[row * w + col] = (unsigned char)(pixVal / pixels);
	}
}
*/

//functional//
__global__
void medianFilter_gpu(uint8_t* inPixels, ImageDim imgDim, uint8_t* outPixels, MedianFilterArgs args) {
	//Thread declarations
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	uint8_t testVal = 0;

	/* Outer Bounding Check */
	/* Doesn't account for non-square images--yet! */
	if (col < imgDim.width && row < imgDim.height) {
		/* Setting the window size & Pixel Count*/
		uint8_t window[16]; // For filter width 4,4
		int pixels = 0;
		for (uint32_t channel = 0; channel < imgDim.channels; ++channel) {
			for (int medianRow = -args.filterH; medianRow < args.filterH + 1; ++medianRow) {
				for (int medianCol = -args.filterW; medianCol < args.filterW + 1; ++medianCol) {
					int curRow = row + medianRow;
					int curCol = col + medianCol;
					/* Filter bounds check, assuming it's inside the bounds then that pixel is added to the window array
						then incremented*/
					if (curRow >= 0 && curRow < imgDim.height && curCol >= 0 && curCol < imgDim.width) {
						window[pixels++] = inPixels[curRow * imgDim.width + curCol];
						testVal = inPixels[curRow * imgDim.width + curCol];
						printf("Testval: %d", testVal);
					}
				}
			}
			/* Swap Sort */
			for (int i = 0; i < pixels - 1; i++) {
				for (int j = i + 1; j < pixels; j++) {
					if (window[i] > window[j]) {
						uint8_t tmp = window[i];
						window[i] = window[j];
						window[j] = tmp;
					}
				}
			}
			for (int k = 0;k < 16;k++) { printf("Window Array Spot: %d | Value: %u \n", k, window[k]); }
			outPixels[(row * imgDim.width + col) * imgDim.channels + channel] = window[pixels / 2];
			//outPixels[(outRow * imgDim.width + outCol) * imgDim.channels + channel] = window[window.size() / 2];
			//printf("Channel: %d", channel);
		}
	}
}

__global__
void medianFilter_gpu(uint8_t* inPixels, ImageDim imgDim, uint8_t* outPixels, MedianFilterArgs args) {
	//Thread declarations
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	uint8_t testVal = 0;

	//Sanity Checks//
	//printf("dimHeight: %d | dimWidth: %d \n", imgDim.height, imgDim.width);
	//printf("dimChannel: %d \n", imgDim.channels);
	//printf("fH: %d | fW: %d \n", (-args.filterW), (-args.filterW));

	/* Outer Bounding Check */
	/* Doesn't account for non-square images--yet! */
	if (col < imgDim.width && row < imgDim.height) {
		/* Setting the window size & Pixel Count*/
		//printf("Bounds Check row/col");
		uint8_t window[16]; // For filter width 4,4
		int pixels = 0;
		for (uint32_t channel = 0; channel < imgDim.channels; ++channel) {
			//printf("Bounds Check channels: %d",channel);
			for (int medianRow = -4; medianRow < 5; ++medianRow) {
				//printf("Bounds Check medianRow");
				for (int medianCol = -4; medianCol < 5; ++medianCol) {
					//printf("Bounds Check medianCol");
					int curRow = row + medianRow;
					int curCol = col + medianCol;
					//printf("CurRow: %d | CurCol: %d \n", curRow, curCol);
					/* Filter bounds check, assuming it's inside the bounds then that pixel is added to the window array
						then incremented*/
					if (curRow >= 0 && curRow < imgDim.height && curCol >= 0 && curCol < imgDim.width) {
						window[pixels++] = inPixels[curRow * imgDim.width + curCol];
						testVal = inPixels[curRow * imgDim.width + curCol];
						//printf("Pixel Count: %d", pixels);
					}
				}
			}
			/* Swap Sort */
			for (int i = 0; i < pixels - 1; ++i) {
				for (int j = i + 1; j < pixels; ++j) {
					if (window[i] > window[j]) {
						uint8_t tmp = window[i];
						window[i] = window[j];
						window[j] = tmp;
					}
				}
			}
			//for (int k = 0; k < 16 ; ++k) { printf("Window Array Spot: %d | Value: %u \n", k, window[k]); }
			outPixels[(row * imgDim.width + col) * imgDim.channels + channel] = window[pixels / 2];
			//outPixels[(outRow * imgDim.width + outCol) * imgDim.channels + channel] = window[window.size() / 2];
			//printf("Channel: %d", channel);
		}
	}
}
__global__
void medianFilter_gpu(uint8_t* inPixels, ImageDim imgDim, uint8_t* outPixels, MedianFilterArgs args) {
	//Thread declarations
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	uint8_t testVal = 0;
	int uniqueArray = 0;
	//Sanity Checks//
	//printf("dimHeight: %d | dimWidth: %d \n", imgDim.height, imgDim.width);
	//printf("dimChannel: %d \n", imgDim.channels);
	//printf("fH: %d | fW: %d \n", (-args.filterW), (-args.filterW));

	/* Outer Bounding Check */
	/* Doesn't account for non-square images--yet! */
	if (col < imgDim.width && row < imgDim.height) {
		/* Setting the window size & Pixel Count*/
		//printf("Bounds Check row/col");
		uint8_t window[16]; // For filter width 4,4
		int pixels = 0;
		for (uint32_t channel = 0; channel < imgDim.channels; ++channel) {
			//printf("Bounds Check channels: %d",channel);
			for (int medianRow = -4; medianRow < 5; ++medianRow) {
				//printf("Bounds Check medianRow");
				for (int medianCol = -4; medianCol < 5; ++medianCol) {
					//printf("Bounds Check medianCol");
					int curRow = row + medianRow;
					int curCol = col + medianCol;
					//printf("CurRow: %d | CurCol: %d \n", curRow, curCol);
					/* Filter bounds check, assuming it's inside the bounds then that pixel is added to the window array
						then incremented*/
					if (curRow >= 0 && curRow < imgDim.height && curCol >= 0 && curCol < imgDim.width) {
						window[pixels++] = inPixels[curRow * imgDim.width + curCol];
						testVal = inPixels[curRow * imgDim.width + curCol];
						//printf("Test Val: %d \n", testVal);
					}
					else {
						printf("OoB Pixel: (%d,%d) | Source Pixel: (%d,%d)\n", curRow, curCol, row, col);
					}
				}
			}
			/* Swap Sort */
			for (int i = 0; i < pixels - 1; ++i) {
				for (int j = i + 1; j < pixels; ++j) {
					if (window[i] > window[j]) {
						uint8_t tmp = window[i];
						window[i] = window[j];
						window[j] = tmp;
					}
				}
			}
			//for (int k = 0; k < 16 ; ++k) { printf("Unique Array#:%d | Spot: %d | Value: %u \n", uniqueArray, k, window[k]); }
			outPixels[(row * imgDim.width + col) * imgDim.channels + channel] = window[pixels / 2];
			uniqueArray++;
			//outPixels[(outRow * imgDim.width + outCol) * imgDim.channels + channel] = window[window.size() / 2];
			//printf("Channel: %d", channel);
		}
	}
}

__global__
void medianFilter_gpu(uint8_t* inPixels, ImageDim imgDim, uint8_t* outPixels, MedianFilterArgs args) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < imgDim.width && row < imgDim.height) {
		uint8_t window[16]; // For filter width 4,4
		int pixels = 0;
		for (uint32_t channel = 0; channel < imgDim.channels; ++channel) {
			for (int medianRow = -4; medianRow < 5; ++medianRow) {
				for (int medianCol = -4; medianCol < 5; ++medianCol) {
					int curRow = row + medianRow;
					int curCol = col + medianCol;
					if (curRow >= 0 && curRow < imgDim.height && curCol >= 0 && curCol < imgDim.width) {
						window[pixels++] = inPixels[curRow * imgDim.width + curCol];
					}
				}
			}
			/* Swap Sort */
			for (int i = 0; i < pixels - 1; ++i) {
				for (int j = i + 1; j < pixels; ++j) {
					if (window[i] > window[j]) {
						uint8_t tmp = window[i];
						window[i] = window[j];
						window[j] = tmp;
					}
				}
			}
			//for (int k = 0; k < 16 ; ++k) { printf("Unique Array#:%d | Spot: %d | Value: %u \n", uniqueArray, k, window[k]); }
			outPixels[(row * imgDim.width + col) * imgDim.channels + channel] = window[pixels / 2];
		}
	}
}


//Completely funcitonal

int runGpuMedianFilter(std::string imgPath, std::string outPath, MedianFilterArgs args) {

	ImageDim h_imgDim;
	uint8_t* h_imgData;

	int bytesRead = loadBytesImage(imgPath, h_imgDim, &h_imgData);
	int img_size = h_imgDim.height * h_imgDim.width * h_imgDim.channels * h_imgDim.pixelSize;

	std::cout << "Size = " << img_size << "\n";
	//h_outData = *h_imgData;
	uint8_t* h_outData = (uint8_t*)malloc(img_size * sizeof(uint8_t));

	//Now allocating memory for matrices on the device
	uint8_t* d_imgData, * d_outData;

	cudaMalloc((void**)&d_imgData, img_size * sizeof(uint8_t));
	cudaMalloc((void**)&d_outData, img_size * sizeof(uint8_t));

	//Now copying the input matrices M and N from H to D
	cudaMemcpy(d_imgData, h_imgData, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_outData, h_outData, imgSize * sizeof(uint8_t), cudaMemcpyHostToDevice);

	//dim3 blockSize(FILTER_H, FILTER_W,1 ); // 4x4 TBs
	//dim3 gridSize(h_imgDim.height / FILTER_H, h_imgDim.width / FILTER_W, 1);
	dim3 blockSize(16, 16, 1); // 4x4 TBs
	dim3 gridSize((h_imgDim.height + 15) / 16, (h_imgDim.width + 15) / 16, 1);


	std::cout << "Grid H is: " << ((h_imgDim.height + 15) / 16) << "\n";
	std::cout << "Grid W is: " << (h_imgDim.width + 15) / 16 << "\n";

	medianFilter_gpu << <gridSize, blockSize >> > (d_imgData, h_imgDim, d_outData, args);

	cudaMemcpy(h_outData, d_outData, img_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);


	writeBytesImage(outPath, h_imgDim, h_outData);

	cudaFree(d_imgData);
	cudaFree(d_outData);

	std::cout << "Lazy, you are! ... ";
	std::cout << "Filter pixels, you must! ... ";

	return 0;
}

__global__
void medianFilter_gpu(uint8_t* inPixels, ImageDim imgDim, uint8_t* outPixels, MedianFilterArgs args) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int fH = 3;
	int fW = 3;


	//int fH = args.filterH;
	//int fW = args.filterW;

	if (col < imgDim.width && row < imgDim.height) {
		uint8_t window[81]; // For filter width 4,4
		int pixels = 0;
		for (int channel = 0; channel < imgDim.channels; ++channel) {
			for (int medianRow = ((-fH) / 2); medianRow <= (fH / 2); ++medianRow) {
				for (int medianCol = ((-fW) / 2); medianCol <= (fW / 2); ++medianCol) {
					int curRow = row + medianRow;
					int curCol = col + medianCol;
					if (curRow >= 0 && curRow < imgDim.height && curCol >= 0 && curCol < imgDim.width) {
						window[pixels++] = inPixels[(curRow * imgDim.width + curCol) * imgDim.channels + channel];
					}
				}
			}
			/* Swap Sort */
			for (int i = 0; i < pixels - 1; ++i) {
				for (int j = i + 1; j < pixels; ++j) {
					if (window[i] > window[j]) {
						uint8_t tmp = window[i];
						window[i] = window[j];
						window[j] = tmp;
					}
				}
			}
			//int uniqueArray = (row * imgDim.width + col) * imgDim.channels + channel;
			//for (int k = 0; k < pixels ; ++k) { printf("Unique Array#:%d | Spot: %d | Value: %u \n", uniqueArray, k, window[k]); }
			outPixels[(row * imgDim.width + col) * imgDim.channels + channel] = window[pixels / 2];
		}
	}
}

extern __global__ void medianFilter_gpu(uint8_t* inPixels, ImageDim imgDim,
	uint8_t* outPixels, MedianFilterArgs args);


__global__
void medianFilter_gpu4(uint8_t* inPixels, uint8_t* outPixels,
	int height, int width, int channels, int filterH, int filterW) {

	extern __shared__ uint8_t sharedMem[];

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int localCol = threadIdx.x + (filterW) / 2;
	//int localCol = threadIdx.x + 1;
	int localRow = threadIdx.y + (filterH) / 2;
	//int localRow = threadIdx.y + 1;

	//int sharedW = blockDim.x + filterW - 1;
	//int sharedH = blockDim.y + filterH - 1;
	int sharedW = blockDim.x + filterW - 1;
	int sharedH = blockDim.y + filterH - 1;

	int globalIndex = (row * width + col) * channels;

	uint8_t* sharedPixels = sharedMem;

	if (col < width && row < height) {
		for (int channel = 0; channel < channels; ++channel) {
			int sharedIndex = (localRow * sharedW + localCol) * channels + channel;
			if (col < width && row < height) {
				sharedPixels[sharedIndex] = inPixels[globalIndex + channel];
				//printf("Real Pixel at: %d \n", sharedIndex);
			}
			else {
				//printf("Fake Pixel at: %d \n", sharedIndex);
				sharedPixels[sharedIndex] = 0;
			}
		}
	}

	__syncthreads();

	if (col < width && row < height) {
		uint8_t window[81];
		int pixels = 0;
		for (int channel = 0; channel < channels; ++channel) {
			pixels = 0;
			for (int medianRow = ((-filterH) / 2); medianRow <= (filterH / 2); ++medianRow) {
				for (int medianCol = ((-filterW) / 2); medianCol <= (filterW / 2); ++medianCol) {
					int curRow = localRow + medianRow;
					//int curRow = threadIdx.y + medianRow;
					int curCol = localCol + medianCol;
					//int curCol = threadIdx.x + medianCol;

					int sharedIdx = (curRow * sharedW + curCol) * channels + channel;
					window[pixels++] = sharedPixels[sharedIdx];
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
			//__syncthreads();
			outPixels[(row * width + col) * channels + channel] = window[(pixels) / 2];
		}
	}
}

__global__
void medianFilter_gpu4(uint8_t* inPixels, uint8_t* outPixels,
	int height, int width, int channels, int filterH, int filterW) {

	extern __shared__ uint8_t sharedMem[];

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int localCol = threadIdx.x + (filterW) / 2;
	//int localCol = threadIdx.x;
	int localRow = threadIdx.y + (filterH) / 2;
	//int localRow = threadIdx.y;

	int sharedW = blockDim.x + filterW - 1;
	int sharedH = blockDim.y + filterH - 1;


	int globalIndex = (row * width + col) * channels;

	uint8_t* sharedPixels = sharedMem;

	if (col < width && row < height) {
		for (int channel = 0; channel < channels; ++channel) {
			if (col < width && row < height) {
				sharedPixels[((threadIdx.y * sharedW + threadIdx.x) * channels + channel)] = inPixels[globalIndex + channel];
				//printf("Shared pixels -- Col %d | Row %d | GlobalIdx %d \n", localCol, localRow, globalIndex);
			}
			else {
				sharedPixels[((threadIdx.y * sharedW + threadIdx.x) * channels + channel)] = 0;
			}
		}
	}

	__syncthreads();

	if (col < width && row < height) {
		uint8_t window[81];
		int pixels = 0;
		for (int channel = 0; channel < channels; ++channel) {
			pixels = 0;
			for (int medianRow = ((-filterH) / 2); medianRow <= (filterH / 2); ++medianRow) {
				int curRow = localRow + medianRow;
				for (int medianCol = ((-filterW) / 2); medianCol <= (filterW / 2); ++medianCol) {
					int curCol = localCol + medianCol;
					//printf("localRow: %d | CurRow: %d | localCol: %d | curCol: %d\n",localRow, curRow, localCol, curCol);
					//int sharedIdx = (curRow * sharedW + curCol) * channels + channel;
					if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
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
			outPixels[(row * width + col) * channels + channel] = window[(pixels) / 2];
		}
	}
}

for (int medianRow = -1 + ((-filterH) / 2); medianRow < -1 + (filterH / 2); ++medianRow) {
	int curRow = threadIdx.y + medianRow;
	for (int medianCol = -1 + ((-filterW) / 2); medianCol < -1 + (filterW / 2); ++medianCol) {
		int curCol = threadIdx.x + medianCol;
		//printf("localRow: %d | CurRow: %d | localCol: %d | curCol: %d\n",localRow, curRow, localCol, curCol);
		//int sharedIdx = (curRow * sharedW + curCol) * channels + channel;
		if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
			window[pixels++] = sharedPixels[((curRow * sharedW + curCol) * channels + channel)];
		}

	}
}

//Shared Memory GPU Filter//
__global__
void medianFilter_gpu4(uint8_t* inPixels, uint8_t* outPixels,
	int height, int width, int channels, int filterH, int filterW) {

	extern __shared__ uint8_t sharedMem[];

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	//int localCol = threadIdx.x + (filterW) / 2;
	int localCol = threadIdx.x;
	//int localRow = threadIdx.y + (filterH) / 2;
	int localRow = threadIdx.y;

	int sharedW = blockDim.x + filterW;
	//int sharedH = blockDim.y + filterH+1;


	int globalIndex = (row * width + col) * channels;

	uint8_t* sharedPixels = sharedMem;

	if (col < width && row < height) {
		for (int channel = 0; channel < channels; ++channel) {
			if (col < width && row < height) {
				sharedPixels[((threadIdx.y * sharedW + threadIdx.x) * channels + channel)] = inPixels[globalIndex + channel];
				//printf("Shared pixels -- Col %d | Row %d | GlobalIdx %d \n", localCol, localRow, globalIndex);
			}
			else {
				sharedPixels[((threadIdx.y * sharedW + threadIdx.x) * channels + channel)] = 0;
			}
		}
	}

	__syncthreads();

	if (col < width && row < height) {
		uint8_t window[81];
		int pixels = 0;
		for (int channel = 0; channel < channels; ++channel) {
			pixels = 0;
			for (int medianRow = ((-filterH) / 2); medianRow < (filterH / 2); ++medianRow) {
				int curRow = threadIdx.y + medianRow;
				for (int medianCol = ((-filterW) / 2); medianCol < (filterW / 2); ++medianCol) {
					int curCol = threadIdx.x + medianCol;
					//printf("localRow: %d | CurRow: %d | localCol: %d | curCol: %d\n",localRow, curRow, localCol, curCol);
					//int sharedIdx = (curRow * sharedW + curCol) * channels + channel;
					if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
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
			outPixels[(row * width + col) * channels + channel] = window[(pixels) / 2];
		}
	}
}


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
		uint8_t window[81];  // Window array
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

if (localCol == (filterW / 2)) {
	//The top left corner? (2,2)
	if (localRow == (filterH / 2)) {
		//If it's in the top left corner, is in globally in the TLC?
		if (globalCol == 0 && globalRow == 0) {
			//Yay

		}
		else {
			//Nay
		}
	}//The bottom left corner? (2,17)
	else if (localRow == (sharedH - (filterH / 2))) {
		//do
	}//Just the left border?
	else {
		//do
	}//The Right Border?
}
else if (localCol == (sharedW - (filterW / 2))) {
	//The top right corner? (17,2)
	if (localRow == (filterH / 2)) {
		//do
	}//The bottom right corner? (17,17)
	else if (localRow == (sharedH - (filterH / 2))) {
		//do
	}//Just the right border?
	else {
		//do
	}
}

__global__
void medianFilter_gpu4(uint8_t* inPixels, uint8_t* outPixels,
	int height, int width, int channels, int filterH, int filterW) {

	extern __shared__ uint8_t sharedMem[];

	int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
	int globalRow = blockIdx.y * blockDim.y + threadIdx.y;

	//Boundary Check for OOBs//
	if (globalCol >= width || globalRow >= height) return;

	int localCol = threadIdx.x + (filterW) / 2;
	int localRow = threadIdx.y + (filterH) / 2;


	//Padding with 2*(FILTER/2) for Halo//
	int sharedW = blockDim.x + 2 * (filterW / 2);
	int sharedH = blockDim.y + 2 * (filterH / 2);

	if (localCol >= sharedW || localRow >= sharedH) return;

	int globalIndex = (globalRow * width + globalCol) * channels;

	// Global Bounds Check// 
	bool isGlobalBorder = (globalCol == 0 || globalCol == (width - 1) ||
		globalRow == 0 || globalRow == (height - 1));

	uint8_t* sharedPixels = sharedMem;

		// Since the pixel is IN the image we can initiate the channel loop//
		for (int channel = 0; channel < channels; ++channel) {
			// Is the Pixel on the Global Border?
			if (isGlobalBorder) {
				// If it's a global border, clamp to the nearest valid pixel//
				int clampedRow = min(max(globalRow, 0), height - 1);
				int clampedCol = min(max(globalCol, 0), width - 1);
				sharedPixels[((localRow * sharedW + localCol) * channels + channel)] =
					inPixels[(clampedRow * width + clampedCol) * channels + channel];
			} else if ((localCol >= (filterW / 2) && localCol < (sharedW - (filterW / 2)))
				&& (localRow >= (filterH / 2) && localRow < (sharedH - (filterH / 2)))) {
				// Core Processing Area (CPA) Check, is the pixel IN the tile? Min (2,2) Max (17,17)//
				sharedPixels[((localRow * sharedW + localCol) * channels + channel)] = inPixels[globalIndex + channel];
			}
			else {
				// The pixel is IN the halo (non-CPA) and we assign a zero to it.//
				sharedPixels[((localRow * sharedW + localCol) * channels + channel)] = 0;
			}
		}
	}


__global__
void medianFilter_gpu4(uint8_t* inPixels, uint8_t* outPixels,
	int height, int width, int channels, int filterH, int filterW) {

	extern __shared__ uint8_t sharedMem[];

	int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
	int globalRow = blockIdx.y * blockDim.y + threadIdx.y;

	//Boundary Check for OOBs//
	if (globalCol >= width || globalRow >= height) return;

	int localCol = threadIdx.x + (filterW) / 2;
	int localRow = threadIdx.y + (filterH) / 2;


	//Padding with 2*(FILTER/2) for Halo//
	int sharedW = blockDim.x + 2 * (filterW / 2);
	int sharedH = blockDim.y + 2 * (filterH / 2);

	if (localCol >= sharedW || localRow >= sharedH) return;

	int globalIndex = (globalRow * width + globalCol) * channels;

	// Global Bounds Check// 
	bool isGlobalBorder = (globalCol == 0 || globalCol == (width - 1) ||
		globalRow == 0 || globalRow == (height - 1));

	uint8_t* sharedPixels = sharedMem;

	//Boundary Check #1, is the pixel IN the image?//
	if (globalCol < width && globalRow < height) {
		// Since the pixel is IN the image we can initiate the channel loop//
		for (int channel = 0; channel < channels; ++channel) {
			// Core Processing Area (CPA) Check, is the pixel IN the tile? Min (2,2) Max (17,17)//
			if ((localCol >= (filterW / 2) && localCol < (sharedW - (filterW / 2)))
				&& (localRow >= (filterH / 2) && localRow < (sharedH - (filterH / 2)))) {

				// If it's on the corner then cR =0, cC=0
				if (isGlobalBorder) {
					// If it's a global border, clamp to the nearest valid pixel
					int clampedRow = min(max(globalRow, 0), height - 1);
					int clampedCol = min(max(globalCol, 0), width - 1);
					sharedPixels[((localRow * sharedW + localCol) * channels + channel)] =
						inPixels[(clampedRow * width + clampedCol) * channels + channel];
				}
				else {
					// Pixel is g2g, inside the CPA and not on a global border
					sharedPixels[((localRow * sharedW + localCol) * channels + channel)] =
						inPixels[globalIndex + channel];
				}
			}
			else {
				// The pixel is IN the halo (non-CPA) and we assign a zero to it.
				sharedPixels[((localRow * sharedW + localCol) * channels + channel)] = 0;
			}
		}
	}

	__global__
		void medianFilter_gpu4(uint8_t * inPixels, uint8_t * outPixels,
			int height, int width, int channels, int filterH, int filterW) {

		extern __shared__ uint8_t sharedMem[];

		int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
		int globalRow = blockIdx.y * blockDim.y + threadIdx.y;

		//Boundary Check for OOBs//
		if (globalCol >= width || globalRow >= height) return;

		int localCol = threadIdx.x + (filterW) / 2;
		int localRow = threadIdx.y + (filterH) / 2;


		//Padding with 2*(FILTER/2) for Halo//
		int sharedW = blockDim.x + 2 * (filterW / 2);
		int sharedH = blockDim.y + 2 * (filterH / 2);

		if (localCol >= sharedW || localRow >= sharedH) return;

		int globalIndex = (globalRow * width + globalCol) * channels;

		// Global Bounds Check// 
		bool isGlobalBorder = (globalCol == 0 || globalCol == (width - 1) ||
			globalRow == 0 || globalRow == (height - 1));

		uint8_t* sharedPixels = sharedMem;

		// Since the pixel is IN the image we can initiate the channel loop//
		for (int channel = 0; channel < channels; ++channel) {
			int sharedIndex = ((localRow * sharedW + localCol) * channels + channel);
			// Is the Pixel on the Global Border?
			if (isGlobalBorder) {
				// If it's a global border, clamp to the nearest valid pixel//
				int clampedRow = min(max(globalRow, 0), height - 1);
				int clampedCol = min(max(globalCol, 0), width - 1);
				sharedPixels[sharedIndex] =
					inPixels[(clampedRow * width + clampedCol) * channels + channel];
				//printf("GlobalBorder\n");
			}
			else if ((localCol > (filterW / 2) && localCol < (sharedW - (filterW / 2) - 1))
				&& (localRow > (filterH / 2) && localRow < (sharedH - (filterH / 2) - 1))) {
				// Core Processing Area (CPA) Check, is the pixel IN the tile? Min (2,2) Max (17,17)//
				sharedPixels[sharedIndex] = inPixels[globalIndex + channel];
				//printf("CPA \n");
			}
			else {
				// The pixel is IN the halo (non-CPA) and we assign a zero to it.//
				// Could clamp to fix blurring //
				//sharedPixels[sharedIndex] = 69;
				int clampedRow = min(max(globalRow, 0), height - 1);
				int clampedCol = min(max(globalCol, 0), width - 1);
				sharedPixels[sharedIndex] = inPixels[(clampedRow * width + clampedCol) * channels + channel];
				//printf("Clamping away!\n");	
			}
		}

		__syncthreads();

		// Now for the median filter'ng//
		//Each thread has it's pixel and surrounding window to perform the median filtering
		//Assuming a static 9x9 (max) Filter Window//
		uint8_t window[81];
		int pixels = 0;
		for (int channel = 0; channel < channels; ++channel) {
			pixels = 0;
			//Start by setting the filter row (starts at -2, ends @ +2) for f=4//
			for (int medianRow = ((-filterH) / 2); medianRow <= (filterH / 2); ++medianRow) {
				//Start by setting the filter col (starts at -2, ends @ +2) for f=4//
				for (int medianCol = ((-filterW) / 2); medianCol <= (filterW / 2); ++medianCol) {
					//For tx,ty (0,0) w/F=4 this is cR = 0 + 2 + -2 which is (0,0) of the sharedmem square!//
					int curRow = localRow + medianRow;
					int curCol = localCol + medianCol;

					if (curRow >= 0 && curRow < sharedH && curCol >= 0 && curCol < sharedW) {
						window[pixels++] = sharedPixels[(curRow * sharedW + curCol) * channels + channel];
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
			//Now to take the result and push to global//
			int globalOutIdx = (globalRow * width + globalCol) * channels + channel;
			if (globalCol < width && globalRow < height) {
				outPixels[globalOutIdx] = window[(pixels + 1) / 2];
			}

		}
	}

	int poolLayer_cpu(float* input, TensorShape inShape,
		float* output, TensorShape outShape, PoolLayerArgs args) {
		float poolPick;

		uint32_t poolH = args.poolH;
		uint32_t poolW = args.poolW;
		//	STUDENT: Calculate or unpack TensorShapes
		std::cout << "Lazy, you are! ... ";
		uint32_t outputH = 1;
		uint32_t outputW = 1;
		uint32_t row, col;

		std::cout << args.opType << " : " << inShape.height << " x " << inShape.width
			<< " with a " << poolH << " x " << poolW << " window -> "
			<< outputH << " x " << outputW << "\n";

		for (uint32_t outRow = 0; outRow < outputH; ++outRow) {
			for (uint32_t outCol = 0; outCol < outputW; ++outCol) {
				//	STUDENT: Assign to first value of pool area
				// poolPick = 0; 

				for (uint32_t poolRow = 0; poolRow < args.poolH; ++poolRow) {
					for (uint32_t poolCol = 0; poolCol < args.poolW; ++poolCol) {
						//	STUDENT: Calculate row and col of element here
						switch (args.opType)
						{
							//	STUDENT: Add cases and complete pooling code for all 3 options
						case PoolOp::MaxPool:

						default:
							std::cout << "Pick max from pool, you must!\n";
							return 0;	//	STUDENT: Remove this as reqd.
							break;
						}
					}
				}
				std::cout << poolPick << " @ (" << outRow << ", " << outCol << ")\n";
			}
		}
	}

	int runCpuPool(TensorShape inShape, PoolLayerArgs poolArgs) {

		srand(time(NULL));

		//	STUDENT: Initialize required memories));
		std::cout << "Set Tensors to stun !!";

		//	STUDENT: call pool function
		//	poolLayer_cpu(inMatrix, inShape, outMatrix, outShape, poolArgs);

		return 0;
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
		float* h_outMatrix = (float*)malloc(outShape.height * outShape.width * sizeof(float));

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
				std::cout << randomValue << ", ";
			}
			std::cout << "\n ";
		}
		std::cout << "]... ";
		std::cout << "Set Tensors to stun !!";

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
		//dim3 gridSize((inShape.width + (TILE_SIZE - 1)) / TILE_SIZE, (inShape.height + (TILE_SIZE - 1)) / TILE_SIZE, 1);
		dim3 gridSize((outputW + (TILE_SIZE - 1)) / TILE_SIZE, ((outputH + (TILE_SIZE - 1)) / TILE_SIZE));

		//Setting Window Array size for the shared memory option//
		//int windowSize = (TILE_SIZE + args.filterH - 1) * (TILE_SIZE + args.filterW - 1) * h_imgDim.channels;
		//size_t sharedMemSize = windowSize * sizeof(uint8_t);

		//Diagnostics just to see memory consumption//
		size_t freeMem, totalMem;
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("Before Kernel Launch: Free memory: %lu bytes | Total memory: %lu bytes\n", freeMem, totalMem);

		poolLayer_gpu << <gridSize, blockSize >> > (d_inMatrix, inShape, d_outMatrix, outShape, poolArgs);

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
		std::cout << "Printing Output Matrix!\n";
		std::cout << "Matrix Begin: [ ";
		for (uint32_t r = 0; r < outShape.height; ++r) {
			for (uint32_t c = 0; c < outShape.width; ++c) {
				float tmp = h_outMatrix[r * outShape.width + c];
				std::cout << tmp << ", ";
			}
			std::cout << "\n ";
		}
		std::cout << "]... ";

		cudaFree(d_inMatrix);
		cudaFree(d_outMatrix);
		free(h_inMatrix);
		free(h_outMatrix);

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

		if (outCol < outShape.width && outRow < outShape.height) {
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
			if ((inRow + args.poolH) <= inShape.height && (inCol + args.poolW) <= inShape.width) {
				for (uint32_t poolRow = 0; poolRow < args.poolH; ++poolRow) {
					for (uint32_t poolCol = 0; poolCol < args.poolW; ++poolCol) {
						//	STUDENT: Calculate row and col of element here
						// Doing the stride with outputH,W
						uint32_t row = inRow + poolRow;
						uint32_t col = inCol + poolCol;
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
				//If loop which just averages the poolPick value across the elements-size of the pool//
				if (args.opType == PoolOp::AvgPool) {
					poolPick /= (args.poolH * args.poolW);
				}

				output[outRow * outShape.width + outCol] = poolPick;
			}

		}
	}

	for (uint32_t r = 0; r < outShape.height; ++r) {
		for (uint32_t c = 0; c < outShape.width; ++c) {
			float tmp_gpu = h_outMatrix[r * outShape.width + c];
			float tmp_cpu = h_outMatrix_cpu[r * outShape.width + c];
			if (tmp_gpu != tmp_cpu) {

				std::cout << "Error at R: " << r << "| C: " << c << ", ";
				std::cout << "CPU: " << tmp_cpu << "| GPU: " << tmp_gpu << ", ";
			}
			else { std::cout << tmp_gpu << ", "; }
		}
		std::cout << "\n ";
	}
	std::cout << "]... \n";