#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>       // clock_t, clock, CLOCKS_PER_SEC

// Copy by using multiple blocks
__global__ void kernel_function_copy(unsigned int *const dst_ptr, unsigned int *const src_ptr) {
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	dst_ptr[tid] = src_ptr[tid];
}
// ------------------------------------------------------------

// Copy by using single block
template<size_t unroll_count>
__global__ void kernel_function_copy_xb(unsigned int *const dst_ptr, unsigned int *const src_ptr, const size_t c_array_size) {
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	for(int k = 0; k < c_array_size; k += unroll_count * blockDim.x * gridDim.x) {
		#pragma unroll
		for(size_t i = 0; i < unroll_count; ++i)
			dst_ptr[tid + i*blockDim.x*gridDim.x + k] = src_ptr[tid + i*blockDim.x*gridDim.x + k];
	}
}
// ------------------------------------------------------------


bool compare_arrays(volatile unsigned char *const host_src_ptr, volatile unsigned char *const host_dst_ptr, const size_t c_array_size) {
	size_t i = 0;
	for(i = 0; i < c_array_size; ++i)
		if (host_src_ptr[i] != host_dst_ptr[i]) {
			std::cout << "find difference in: " << i << std::endl; 
			return false;
		}
	/*std::cout << "host_src_ptr & host_dst_ptr are equal " << std::endl;*/ 
	return true;
}
// ------------------------------------------------------------

// DMA copy test-case
void dma_copy_host_device_host(unsigned char *const host_dst_ptr, unsigned char *const host_src_ptr, unsigned char *const dev_ptr, const size_t c_array_size) 
{
	// ------------------------------------ DMA -----------------------------------
	clock_t end, start;
	size_t iterations = 0;

	// Gen data
	srand (time(NULL));
	for(size_t i = 0; i < c_array_size; ++i) {
		const_cast<volatile unsigned char *>(host_src_ptr)[i] = rand() % 256;
		const_cast<volatile unsigned char *>(host_dst_ptr)[i] = 0;
	}
	cudaMemcpy(dev_ptr, host_src_ptr, c_array_size, cudaMemcpyDefault);	// fill zero GPU memory

	cudaDeviceSynchronize();
	start = clock();

	// copy data with DMA
	do {	
		cudaMemcpy(dev_ptr, host_src_ptr, c_array_size, cudaMemcpyDefault);	// Host to Device
		cudaMemcpy(host_dst_ptr, dev_ptr, c_array_size, cudaMemcpyDefault);	// Device to Host
		++iterations;
	} while(clock() - start < CLOCKS_PER_SEC);
	end = clock();

//	std::cout << "----------------------------------------------------- \n";
	const float c_time = (float)(end - start)/(CLOCKS_PER_SEC*iterations);
	std::cout << "DMA: block_size = " << c_array_size << 
		", \t time H->D->H: " << c_time << 
		", " << 2*c_array_size/(c_time * 1024*1024) << " MB/sec" << std::endl;

	// Compare data
	compare_arrays(host_src_ptr, host_dst_ptr, c_array_size);

//	std::cout << "host_src_ptr[0] = " << (unsigned)(host_src_ptr[0]) << std::endl;
//	std::cout << "host_dst_ptr[0] = " << (unsigned)(host_dst_ptr[0]) << std::endl;
}
// ------------------------------------------------------------

// UVA copy test-case
void uva_copy_host_device_host(unsigned char *const host_dst_ptr, unsigned char *const host_src_ptr, unsigned char *const dev_ptr, const size_t c_array_size) 
{
	// ------------------------------------ UVA -----------------------------------
	clock_t end, start;
	size_t iterations = 0;

	// Get UVA-pointers
	unsigned int * uva_host_src_ptr = NULL;
	unsigned int * uva_host_dst_ptr = NULL;
	cudaHostGetDevicePointer(&uva_host_src_ptr, host_src_ptr, 0);
	cudaHostGetDevicePointer(&uva_host_dst_ptr, host_dst_ptr, 0);

	// Gen data
	srand (time(NULL));
	for(size_t i = 0; i < c_array_size; ++i) {
		const_cast<volatile unsigned char *>(host_src_ptr)[i] = rand() % 256;
		const_cast<volatile unsigned char *>(host_dst_ptr)[i] = 0;
	}
	cudaMemcpy(dev_ptr, host_src_ptr, c_array_size, cudaMemcpyDefault);	// fill zero GPU memory

	// copy data with GPU-Cores in UVA
	size_t THREADS_NUMBER = c_array_size/(sizeof(unsigned int));
	if(THREADS_NUMBER > 1024) THREADS_NUMBER = 1024;

	const size_t BLOCKS_NUMBER = c_array_size/(THREADS_NUMBER*sizeof(unsigned int));
	
	if(BLOCKS_NUMBER >= 65536) {
		std::cout << "BLOCKS_NUMBER can't be large than 65536 \n";
		return;
	}

	cudaDeviceSynchronize();
	start = clock();

	// copy data with UVA
	do {
		// Host to Device
		kernel_function_copy<<<BLOCKS_NUMBER, THREADS_NUMBER, 0, 0>>>((unsigned int *)dev_ptr, uva_host_src_ptr);
		cudaDeviceSynchronize();

		// Device to Host
		kernel_function_copy<<<BLOCKS_NUMBER, THREADS_NUMBER, 0, 0>>>(uva_host_dst_ptr, (unsigned int *)dev_ptr);
		cudaDeviceSynchronize();
		++iterations;
	} while(clock() - start < CLOCKS_PER_SEC);
	end = clock();

//	std::cout << "----------------------------------------------------- \n";
	const float c_time = (float)(end - start)/(CLOCKS_PER_SEC*iterations);
	std::cout << "UVA: block_size = " << c_array_size << 
		", \t time H->D->H: " << c_time << 
		", " << 2*c_array_size/(c_time * 1024*1024) << " MB/sec" << std::endl;

	// Compare data
	compare_arrays(host_src_ptr, host_dst_ptr, c_array_size);

	//std::cout << "host_src_ptr[0] = " << (unsigned)(host_src_ptr[0]) << std::endl;
	//std::cout << "host_dst_ptr[0] = " << (unsigned)(host_dst_ptr[0]) << std::endl;
}
// ------------------------------------------------------------


// UVA copy test-case 1B
void uva_copy_host_device_host_xb(unsigned char *const host_dst_ptr, unsigned char *const host_src_ptr, unsigned char *const dev_ptr, const size_t c_array_size,
								  static const size_t BLOCKS_NUMBER) 
{
	// ------------------------------------ UVA -----------------------------------
	clock_t end, start;
	size_t iterations = 0;

	// Get UVA-pointers
	unsigned int * uva_host_src_ptr = NULL;
	unsigned int * uva_host_dst_ptr = NULL;
	cudaHostGetDevicePointer(&uva_host_src_ptr, host_src_ptr, 0);
	cudaHostGetDevicePointer(&uva_host_dst_ptr, host_dst_ptr, 0);

	// Gen data
	srand (time(NULL));
	for(size_t i = 0; i < c_array_size; ++i) {
		const_cast<volatile unsigned char *>(host_src_ptr)[i] = rand() % 256;
		const_cast<volatile unsigned char *>(host_dst_ptr)[i] = 0;
	}
	cudaMemcpy(dev_ptr, host_src_ptr, c_array_size, cudaMemcpyDefault);	// fill zero GPU memory

	static const size_t unroll_count = 32;

	// copy data with GPU-Cores in UVA
	size_t THREADS_NUMBER = c_array_size/(sizeof(unsigned int) * unroll_count);
	if(THREADS_NUMBER > 1024) THREADS_NUMBER = 1024;

	//std::cout << "THREADS_NUMBER = " << THREADS_NUMBER << std::endl;
	//std::cout << "unroll_count = " << unroll_count << std::endl;
	//std::cout << "c_array_size = " << c_array_size << std::endl;

	cudaDeviceSynchronize();
	start = clock();

	// copy data with UVA
	do {
		// Host to Device
		kernel_function_copy_xb<unroll_count><<<BLOCKS_NUMBER, THREADS_NUMBER, 0, 0>>>((unsigned int *)dev_ptr, uva_host_src_ptr, c_array_size/4);
		cudaDeviceSynchronize();

		// Device to Host
		kernel_function_copy_xb<unroll_count><<<BLOCKS_NUMBER, THREADS_NUMBER, 0, 0>>>(uva_host_dst_ptr, (unsigned int *)dev_ptr, c_array_size/4);
		cudaDeviceSynchronize();
		++iterations;
	} while(clock() - start < CLOCKS_PER_SEC);
	end = clock();

//	std::cout << "----------------------------------------------------- \n";
	const float c_time = (float)(end - start)/(CLOCKS_PER_SEC*iterations);
	std::cout << "UVA " << BLOCKS_NUMBER << "B: block_size = " << c_array_size << 
		", time H->D->H: " << c_time << 
		", " << 2*c_array_size/(c_time * 1024*1024) << " MB/sec" << std::endl;

	// Compare data
	compare_arrays(host_src_ptr, host_dst_ptr, c_array_size);

	//std::cout << "host_src_ptr[0] = " << (unsigned)(host_src_ptr[0]) << std::endl;
	//std::cout << "host_dst_ptr[0] = " << (unsigned)(host_dst_ptr[0]) << std::endl;
}
// ------------------------------------------------------------


int main() {
	const size_t c_array_size = 128*1024*1024;
	srand (time(NULL));


	// count devices & info
	int device_count;
	cudaDeviceProp device_prop;

	// get count Cuda Devices
	cudaGetDeviceCount(&device_count);
	std::cout << "Device count: " <<  device_count << std::endl;

	if (device_count > 100) device_count = 0;
	for (int i = 0; i < device_count; i++)
	{
		// get Cuda Devices Info
		cudaGetDeviceProperties(&device_prop, i);
		std::cout << "Device" << i << ": " <<  device_prop.name;
		std::cout << " (" <<  device_prop.totalGlobalMem/(1024*1024) << " MB)";
		std::cout << ", CUDA capability: " <<  device_prop.major << "." << device_prop.minor << std::endl;	
		std::cout << "UVA: " <<  device_prop.unifiedAddressing << std::endl;
		std::cout << "MAX BLOCKS NUMBER: " <<  device_prop.maxGridSize[0] << std::endl;
	}
	std::cout << std::endl;


	// init pointers
	unsigned char * host_src_ptr = NULL;
	unsigned char * host_dst_ptr = NULL;
	unsigned char * dev_ptr = NULL;
	
	// Can Host map memory
	cudaSetDeviceFlags(cudaDeviceMapHost);

	// Allocate memory
	cudaHostAlloc(&host_src_ptr, c_array_size, cudaHostAllocMapped);
	cudaHostAlloc(&host_dst_ptr, c_array_size, cudaHostAllocMapped);
	cudaMalloc(&dev_ptr, c_array_size);


	for(size_t i = 1; i <= 1024*1024; i*=2) {
		// DMA copy test-case
		dma_copy_host_device_host(host_dst_ptr, host_src_ptr, dev_ptr, c_array_size/i);

		// UVA copy test-case
		uva_copy_host_device_host(host_dst_ptr, host_src_ptr, dev_ptr, c_array_size/i);

		// UVA copy test-case 2B
		uva_copy_host_device_host_xb(host_dst_ptr, host_src_ptr, dev_ptr, c_array_size/i, 2);

		// UVA copy test-case 1B
		uva_copy_host_device_host_xb(host_dst_ptr, host_src_ptr, dev_ptr, c_array_size/i, 1);
		std::cout << "----------------------------------------------------- \n";
	}


	// ------------------------------------ END -----------------------------------
	// Free memory
	cudaFreeHost(host_src_ptr);		// cudaHostAlloc()
	cudaFreeHost(host_dst_ptr);		// cudaHostAlloc()
	cudaFree(dev_ptr);	// cudaMalloc()

	int b; 
	std::cin >> b;

	return 0;
}