#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

#define RGBTOGRAYFLAG 0
#define NOISE_CHANNEL 1

template <typename T>
__device__ T atomicAdd(T* address, T val);

/* Atomic addition template specialization for double precision */
/* ************************************************************ */
template <>
__device__ double atomicAdd<double>(double* address, double val)
{
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

/* ********************************* */
/* Kernel function to calculate PSNR */
/* ********************************* */
__global__
void calculate_psnr(const unsigned char* gray1, const unsigned char* gray2, const int size, double* psnr)
{
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    double squared_diff_sum = 0.0;

    for (int i = thread_idx; i < size; i += stride)
    {
        int diff = gray1[i] - gray2[i];
        squared_diff_sum = squared_diff_sum + diff * diff;        
    }
    
    atomicAdd(psnr, squared_diff_sum);
}

/* ********************************* */
/* Kernel function to calculate SSIM */
/* ********************************* */
__global__
void calculate_ssim(const unsigned char* gray1, const unsigned char* gray2, const int size, double* ssim)
{
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    double c1 = 0.0001; // Small constant to avoid division by zero
    double c2 = 0.0009; 

    double local_mean1 = 0.0;
    double local_mean2 = 0.0;
    double local_var1 = 0.0;
    double local_var2 = 0.0;
    double local_cov = 0.0;

    // Accumulate values within the loop
    for (int i = thread_idx; i < size; i += stride)
    {
        double x = gray1[i];
        double y = gray2[i];

        local_mean1 = local_mean1 + x;
        local_mean2 = local_mean2 + y;
        local_var1 = local_var1 + x * x;
        local_var2 = local_var2 + y * y;
        local_cov = local_cov + x * y;
    }

    // Reduce the values across threads in a block
    __shared__ double shared_mean1;
    __shared__ double shared_mean2;
    __shared__ double shared_var1;
    __shared__ double shared_var2;
    __shared__ double shared_cov;

    // Accumulate the values from all threads
    atomicAdd(&shared_mean1, local_mean1);
    atomicAdd(&shared_mean2, local_mean2);
    atomicAdd(&shared_var1, local_var1);
    atomicAdd(&shared_var2, local_var2);
    atomicAdd(&shared_cov, local_cov);

    __syncthreads();

    // Only one thread performs the final reduction and SSIM computation
    if (thread_idx == 0)
    {
        // Compute the block-wise mean, variance, and covariance
        double mean1 = shared_mean1 / size;
        double mean2 = shared_mean2 / size;
        double var1 = shared_var1 / size - mean1 * mean1;
        double var2 = shared_var2 / size - mean2 * mean2;
        double cov = shared_cov / size - mean1 * mean2;

        double numerator = (2.0 * mean1 * mean2 + c1) * (2.0 * cov + c2);
        double denominator = (mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2);
        double tssim = numerator / denominator;

        atomicAdd(ssim, tssim);        
    }
}

/* ******************************************************************************************
 * compute_psnr(rgb1, rgb2) calls the kernel function calculate_psnr()
 * to evaluate the Peak Signal-to-Noise Ratio (PSNR) on GPU cores in an asynchronous way.
 */
double compute_psnr(const cv::cuda::GpuMat& rgb1, const cv::cuda::GpuMat& rgb2)
{
    cv::cuda::GpuMat gray1, gray2;

    if (RGBTOGRAYFLAG)
    {
        cv::cuda::cvtColor(rgb1, gray1, cv::COLOR_BGR2GRAY);
        cv::cuda::cvtColor(rgb2, gray2, cv::COLOR_BGR2GRAY);
    }
    else
    {
        const unsigned char* d_rgb1 = rgb1.ptr<unsigned char>();
        const unsigned char* d_rgb2 = rgb2.ptr<unsigned char>();
        gray1 = cv::cuda::GpuMat(rgb1.rows, rgb1.cols, CV_8UC1, (void*)(d_rgb1 + NOISE_CHANNEL), rgb1.step);
        gray2 = cv::cuda::GpuMat(rgb2.rows, rgb2.cols, CV_8UC1, (void*)(d_rgb2 + NOISE_CHANNEL), rgb2.step);
    }

    cv::Mat cpuGray1, cpuGray2;
    gray1.download(cpuGray1);
    gray2.download(cpuGray2);

    cv::imwrite("gray1.jpg", cpuGray1);
    cv::imwrite("gray2.jpg", cpuGray2);

    int size = gray1.rows * gray1.cols;

    const unsigned char* d_gray1 = gray1.ptr<unsigned char>();
    const unsigned char* d_gray2 = gray2.ptr<unsigned char>();

    double* d_psnr;
    cudaMalloc((void**)&d_psnr, sizeof(double));
    cudaMemset(d_psnr, 0.0, sizeof(double));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    calculate_psnr<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_gray1, d_gray2, size, d_psnr);

    double h_psnr;
    cudaMemcpyAsync(&h_psnr, d_psnr, sizeof(double), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaFree(d_psnr);

    h_psnr = h_psnr / size;
    h_psnr = 255.0 * 255.0 / h_psnr;
    h_psnr = 10.0 * std::log10(h_psnr);

    cudaStreamDestroy(stream);

    return h_psnr;
}

/* 
 * compute_ssim(rgb1, rgb2) calls the kernel function calculate_ssim()
 * to evaluate the Structural Similarity Index (SSIM) on GPU cores in an asynchronous way.
 */
double compute_ssim(const cv::cuda::GpuMat& rgb1, const cv::cuda::GpuMat& rgb2)
{
    cv::cuda::GpuMat gray1, gray2;

    if (RGBTOGRAYFLAG)
    {
        cv::cuda::cvtColor(rgb1, gray1, cv::COLOR_BGR2GRAY);
        cv::cuda::cvtColor(rgb2, gray2, cv::COLOR_BGR2GRAY);
    }
    else
    {
        const unsigned char* d_rgb1 = rgb1.ptr<unsigned char>();
        const unsigned char* d_rgb2 = rgb2.ptr<unsigned char>();
        gray1 = cv::cuda::GpuMat(rgb1.rows, rgb1.cols, CV_8UC1, (void*)(d_rgb1 + NOISE_CHANNEL), rgb1.step);
        gray2 = cv::cuda::GpuMat(rgb2.rows, rgb2.cols, CV_8UC1, (void*)(d_rgb2 + NOISE_CHANNEL), rgb2.step);
    }

    int size = gray1.rows * gray1.cols;

    const unsigned char* d_gray1 = gray1.ptr<unsigned char>();
    const unsigned char* d_gray2 = gray2.ptr<unsigned char>();

    double* d_ssim;
    cudaMalloc((void**)&d_ssim, sizeof(double));
    cudaMemset(d_ssim, 0.0, sizeof(double));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    calculate_ssim<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_gray1, d_gray2, size, d_ssim);

    cudaStreamSynchronize(stream);

    double h_ssim = 0.0;    
    cudaMemcpyAsync(&h_ssim, d_ssim, sizeof(double), cudaMemcpyDeviceToHost, stream);

    cudaStreamDestroy(stream);
    cudaFree(d_ssim);

    return h_ssim;
}
