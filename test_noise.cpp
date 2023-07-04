#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

double compute_psnr(const cv::cuda::GpuMat& rgb1, const cv::cuda::GpuMat& rgb2);
double compute_ssim(const cv::cuda::GpuMat& rgb1, const cv::cuda::GpuMat& rgb2);

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <image1_path> <image2_path>" << std::endl;
        return 1;
    }
    
    cv::Mat image1 = cv::imread(argv[1]);
    cv::Mat image2 = cv::imread(argv[2]);

    if (image1.empty() || image2.empty())
    {
        std::cout << "Failed to load images." << std::endl;
        return 1;
    }

    if (image1.channels() != 3)
        cv::cvtColor(image1, image1, cv::COLOR_GRAY2BGR);
    if (image2.channels() != 3)
        cv::cvtColor(image2, image2, cv::COLOR_GRAY2BGR);

    cv::cuda::GpuMat gpuImage1(image1);
    cv::cuda::GpuMat gpuImage2(image2);

    double psnr = compute_psnr(gpuImage1, gpuImage2);
    double ssim = compute_ssim(gpuImage1, gpuImage2);

    std::cout << "PSNR: " << psnr << std::endl;
    std::cout << "SSIM: " << ssim << std::endl;

    return 0;
}
