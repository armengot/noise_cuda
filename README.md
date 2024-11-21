# SSIM & PSNR
Get more info about noise measures:
- [PSNR](https://en.wikipedia.org/wiki/PSNR)
- [SSIM](https://en.wikipedia.org/wiki/SSIM)

# BUILT with CUDA <img src="https://img.shields.io/badge/Ubuntu 20.4-dev_env-yellow"> <img src="https://img.shields.io/badge/Mint_21.1-passed-brown">
Here you'll find CUDA examples to get a binary C++ example to compute SSIM and PSNR on GPU cores with CUDA.

<img src="https://img.shields.io/badge/cuda_12.1-dev_env-yellow">
<img src="https://img.shields.io/badge/OpenCV_4.7.0-dev_env-yellow">

<img src="https://img.shields.io/badge/cuda_11.7-tested-brown">
<img src="https://img.shields.io/badge/OpenCV_4.10.0-tested-brown">


# Dependencies
Install OpenCV from [GitGub](https://github.com/opencv/opencv.git) with the same versión of [```opencv_contrib```](https://github.com/opencv/opencv_contrib/):

```
$ git clone https://github.com/opencv/opencv.git
$ cd opencv
$ git checkout 4.10.0
$ cd ..
$ git clone https://github.com/opencv/opencv_contrib/
$ cd opencv_contrib
$ git checkout 4.10.0
```
and compile together (check your cuda version):
```
$ cd ../opencv
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.7 \
      -D WITH_CUDNN=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_TBB=ON ..
```
The version of CUDA must match the version required for compiling OpenCV.

# Compilation
Review the ```CMakeLists.txt``` file, ckeck the local installation of OpenCV in your machine and replace my path ```/usr/local/include/opencv4``` with yours. Usually, OpenCV must be compiled with CUDA support.

Use ```cmake``` as follows:
```
$ mkdir build
$ cd build
$ cmake ..
```
The ```cmake``` command must show as result:
```
-- The C compiler identification is GNU 10.5.0
-- The CXX compiler identification is GNU 10.5.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE  
-- Found CUDA: /usr/local/cuda-11.7 (found version "11.7") 
-- Found OpenCV: /usr/local (found version "4.10.0") 
-- Configuring done
-- Generating done
-- Build files have been written to: /home/marcelo/dev/personal/github/noise_cuda/build

```
And after ```make``` command:

```
[ 25%] Building NVCC (Device) object CMakeFiles/noisemeter.dir/noisemeter_generated_noisemeter.cu.o
[ 50%] Linking CXX static library libnoisemeter.a
[ 50%] Built target noisemeter
[ 75%] Building CXX object CMakeFiles/test_noise.dir/test_noise.cpp.o
[100%] Linking CXX executable test_noise
[100%] Built target test_noise

```

# Usage:
Two path images from command line:
```
$ [...]/build$ cp ../img/* .
$ [...]/build$ ./test_noise lena_hires.jpg lena_hires_15noisy.jpg
PSNR: 19.3948
SSIM: 0.948638
```
# Create your noisy images
Use python script ```noise.py``` to generate a test noisy image.
Use standard Lenna images in [```img```](img/lena_hires.jpg) folder to compare index.
Use macros in ```noisemeter.cu``` to convert RGB to GRAY image using a function or selection one RGB channel.


