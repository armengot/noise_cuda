# SSIM & PSNR
Get more info about noise measures:
- [PSNR](https://en.wikipedia.org/wiki/PSNR)
- [SSIM](https://en.wikipedia.org/wiki/SSIM)

## CUDA
Here you'll find CUDA examples to get a binary C++ example to compute SSIM and PSNR on GPU cores with CUDA.

## Compilation
Use ```cmake``` as follows:
```
$ mkdir build
$ cd build
$ cmake ..
$ make 
```

## Usage:
Two path images from command line:
```
$ ./test_noise img1.jpg img2.jpg
```

## Test
Use python script ```noise.py``` to a test image.
Use standard Lenna images in ```img``` folder to compare index.
Use macros in ```noisemeter.cu``` to convert RGB to GRAY image using a function or selection one RGB channel.