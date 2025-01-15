# Tools

I used CUDA toolkit v12.6 and Nvidia video codec sdk v12.2.

Video sdk is in the third_party folder, it isn't included in the cuda toolkit v12.6. Video decoder isn't implemented in the project yet. The project uses pre-extracted video frames for now.

# Project

I am trying to gain experience about computer vision subjects. In this project, I used the Cuda language to harness GPU parallel execution power, and I didn't use any external libraries. I used some Nvidia Cuda libraries for decoding and encoding media files.

* I made an image processing library in CUDA from scratch. I can manipulate image pixels, make edge detection and calculate some statistical information like mean, variance, covariance, SSIM, etc., on a GPU.

* I developed a template matcher library that detects objects in a frame using image samples of the target object. The library first performs edge detection on both the frame and the samples. Then, it calculates the Structural Similarity Index (SSIM) for each sample against corresponding regions of the frame, iterating over frame segments of the same size as the samples. The SSIM values are averaged across the samples, and the function identifies the coordinates with the highest score as the top-left corner of the detected object.

Template matcher works correct but it isn't as fast as I want for now. I will continue to improve.

# Build on Windows:

Instal Cmake, VS Community 2022 and CUDA toolkit v12.6. Add MSBuild to path.

```
mkdir build
cd build
cmake .. 
(cmake .. -DCMAKE_BUILD_TYPE=DEBUG)
msbuild compvis.sln /p:Configuration=Release /p:Platform=x64 
(msbuild compvis.sln /p:Configuration=Debug /p:Platform=x64)
```

Building on linux shouldn't be so different, you just need to use make command instead of msbuild. I wrote the Cmake file to be crossplatform. Currently, I don't have linux on my computers so I didn't try.

[ŞÜKRÜ ÇİRİŞ 2024](https://sukruciris.github.io)