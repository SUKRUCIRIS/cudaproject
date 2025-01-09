# Tools

I used CUDA toolkit v12.6 and Nvidia video codec sdk v12.2.

Video sdk is in the third_party folder, it isn't included in the cuda toolkit v12.6. Video decoder isn't implemented in the project yet. The project uses pre-extracted video frames for now.

# Project Purpose

I am trying to gain experience about computer vision subjects. In this project, I used the Cuda language to harness GPU parallel execution power, and I didn't use any external libraries like OpenCV. I used some Nvidia Cuda libraries for decoding and encoding media files.

The purpose was to detect an object in a video and make it as fast as I could. My goal is for it to have real-time speed at least. (33 ms per frame for a 30 fps video.) 33ms was my target. Also, I wanted to learn the math behind the image processing and write the GPU kernel programs to do the calculations as fast as I can.

# Results

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