I used CUDA toolkit v12.6 and Nvidia video codec sdk v12.2.
Video sdk is in the third_party folder, it isn't included in the cuda toolkit v12.6.

# Windows:

Instal Cmake and VS Community 2022. Add MSBuild to path.

```
mkdir build
cd build
cmake .. 
(cmake .. -DCMAKE_BUILD_TYPE=DEBUG)
msbuild compvis.sln /p:Configuration=Release /p:Platform=x64 
(msbuild compvis.sln /p:Configuration=Debug /p:Platform=x64)
```

[ŞÜKRÜ ÇİRİŞ 2024](https://sukruciris.github.io)