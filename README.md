Windows:
(instal cmake and vs community 2022. add msbuild to path.)
```
mkdir build
cd build
cmake ..
msbuild compvis.sln /p:Configuration=Release /p:Platform=x64
```