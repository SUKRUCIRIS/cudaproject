Windows:
(instal cmake and vs community 2022. add msbuild to path.)
```
mkdir build
cd build
cmake .. 
(cmake .. -DCMAKE_BUILD_TYPE=DEBUG)
msbuild compvis.sln /p:Configuration=Release /p:Platform=x64 
(msbuild compvis.sln /p:Configuration=Debug /p:Platform=x64)
```

[ŞÜKRÜ ÇİRİŞ 2024](https://sukruciris.github.io)