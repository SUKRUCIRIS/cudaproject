cmake_minimum_required(VERSION 3.10)
project(compvis LANGUAGES CUDA)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}$<0:>)

if (WIN32)

	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
	find_library(nvcuvid_LIBRARY nvcuvid PATHS ${CMAKE_SOURCE_DIR}/third_party/nvideo/Lib/x64/)
	find_library(nvencodeapi_LIBRARY nvencodeapi PATHS ${CMAKE_SOURCE_DIR}/third_party/nvideo/Lib/x64/)

	if (CUDA_TOOLKIT_ROOT_DIR STREQUAL "" OR NOT IS_DIRECTORY CUDA_TOOLKIT_ROOT_DIR)
		set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
	endif()

endif()

set(INTERPROCEDURAL_OPTIMIZATION ON)

if (CMAKE_BUILD_TYPE STREQUAL "Release")

	set(CMAKE_CUDA_FLAGS "-O3 -DNDEBUG -use_fast_math")

endif()

file(GLOB_RECURSE src_files "${CMAKE_SOURCE_DIR}/src/*.cu")

add_compile_definitions("DEBUG=$<CONFIG:Debug>")

add_executable(compvis ${src_files})

set_target_properties(compvis PROPERTIES CUDA_ARCHITECTURES native)

message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")

find_library(NVJPEG_LIBRARY nvjpeg PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(compvis ${NVJPEG_LIBRARY} ${nvcuvid_LIBRARY} ${nvencodeapi_LIBRARY} CUDA::cudart CUDA::cuda_driver)