cmake_minimum_required(VERSION 3.10)
project(compvis LANGUAGES CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}$<0:>)

set(INTERPROCEDURAL_OPTIMIZATION ON)

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG -use_fast_math)

file(GLOB_RECURSE src_files "${CMAKE_SOURCE_DIR}/src/*.cu")

add_executable(compvis ${src_files})

set_target_properties(compvis PROPERTIES CUDA_ARCHITECTURES native)