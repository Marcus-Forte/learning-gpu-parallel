cmake_minimum_required(VERSION 3.15)
project(learning-gpu-parallel)
enable_language(CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set_target_properties(mylib PROPERTIES
#                             CUDA_SEPARABLE_COMPILATION ON)

add_executable(main 
src/main.cu
src/kernel.cu)

target_include_directories(
    main 
    PRIVATE 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

