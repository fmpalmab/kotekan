add_library(dsabeamformer STATIC src/beamformer.cu)
set_target_properties(dsabeamformer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(dsabeamformer PUBLIC src)
target_link_libraries(dsabeamformer PUBLIC CUDA::cudart CUDA::cublas)
