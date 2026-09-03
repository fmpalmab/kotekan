project(test_charts)

if(CUDAToolkit_FOUND)
    # Benchmark Kotekan CUDA Tracker V5
    add_executable(benchmark_kotekan_cuda_tracker_v5 benchmark_kotekan_cuda_tracker_v5.cpp)
    target_link_libraries(
        benchmark_kotekan_cuda_tracker_v5
        PRIVATE -Wl,--whole-archive $<TARGET_PROPERTY:kotekan_libs,INTERFACE_LINK_LIBRARIES>
                -Wl,--no-whole-archive)
    target_include_directories(benchmark_kotekan_cuda_tracker_v5 PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/metadata
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/core
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/json
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmt)

    # Benchmark Pipeline Stages
    add_executable(benchmark_pipeline_stages benchmark_pipeline_stages.cpp)
    target_link_libraries(
        benchmark_pipeline_stages
        PRIVATE -Wl,--whole-archive $<TARGET_PROPERTY:kotekan_libs,INTERFACE_LINK_LIBRARIES>
                -Wl,--no-whole-archive)
    target_include_directories(benchmark_pipeline_stages PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/metadata
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/core
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/json
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmt)

    # Extensive Kotekan Beam Tracker Comprehensive Test Suite
    add_executable(test_beam_tracker_suite test_beam_tracker_suite.cpp)
    target_link_libraries(
        test_beam_tracker_suite
        PRIVATE -Wl,--whole-archive $<TARGET_PROPERTY:kotekan_libs,INTERFACE_LINK_LIBRARIES>
                -Wl,--no-whole-archive)
    target_include_directories(test_beam_tracker_suite PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/metadata
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/core
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/dpdk
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/stages
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/json
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmt)

    # Chaos Engineering & Fault Injection Test Suite
    add_executable(test_chaos_engineering test_chaos_engineering.cpp)
    target_link_libraries(
        test_chaos_engineering
        PRIVATE -Wl,--whole-archive $<TARGET_PROPERTY:kotekan_libs,INTERFACE_LINK_LIBRARIES>
                -Wl,--no-whole-archive)
    target_include_directories(test_chaos_engineering PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/metadata
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/core
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/dpdk
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/stages
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/json
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmt)

    # Benchmark Direct Beam Tracker (Zero Integration Window)
    add_executable(benchmark_direct_beam_tracker benchmark_direct_beam_tracker.cpp)
    target_link_libraries(
        benchmark_direct_beam_tracker
        PRIVATE -Wl,--whole-archive $<TARGET_PROPERTY:kotekan_libs,INTERFACE_LINK_LIBRARIES>
                -Wl,--no-whole-archive)
    target_include_directories(benchmark_direct_beam_tracker PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/metadata
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/core
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/json
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmt)

    # Head-to-Head Comparative Benchmark: Beam Tracker V5 vs. Direct Beam Tracker
    add_executable(benchmark_tracker_v5_vs_direct benchmark_tracker_v5_vs_direct.cpp)
    target_link_libraries(
        benchmark_tracker_v5_vs_direct
        PRIVATE -Wl,--whole-archive $<TARGET_PROPERTY:kotekan_libs,INTERFACE_LINK_LIBRARIES>
                -Wl,--no-whole-archive)
    target_include_directories(benchmark_tracker_v5_vs_direct PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/metadata
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/core
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/json
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmt)
endif()
