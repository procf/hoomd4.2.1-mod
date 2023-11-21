if(ENABLE_HIP)

    if (HOOMD_GPU_PLATFORM STREQUAL "HIP")
        find_package(HIP REQUIRED)
        CMAKE_MINIMUM_REQUIRED(VERSION 3.21 FATAL_ERROR)
        ENABLE_LANGUAGE(HIP)
        SET(HOOMD_DEVICE_LANGUAGE HIP)

        # setup nvcc to build for all CUDA architectures. Allow user to modify the list if desired
        set(CMAKE_HIP_ARCHITECTURES gfx900 gfx906 gfx908 gfx90a CACHE STRING "List of AMD GPU to compile HIP code for. Separate with semicolons.")
        set(HIP_PLATFORM hip-clang)
    elseif (HOOMD_GPU_PLATFORM STREQUAL "CUDA")
        # here we go if hipcc is not available, fall back on internal HIP->CUDA headers
        ENABLE_LANGUAGE(CUDA)
        SET(HOOMD_DEVICE_LANGUAGE CUDA)

        set(HIP_INCLUDE_DIR "$<IF:$<STREQUAL:${CMAKE_PROJECT_NAME},HOOMD>,${CMAKE_CURRENT_SOURCE_DIR},${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}/include>/hoomd/extern/HIP/include/")

        # use CUDA runtime version
        string(REGEX MATCH "([0-9]*).([0-9]*).([0-9]*).*" _hip_version_match "${CMAKE_CUDA_COMPILER_VERSION}")
        set(HIP_VERSION_MAJOR "${CMAKE_MATCH_1}")
        set(HIP_VERSION_MINOR "${CMAKE_MATCH_2}")
        set(HIP_VERSION_PATCH "${CMAKE_MATCH_3}")
        set(HIP_PLATFORM "nvcc")

        # hipCUB
        # Use system provided CUB for CUDA 11 and newer
        set(HIPCUB_INCLUDE_DIR "$<IF:$<STREQUAL:${CMAKE_PROJECT_NAME},HOOMD>,${CMAKE_CURRENT_SOURCE_DIR},${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}/include>/hoomd/extern/hipCUB/hipcub/include/")
    else()
        message(FATAL_ERROR "HOOMD_GPU_PLATFORM must be either CUDA or HIP")
    endif()

    if(NOT TARGET hip::host)
        add_library(hip::host INTERFACE IMPORTED)
        set_target_properties(hip::host PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR};${HIPCUB_INCLUDE_DIR}")

        # set HIP_VERSION_* on non-CUDA targets (the version is already defined on AMD targets through hipcc)
        set_property(TARGET hip::host APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:HIP_VERSION_MAJOR=${HIP_VERSION_MAJOR}>)
        set_property(TARGET hip::host APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:HIP_VERSION_MINOR=${HIP_VERSION_MINOR}>)
        set_property(TARGET hip::host APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:HIP_VERSION_PATCH=${HIP_VERSION_PATCH}>)

    endif()

    # branch upon HCC or NVCC target
    if(${HIP_PLATFORM} STREQUAL "nvcc")
        set_property(TARGET hip::host APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS __HIP_PLATFORM_NVCC__)
    elseif(${HIP_PLATFORM} STREQUAL "hip-clang")
        set_property(TARGET hip::host APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS __HIP_PLATFORM_AMD__)
    endif()

    find_package(CUDALibs REQUIRED)
endif()
