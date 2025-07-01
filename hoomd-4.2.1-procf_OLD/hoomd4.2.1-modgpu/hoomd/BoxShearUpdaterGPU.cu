//~ ########## Created by the Rheoinformatic research group ##########
//~ HOOMD-blue:
// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.
//~
//~ This file:
//~ Written by Mingyang Tan


// ########## Created by Rheoinformatic //~ [RHEOINF] ##########
#include "BoxShearUpdaterGPU.cuh"
#include <cuda_runtime.h>

namespace hoomd
    {
namespace kernel
    {

__global__ void gpu_box_shear_wrap_kernel(const unsigned int N,
                                          BoxDim local_box,
                                          Scalar Ly,
                                          Scalar4* d_pos,
                                          int3* d_image,
                                          Scalar4* d_vel,
                                          Scalar cur_erate)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        int img0 = d_image[idx].y;
        local_box.wrap(d_pos[idx], d_image[idx]);
        img0 -= d_image[idx].y;
        d_vel[idx].x += (img0 * cur_erate * Ly);
        }
    } /* end of gpu_box_shear_wrap_kernel */

hipError_t gpu_box_shear_wrap(const unsigned int N,
                              const BoxDim& local_box,
                              Scalar Ly,
                              Scalar4* d_pos,
                              int3* d_image,
                              Scalar4* d_vel,
                              Scalar cur_erate,
                              unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_box_shear_wrap_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid((N / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    hipLaunchKernelGGL((gpu_box_shear_wrap_kernel),
        dim3(grid),
        dim3(threads),
        0,
        0,
        N,
        local_box,
        Ly,
        d_pos,
        d_image,
        d_vel,
        cur_erate);

    return hipSuccess;
    } /* end of gpu_box_shear_wrap*/


    } // end namespace kernel
    } // end namespace hoomd