// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "BoxResizeUpdaterGPU.cuh"

namespace hoomd
    {
namespace kernel
    {

__global__ void gpu_box_resize_scale_kernel(Scalar4* d_pos,
                                            const BoxDim cur_box,
                                            const BoxDim new_box,
                                            const unsigned int* d_group_members,
                                            const unsigned int group_size)
    {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        Scalar4 pos = d_pos[idx];

        Scalar3 fractional_pos = cur_box.makeFraction(make_scalar3(pos.x, pos.y, pos.z));

        Scalar3 scaled_pos = new_box.makeCoordinates(fractional_pos);
        d_pos[idx].x = scaled_pos.x;
        d_pos[idx].y = scaled_pos.y;
        d_pos[idx].z = scaled_pos.z;
        }
    }

//~ [RHEOINF]
__global__ void
gpu_box_resize_wrap_kernel(unsigned int N,
                           Scalar4* d_pos, 
                           Scalar4* d_vel, //~ [RHEOINF]
                           int3* d_image, 
                           const BoxDim local_box, //~ [RHEOINF]
                           Scalar cur_vel) //~ [RHEOINF]
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        int img0 = d_image[idx].y;   //~ get y-image for velocity scaling [RHEOINF]
        local_box.wrap(d_pos[idx], d_image[idx]);
        img0 -= d_image[idx].y;  //~ use current data to modify image [RHEOINF]
        d_vel[idx].x += (img0 * cur_vel);
        }
    }

hipError_t gpu_box_resize_scale(Scalar4* d_pos,
                                const BoxDim& cur_box,
                                const BoxDim& new_box,
                                const unsigned int* d_group_members,
                                const unsigned int group_size,
                                unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_box_resize_wrap_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid((group_size / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    hipLaunchKernelGGL((gpu_box_resize_scale_kernel),
                       grid,
                       threads,
                       0,
                       0,
                       d_pos,
                       cur_box,
                       new_box,
                       d_group_members,
                       group_size);

    return hipSuccess;
    }

hipError_t gpu_box_resize_wrap(const unsigned int N,
                               Scalar4* d_pos,
                               Scalar4* d_vel, //~ [RHEOINF]
                               int3* d_image,
                            //    const BoxDim& new_box,
                               const BoxDim& local_box, //~ [RHEOINF]
                               Scalar cur_vel, //~ [RHEOINF]
                               unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_box_resize_wrap_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid((N / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    hipLaunchKernelGGL((gpu_box_resize_wrap_kernel),
                       grid,
                       threads,
                       0,
                       0,
                       N,
                       d_pos,
                       d_vel, //~ [RHEOINF]
                       d_image,
                       local_box,   //~ [RHEOINF]
                       cur_vel);    //~ [RHEOINF]

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace hoomd
