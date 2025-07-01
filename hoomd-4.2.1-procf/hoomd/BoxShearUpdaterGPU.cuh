//~ ########## Created by the Rheoinformatic research group ##########
//~ HOOMD-blue:
// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.
//~
//~ This file:
//~ Written by Mingyang Tan


// ########## Created by Rheoinformatic //~ [RHEOINF] ##########
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/BoxDim.h"

namespace hoomd
    {
namespace kernel
    {

hipError_t gpu_box_shear_wrap(const unsigned int N,
                              const BoxDim& local_box,
                              Scalar Ly,
                              Scalar4* d_pos,
                              int3* d_image,
                              Scalar4* d_vel,
                              Scalar cur_erate,
                              unsigned int block_size);

    } // end namespace kernel
    } // end namespace hoomd