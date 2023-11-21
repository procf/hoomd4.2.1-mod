// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "IntegratorHPMCMonoGPUMoves.cuh"

//! This file, with a .cu ending, is auto-generated from the .cu.in template. Do not edit directly.

// clang-format off

//! A few defines to instantiate a kernel template
#cmakedefine SHAPE @SHAPE@                  // the class name of the shape
#cmakedefine SHAPE_INCLUDE @SHAPE_INCLUDE@  // the name of the include file
#cmakedefine IS_UNION_SHAPE                 // define to generate a kernel for a ShapeUnion<...>

// clang-format on

#define XSTR(x) #x
#define STR(x) XSTR(x)
#include STR(SHAPE_INCLUDE)

#ifdef IS_UNION_SHAPE
#include "ShapeUnion.h"
#define SHAPE_CLASS(T) ShapeUnion<T>
#else
#define SHAPE_CLASS(T) T
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace gpu
    {
//! Driver for kernel::hpmc_update_pdata()
template void hpmc_update_pdata<SHAPE_CLASS(SHAPE)>(const hpmc_update_args_t& args,
                                                    const SHAPE_CLASS(SHAPE)::param_type* params);
    } // namespace gpu

    } // end namespace hpmc
    } // end namespace hoomd