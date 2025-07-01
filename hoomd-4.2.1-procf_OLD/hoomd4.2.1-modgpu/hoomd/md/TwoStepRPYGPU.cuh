//~ ########## Created by the Rheoinformatic research group ##########
//~ HOOMD-blue:
// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.
//~
//~ This file:
//~ Written by Mingyang Tan

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#include "hoomd/GPUPartition.cuh"

#include "hip/hip_runtime.h"
#include <cublas_v2.h>
#if defined(ENABLE_HIP)
#ifdef __HIP_PLATFORM_HCC__
#include <hipfft.h>
#else
#include <cufft.h>
typedef cufftComplex hipfftComplex;
#endif
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

void gpu_convert_scalar4_to_scalar(const unsigned int group_size,
                                   const unsigned int * d_index_array,
                                   const Scalar4 * d_vec4,
                                   Scalar * d_vec,
                                   unsigned int block_size);

void gpu_convert_scalar_to_scalar4(const unsigned int group_size,
                                   const unsigned int * d_index_array,
                                   const Scalar * d_vec,
                                   Scalar4 * d_vec4,
                                   unsigned int block_size);

void gpu_check_neighborlist(const unsigned int group_size,
                            const unsigned int * d_index_array,
                            const unsigned int * d_nneigh,
                            const unsigned int * d_nlist,
                            const size_t * d_headlist,
                            unsigned int block_size);

void gpu_compute_wave_value(const uint3 mesh_dim,
                            Scalar4 * d_gridk,
                            Scalar3 * d_ymob,
                            Scalar2 * d_sf,
                            const BoxDim& box,
                            const bool local_fft,
                            const Scalar xi2,
                            const Scalar eta,
                            unsigned int block_size);

void gpu_mobility_real_uf(const unsigned int group_size,
                          const Scalar4 * d_postype,
                          const unsigned int * d_index_array,
                          const unsigned int * d_nneigh,
                          const unsigned int * d_nlist,
                          const size_t * d_headlist,
                          const Scalar2 self_a,
                          const Scalar xi,
                          const Scalar pisqrt,
                          const Scalar xi_pisqrt_inv,
                          const Scalar rcut,
                          const BoxDim& box,
                          Scalar * d_force,
                          Scalar * d_vel,
                          unsigned int block_size);

void gpu_assign_particle_force(const uint3 mesh_dim,
                               const uint3 n_ghost_bins,
                               const uint3 grid_dim,
                               const unsigned int group_size,
                               const unsigned int * d_index_array,
                               const Scalar4 * d_postype,
                               const Scalar * d_force,
                               const int P,
                               const Scalar gauss_fac,
                               const Scalar gauss_exp,
                               const BoxDim& box,
                               const Scalar3 h,
                               hipfftComplex * d_mesh_Fx,
                               hipfftComplex * d_mesh_Fy,
                               hipfftComplex * d_mesh_Fz,
                               unsigned int block_size);

void gpu_mesh_green(const unsigned int n_wave_vectors,
                    hipfftComplex * d_mesh_Fx,
                    hipfftComplex * d_mesh_Fy,
                    hipfftComplex * d_mesh_Fz,
                    const Scalar4 * d_gridk,
                    const Scalar3 * d_ymob,
                    const Scalar2 * d_sf,
                    hipfftComplex * d_fourier_mesh_Fx,
                    hipfftComplex * d_fourier_mesh_Fy,
                    hipfftComplex * d_fourier_mesh_Fz,
                    unsigned int block_size);

void gpu_interpolate_particle_velocity(const uint3 mesh_dim,
                                       const uint3 n_ghost_bins,
                                       const uint3 grid_dim,
                                       const unsigned int group_size,
                                       const unsigned int * d_index_array,
                                       const Scalar4 * d_postype,
                                       const int P,
                                       const Scalar gauss_fac,
                                       const Scalar gauss_exp,
                                       const Scalar vk,
                                       const BoxDim& box,
                                       const Scalar3 h,
                                       bool local_fft,
                                       unsigned int inv_mesh_elements,
                                       const hipfftComplex * d_mesh_inv_Fx,
                                       const hipfftComplex * d_mesh_inv_Fy,
                                       const hipfftComplex * d_mesh_inv_Fz,
                                       Scalar * d_vel,
                                       unsigned int block_size);

void gpu_mobility_velocity_sum(const unsigned int group_size,
                               const unsigned int * d_index_array,
                               const Scalar * d_u1,
                               const Scalar * d_u2,
                               Scalar * d_u,
                               unsigned int block_size);

void gpu_brownian_farfield_particle_rng(const Scalar kT,
                                        const uint64_t timestep,
                                        const uint16_t seed,
                                        const Scalar dt,
                                        const unsigned int group_size,
                                        const unsigned int * d_index_array,
                                        const unsigned int * d_tag,
                                        Scalar * d_psi,
                                        unsigned int block_size);

void gpu_brownian_lanczos(const unsigned int group_size,
                          const Scalar4 * d_postype,
                          const unsigned int * d_index_array,
                          const unsigned int * d_nneigh,
                          const unsigned int * d_nlist,
                          const size_t * d_headlist,
                          const Scalar2 self_a,
                          const Scalar xi,
                          const Scalar pisqrt,
                          const Scalar xi_pisqrt_inv,
                          const Scalar rcut,
                          const BoxDim& box,
                          const unsigned int m_lanczos_ff,
                          const unsigned int mmax,
                          const Scalar error,
                          Scalar * d_psi,
                          Scalar * d_iter_ff_v,
                          Scalar * d_iter_ff_vj,
                          Scalar * d_iter_ff_vjm1,
                          Scalar * d_iter_ff_uold,
                          Scalar * d_iter_ff_V,
                          Scalar * d_iter_Mpsi,
                          Scalar * d_Tm,
                          Scalar * d_uslip_real,
                          cublasHandle_t blasHandle,
                          unsigned int block_size);

void gpu_brownian_farfield_grid_rng(const unsigned int n_wave_vectors,
                                    const uint64_t timestep,
                                    const uint16_t seed,
                                    const uint3 mesh_dim,
                                    const Scalar T,
                                    const Scalar vk,
                                    const Scalar dt,
                                    const Scalar4 * d_gridk,
                                    const Scalar3 * d_ymob,
                                    const Scalar2 * d_sf,
                                    hipfftComplex * d_mesh_inv_Fx,
                                    hipfftComplex * d_mesh_inv_Fy,
                                    hipfftComplex * d_mesh_inv_Fz,
                                    unsigned int block_size);

void gpu_rpy_step_one(const unsigned int group_size,
                      const unsigned int * d_index_array,
                      const BoxDim& box,
                      int3 * d_image,
                      const Scalar dt,
                      Scalar4 * d_vel,
                      Scalar4 * d_pos,
                      unsigned int block_size,
                      const GPUPartition& gpu_partition);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
