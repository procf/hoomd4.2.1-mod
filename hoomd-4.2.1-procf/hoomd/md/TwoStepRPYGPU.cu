//~ ########## Created by the Rheoinformatic research group ##########
//~ HOOMD-blue:
// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.
//~
//~ This file:
//~ Written by Mingyang Tan

#include "TwoStepRPYGPU.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include <lapacke.h>
#include <cblas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// __scalar2int_rd is __float2int_rd in single, __double2int_rd in double
#if HOOMD_LONGREAL_SIZE == 32
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif

#define GPU_PPPM_MAX_ORDER 7

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
// workaround for HIP bug
#ifdef __HIP_PLATFORM_HCC__
inline __device__ float myAtomicAdd(float* address, float val)
    {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;

    do
        {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(val + __uint_as_float(assumed)));
        } while (assumed != old);

    return __uint_as_float(old);
    }
#else
inline __device__ float myAtomicAdd(float* address, float val)
    {
    return atomicAdd(address, val);
    }
#endif

__global__ void gpu_convert_scalar4_to_scalar_kernel(const unsigned int group_size,
                                                     const unsigned int * d_index_array,
                                                     const Scalar4 * d_vec4,
                                                     Scalar * d_vec)
    {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx < group_size)
        {
        unsigned int idx = d_index_array[group_idx];
        unsigned int i3 = idx * 3;

        d_vec[i3    ] = d_vec4[idx].x;
        d_vec[i3 + 1] = d_vec4[idx].y;
        d_vec[i3 + 2] = d_vec4[idx].z;
        }

    } /* end of gpu_convert_scalar4_to_scalar_kernel */

void gpu_convert_scalar4_to_scalar(const unsigned int group_size,
                                   const unsigned int * d_index_array,
                                   const Scalar4 * d_vec4,
                                   Scalar * d_vec,
                                   unsigned int block_size)
    {
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    hipLaunchKernelGGL(
        (gpu_convert_scalar4_to_scalar_kernel),
        dim3(grid),
        dim3(threads),
        0,
        0,
        group_size,
        d_index_array,
        d_vec4,
        d_vec);

    } /* end of gpu_convert_scalar4_to_scalar */

__global__ void gpu_convert_scalar_to_scalar4_kernel(const unsigned int group_size,
                                                     const unsigned int * d_index_array,
                                                     const Scalar * d_vec,
                                                     Scalar4 * d_vec4)
    {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx < group_size)
        {
        unsigned int idx = d_index_array[group_idx];
        unsigned int i3 = idx * 3;

        d_vec4[idx].x = d_vec[i3    ];
        d_vec4[idx].y = d_vec[i3 + 1];
        d_vec4[idx].z = d_vec[i3 + 2];
        }

    } /* end of gpu_convert_scalar4_to_scalar_kernel */

void gpu_convert_scalar_to_scalar4(const unsigned int group_size,
                                   const unsigned int * d_index_array,
                                   const Scalar * d_vec,
                                   Scalar4 * d_vec4,
                                   unsigned int block_size)
    {
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    hipLaunchKernelGGL(
        (gpu_convert_scalar_to_scalar4_kernel),
        dim3(grid),
        dim3(threads),
        0,
        0,
        group_size,
        d_index_array,
        d_vec,
        d_vec4);
    } /* end of gpu_convert_scalar_to_scalar4*/

__global__ void gpu_check_neighborlist_kernel(const unsigned int group_size,
                                              const unsigned int * d_index_array,
                                              const unsigned int * d_nneigh,
                                              const unsigned int * d_nlist,
                                              const size_t * d_headlist)
    {
    
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    printf("group size: %u, group idx: %d\n", group_size, group_idx);

    if (group_idx < group_size)
        {
        unsigned int idx = d_index_array[group_idx];
        // contribution from neighbors
        unsigned int nneigh_i = d_nneigh[idx];
        size_t head_i = d_headlist[idx];

        printf("Particle %u has %u neighbors: ", idx, nneigh_i);
        for (unsigned int neigh_idx = 0; neigh_idx < nneigh_i; neigh_idx++)
            {
            unsigned int jdx = d_nlist[neigh_idx + head_i];
            printf("particle %u, ", jdx);

            } // for neigh_idx
        printf("\n");

        }

    } /* end of gpu_check_neighborlist_kernel */

void gpu_check_neighborlist(const unsigned int group_size,
                            const unsigned int * d_index_array,
                            const unsigned int * d_nneigh,
                            const unsigned int * d_nlist,
                            const size_t * d_headlist,
                            unsigned int block_size)
    {
    unsigned int run_block_size = 32;

    dim3 grid( (group_size + run_block_size - 1)/ run_block_size, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    printf("group size = %u, run block size = %u\n", group_size, run_block_size);

    printf("kernel launched with grid = %u, threads = %u\n", grid.x, threads.x);

    printf("Checking device pointers:\n");
    printf("d_index_array: %p\n", d_index_array);
    printf("d_nneigh: %p\n", d_nneigh);
    printf("d_nlist: %p\n", d_nlist);
    printf("d_headlist: %p\n", d_headlist);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = d_index_array[group_idx];
        // contribution from neighbors
        unsigned int nneigh_i = d_nneigh[idx];
        size_t head_i = d_headlist[idx];

        printf("Particle %u has %u neighbors: ", idx, nneigh_i);
        for (unsigned int neigh_idx = 0; neigh_idx < nneigh_i; neigh_idx++)
            {
            unsigned int jdx = d_nlist[neigh_idx + head_i];
            printf("particle %u, ", jdx);

            } // for neigh_idx
        printf("\n");
        }
    

    hipLaunchKernelGGL(
        (gpu_check_neighborlist_kernel),
        dim3(grid),
        dim3(threads),
        0,
        0,
        group_size,
        d_index_array,
        d_nneigh,
        d_nlist,
        d_headlist);
    
    hipDeviceSynchronize();

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        {
        printf("Kernel launch failed: %s\n", hipGetErrorString(err));
        }
    else
        {
        printf("Kernel launch succeeded\n");
        }


    } /* end of gpu_check_neighborlist */

__global__ void gpu_compute_wave_value_kernel(const uint3 mesh_dim,
                                              const unsigned int n_wave_vectors,
                                              const Scalar xi2,
                                              const Scalar eta,
                                              const Scalar3 b1,
                                              const Scalar3 b2,
                                              const Scalar3 b3,
                                              Scalar4 * d_gridk,
                                              Scalar3 * d_ymob,
                                              Scalar2 * d_sf)
    {
    unsigned int kidx;
    kidx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int NNN = mesh_dim.x * mesh_dim.y * mesh_dim.z;
    // printf("kidx = %u\n", kidx);
    // printf("number of wave vectors = %u, NNN = %u\n", n_wave_vectors, NNN);

    if (kidx >= n_wave_vectors)
        {
        return;
        }
    // printf("In Compute Wave Value Kernel, kidx = %u\n", kidx );

    int l, m, n;
    n = kidx / mesh_dim.x / mesh_dim.y;
    m = (kidx - n * mesh_dim.x * mesh_dim.y) / mesh_dim.x;
    l = kidx % mesh_dim.x;

    if (l >= (int)(mesh_dim.x / 2 + mesh_dim.x % 2))
        l -= (int)mesh_dim.x;
    if (m >= (int)(mesh_dim.y / 2 + mesh_dim.y % 2))
        m -= (int)mesh_dim.y;
    if (n >= (int)(mesh_dim.z / 2 + mesh_dim.z % 2))
        n -= (int)mesh_dim.z;
    
    Scalar3 kvec = (Scalar)l * b1 + (Scalar)m * b2 + (Scalar)n * b3;
    Scalar kk = dot(kvec, kvec);
    Scalar k = sqrt(kk);

    if (l != 0 || m != 0 || n != 0)
        {
        Scalar kxi = kk / (4.0 * xi2);
        Scalar ya = 6.0 * M_PI * (1.0 + kxi) / kk * exp( - (1.0 - eta) * kxi ) / Scalar(NNN);
        Scalar yb = 0.5 * k * ya;
        Scalar yc = 0.25 * kk * ya;

        Scalar sf_uf = 1.0 - kk / 6.0;
        Scalar sf_ot = 1.0 - kk / 10.0;

        d_gridk[kidx] = make_scalar4(kvec.x / k, kvec.y / k, kvec.z / k, k);
        d_ymob[kidx].x = ya;
        d_ymob[kidx].y = yb;
        d_ymob[kidx].z = yc;

        d_sf[kidx].x = sf_uf;
        d_sf[kidx].y = sf_ot;
        }
    else
        {
        d_gridk[kidx] = make_scalar4(0.0, 0.0, 0.0, 0.0);
        d_ymob[kidx] = make_scalar3(0.0, 0.0, 0.0);
        d_sf[kidx] = make_scalar2(0.0, 0.0);
        }
    } /* end of gpu_compute_wave_value_kernel */

void gpu_compute_wave_value(const uint3 mesh_dim,
                            Scalar4 * d_gridk,
                            Scalar3 * d_ymob,
                            Scalar2 * d_sf,
                            const BoxDim& box,
                            const bool local_fft,
                            const Scalar xi2,
                            const Scalar eta,
                            unsigned int block_size)
    {
    // compute reciprocal lattice vectors
    Scalar3 a1 = box.getLatticeVector(0);
    Scalar3 a2 = box.getLatticeVector(1);
    Scalar3 a3 = box.getLatticeVector(2);
    Scalar V_box = box.getVolume();

    Scalar3 b1 = Scalar(2.0 * M_PI) 
                 * make_scalar3(a2.y * a3.z - a2.z * a3.y,
                                a2.z * a3.x - a2.x * a3.z,
                                a2.x * a3.y - a2.y * a3.x) 
                 / V_box;
    Scalar3 b2 = Scalar(2.0 * M_PI) 
                 * make_scalar3(a3.y * a1.z - a3.z * a1.y,
                                a3.z * a1.x - a3.x * a1.z,
                                a3.x * a1.y - a3.y * a1.x) 
                 / V_box;
    Scalar3 b3 = Scalar(2.0 * M_PI) 
                 * make_scalar3(a1.y * a2.z - a1.z * a2.y,
                                a1.z * a2.x - a1.x * a2.z,
                                a1.x * a2.y - a1.y * a2.x) 
                 / V_box;
    unsigned int num_wave_vectors = mesh_dim.x * mesh_dim.y * mesh_dim.z;

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_wave_value_kernel); //hipFuncGetAttributes(&attr, (const void*)gpu_compute_wave_value_kernel<true>);
    max_block_size = attr.maxThreadsPerBlock;
    std::cout << "CU: In Wave Computation" << std::endl;
    std::cout << "max block size = " << max_block_size << std::endl;
    unsigned int run_block_size = min(max_block_size, block_size);
    unsigned int n_blocks = num_wave_vectors / run_block_size;
    if (num_wave_vectors % run_block_size)
        {
        n_blocks += 1;
        }
    std::cout << "number of blocks = " << n_blocks << std::endl;
    std::cout << "number of threads per block = " << run_block_size << std::endl;
    std::cout << std::endl;
    dim3 grid(n_blocks, 1, 1);
    hipLaunchKernelGGL
        ((gpu_compute_wave_value_kernel),
        dim3(grid),
        dim3(run_block_size),
        0,
        0,
        mesh_dim,
        num_wave_vectors,
        xi2,
        eta,
        b1,
        b2,
        b3,
        d_gridk,
        d_ymob,
        d_sf);
    } /* end of gpu_compute_wave_value */

__device__ void mobility_real_func(Scalar r,
                                   Scalar xi,
                                   Scalar pisqrt, 
                                   Scalar xi_pisqrt_inv,
                                   Scalar& xa, 
                                   Scalar& ya)
    {
    Scalar xi2 = xi * xi;
    Scalar xir = xi * r;
    Scalar xir2 = xir * xir;
    Scalar expxir2 = fast::exp(-xir2);
    Scalar phi0 = pisqrt / xir * fast::erfc(xir);
    Scalar phi1 = Scalar(1.0) / xir2 * (Scalar(0.5) * phi0 + expxir2);
    Scalar phi2 = Scalar(1.0) / xir2 * (Scalar(1.5) * phi1 + expxir2);
    Scalar phi3 = Scalar(1.0) / xir2 * (Scalar(2.5) * phi2 + expxir2);
    Scalar phi4 = Scalar(1.0) / xir2 * (Scalar(3.5) * phi3 + expxir2);
    Scalar phi5 = Scalar(1.0) / xir2 * (Scalar(4.5) * phi4 + expxir2);

    xa = 0.0;
    ya = 0.0;

    xa = xi_pisqrt_inv * 
                    (
                    Scalar(2.0) * phi0
                    + xi2 / Scalar(3.0) * ( - Scalar(20.0) * phi1 + Scalar(8.0) * xir2 * phi2
                                    + xi2 / Scalar(3.0) * (
                                                    Scalar(70.0) * phi2 + xir2 * ( - Scalar(56.0) * phi3 + xir2 * (Scalar(8.0) * phi4) )
                                                )
                                    )
                    );
    ya = xi_pisqrt_inv * 
                    (
                    Scalar(2.0) * phi0 + xir2 * ( - Scalar(2.0) * phi1)
                    + xi2 / Scalar(3.0) * (
                                    - Scalar(20.0) * phi1 + xir2 * (Scalar(36.0) * phi2 + xir2 * ( - Scalar(8.0) * phi3))
                                    + xi2 / Scalar(3.0) * (
                                                    Scalar(70.0) * phi2 + xir2 * ( - Scalar(182.0) * phi3 + xir2 * (Scalar(80.0) * phi4 - xir2 * (Scalar(8.0) * phi5)) )
                                                )
                                    )
                    );

    xa *= Scalar(0.75);
    ya *= Scalar(0.75);
    } /* end of mobility_real_func*/

__global__ void gpu_mobility_real_uf_kernel(const unsigned int group_size,
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
                                            BoxDim box,
                                            Scalar * d_force,
                                            Scalar * d_vel)
    {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx < group_size)
        {
        unsigned int idx = d_index_array[group_idx];
        unsigned int i3 = idx * 3;

        Scalar3 U = make_scalar3(0.0, 0.0, 0.0);
        Scalar3 Fi = make_scalar3(d_force[i3], d_force[i3 + 1], d_force[i3 + 2]);

        // self contribution
        U.x = self_a.x * Fi.x;
        U.y = self_a.x * Fi.y;
        U.z = self_a.x * Fi.z;

        // contribution from neighbors
        unsigned int nneigh_i = d_nneigh[idx];
        size_t head_i = d_headlist[idx];

        Scalar3 posi = make_scalar3(d_postype[idx].x, d_postype[idx].y, d_postype[idx].z);

        for (unsigned int neigh_idx = 0; neigh_idx < nneigh_i; neigh_idx++)
            {
            unsigned int jdx = d_nlist[neigh_idx + head_i];
            unsigned int j3 = jdx * 3;

            Scalar3 posj = make_scalar3(d_postype[jdx].x, d_postype[jdx].y, d_postype[jdx].z);
            Scalar3 Fj = make_scalar3(d_force[j3], d_force[j3 + 1], d_force[j3 + 2]);

            Scalar3 dx = posj - posi;
            dx = box.minImage(dx);
            Scalar r2 = dot(dx, dx);
            Scalar r = sqrt(r2);

            if (r < rcut)
                {
                Scalar3 e = dx / r;
                if (r < 2.0) r = 2.0;
                
                Scalar xa, ya;
                mobility_real_func(r,
                                   xi,
                                   pisqrt,
                                   xi_pisqrt_inv,
                                   xa,
                                   ya);
                Scalar xmya = xa - ya;
                Scalar Fj_dot_e = dot(Fj, e);

                U.x += ya * Fj.x + xmya * Fj_dot_e * e.x;
                U.y += ya * Fj.y + xmya * Fj_dot_e * e.y;
                U.z += ya * Fj.z + xmya * Fj_dot_e * e.z;
                } // if r < rcut
            
            } // for neigh_idx
        
        d_vel[i3    ] = U.x;
        d_vel[i3 + 1] = U.y;
        d_vel[i3 + 2] = U.z;

        }
    
    } /* end of gpu_mobility_real_uf_kernel */

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
                          unsigned int block_size)
    {
    unsigned int n_blocks = group_size / block_size;
    if (group_size % block_size)
        {
        n_blocks += 1;
        }
    
    // std::cout << "\nCU: In Mobility Real" << std::endl;
    // std::cout << "number of blocks = " << n_blocks << std::endl;
    // std::cout << "number of threads per block = " << block_size << std::endl;
    // std::cout << std::endl;

    dim3 grid(n_blocks, 1, 1);
    dim3 threads(block_size, 1, 1);

    hipLaunchKernelGGL(
        (gpu_mobility_real_uf_kernel),
        dim3(grid),
        dim3(threads),
        0,
        0,
        group_size,
        d_postype,
        d_index_array,
        d_nneigh,
        d_nlist,
        d_headlist,
        self_a,
        xi,
        pisqrt,
        xi_pisqrt_inv,
        rcut,
        box,
        d_force,
        d_vel);

    } /* end of gpu_mobility_real_uf */

__device__ int3 find_cell_id(const Scalar3& pos,
                             const unsigned int& inner_nx,
                             const unsigned int& inner_ny,
                             const unsigned int& inner_nz,
                             const uint3& n_ghost_cells,
                             BoxDim box,
                             int P,
                             Scalar3& dr)
    {
    Scalar3 f = box.makeFraction(pos);
    uchar3 periodic = box.getPeriodic();

    Scalar3 reduced_pos = make_scalar3(f.x * (Scalar)inner_nx, f.y * (Scalar)inner_ny, f.z * (Scalar)inner_nz);
    reduced_pos += make_scalar3(n_ghost_cells.x, n_ghost_cells.y, n_ghost_cells.z);

    Scalar shift, shiftone;
    if (P % 2)
        {
        shift = Scalar(0.5);
        shiftone = Scalar(0.0);
        }
    else
        {
        shift = Scalar(0.0);
        shiftone = Scalar(0.5);
        }

    int ix = __scalar2int_rd(reduced_pos.x + shift);
    int iy = __scalar2int_rd(reduced_pos.y + shift);
    int iz = __scalar2int_rd(reduced_pos.z + shift);

    // set distance to cell center
    dr.x = shiftone + (Scalar)ix - reduced_pos.x;
    dr.y = shiftone + (Scalar)iy - reduced_pos.y;
    dr.z = shiftone + (Scalar)iz - reduced_pos.z;

    // handle particles on the boundary
    if (periodic.x && ix == (int)inner_nx)
        ix = 0;
    if (periodic.y && iy == (int)inner_ny)
        iy = 0;
    if (periodic.z && iz == (int)inner_nz)
        iz = 0;

    return make_int3(ix, iy, iz);

    } /* end of find_cell_id */

__global__ void gpu_assign_particle_force_kernel(const uint3 mesh_dim,
                                                 const uint3 n_ghost_bins,
                                                 const unsigned int work_size,
                                                 const unsigned int * d_index_array,
                                                 const Scalar4 * d_postype,
                                                 const Scalar * d_force,
                                                 BoxDim box,
                                                 const Scalar3 h,
                                                 const int P,
                                                 const Scalar gauss_fac,
                                                 const Scalar gauss_exp,
                                                 unsigned int offset,
                                                 hipfftComplex * d_mesh_Fx,
                                                 hipfftComplex * d_mesh_Fy,
                                                 hipfftComplex * d_mesh_Fz)
    {
    unsigned int work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_idx >= work_size)
        {
        return;
        }
    unsigned int group_idx = work_idx + offset;
    unsigned int idx = d_index_array[group_idx];
    unsigned int i3 = idx * 3;

    int3 bin_dim = make_int3(mesh_dim.x + 2 * n_ghost_bins.x,
                             mesh_dim.y + 2 * n_ghost_bins.y,
                             mesh_dim.z + 2 * n_ghost_bins.z);
    Scalar3 F = make_scalar3(d_force[i3], d_force[i3 + 1], d_force[i3 + 2]);

    Scalar4 postype = d_postype[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    Scalar3 dr = make_scalar3(0.0, 0.0, 0.0);
    int3 bin_coord = find_cell_id(pos,
                                  mesh_dim.x,
                                  mesh_dim.y,
                                  mesh_dim.z,
                                  n_ghost_bins,
                                  box,
                                  P,
                                  dr);
    if (bin_coord.x < 0 || bin_coord.x >= bin_dim.x ||
        bin_coord.y < 0 || bin_coord.y >= bin_dim.y ||
        bin_coord.z < 0 || bin_coord.z >= bin_dim.z)
        {
        return;
        }
    int nlower = - (static_cast<int>(P) - 1) / 2;
    int nupper = P / 2;
    // printf("nlower = %d, nupper = %d\n", nlower, nupper);

    int i = bin_coord.x;
    int j = bin_coord.y;
    int k = bin_coord.z;

    bool ignore_x = false;
    bool ignore_y = false;
    bool ignore_z = false;

    for (int l = nlower; l <= nupper; l++)
        {
        int neighi = i + l;
        if (neighi >= (int) bin_dim.x)
            {
            neighi -= (int) bin_dim.x;
            }
        else if (neighi < 0)
            {
            neighi += (int) bin_dim.x;
            }
        Scalar rx = (Scalar(l) + dr.x) * h.x;

        for (int m = nlower; m <= nupper; m++)
            {
            int neighj = j + m;
            if (neighj >= (int) bin_dim.y)
                {
                neighj -= (int) bin_dim.y;
                }
            else if (neighj < 0)
                {
                neighj += (int) bin_dim.y;
                }
            Scalar ry = (Scalar(m) + dr.y) * h.y;

            for (int n = nlower; n <= nupper; n++)
                {
                int neighk = k + n;
                if (neighk >= (int) bin_dim.z)
                    {
                    neighk -= (int) bin_dim.z;
                    }
                else if (neighk < 0)
                    {
                    neighk += (int) bin_dim.z;
                    }
                Scalar rz = (Scalar(n) + dr.z) * h.z;

                unsigned int cell_idx = neighi + bin_dim.x * (neighj + bin_dim.y * neighk);
                Scalar r2 = rx * rx + ry * ry + rz * rz;
                Scalar fac = gauss_fac * exp(- gauss_exp * r2);
                // printf("Particle %u: cell %u, r2 = %f, xfac = %f, yfac = %f, zfac = %f\n", idx, cell_idx, r2, F.x * fac, F.y * fac, F.z * fac);
                myAtomicAdd(&d_mesh_Fx[cell_idx].x, F.x * fac);
                myAtomicAdd(&d_mesh_Fy[cell_idx].x, F.y * fac);
                myAtomicAdd(&d_mesh_Fz[cell_idx].x, F.z * fac);

                // if (!ignore_x && !ignore_y && !ignore_z)
                //     {
                //     unsigned int cell_idx = neighi + bin_dim.x * (neighj + bin_dim.y * neighk);
                //     Scalar r2 = rx * rx + ry * ry + rz * rz;
                //     Scalar fac = gauss_fac * exp(- gauss_exp * r2);
                //     printf("Particle %u: cell %u, r2 = %f, xfac = %f, yfac = %f, zfac = %f\n", idx, cell_idx, r2, F.x * fac, F.y * fac, F.z * fac);
                //     myAtomicAdd(&d_mesh_Fx[cell_idx].x, F.x * fac);
                //     myAtomicAdd(&d_mesh_Fy[cell_idx].x, F.y * fac);
                //     myAtomicAdd(&d_mesh_Fz[cell_idx].x, F.z * fac);
                //     }
                
                ignore_z = false;
                } // for n
            
            ignore_y = false;
            } // for m
        
        ignore_x = false;
        } // for l
    
    } /* end of gpu_assign_particle_force_kernel */

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
                               unsigned int block_size)
    {
    hipMemsetAsync(d_mesh_Fx, 0, sizeof(hipfftComplex) * grid_dim.x * grid_dim.y * grid_dim.z);
    hipMemsetAsync(d_mesh_Fy, 0, sizeof(hipfftComplex) * grid_dim.x * grid_dim.y * grid_dim.z);
    hipMemsetAsync(d_mesh_Fz, 0, sizeof(hipfftComplex) * grid_dim.x * grid_dim.y * grid_dim.z);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_assign_particle_force_kernel);
    max_block_size = attr.maxThreadsPerBlock;
    
    // std::cout << "CU: In Particle Spreading" << std::endl;
    // std::cout << "max block size = " << max_block_size << std::endl;

    unsigned int run_block_size = min(max_block_size, block_size);
    // std::cout << "number of threads per block = " << run_block_size << std::endl;

    unsigned int n_blocks = group_size / run_block_size + 1;
    // std::cout << "number of blocks = " << n_blocks << std::endl;
    hipLaunchKernelGGL(
        (gpu_assign_particle_force_kernel),
        dim3(n_blocks),
        dim3(run_block_size),
        0,
        0,
        mesh_dim,
        n_ghost_bins,
        group_size,
        d_index_array,
        d_postype,
        d_force,
        box,
        h,
        P,
        gauss_fac,
        gauss_exp,
        0,
        d_mesh_Fx,
        d_mesh_Fy,
        d_mesh_Fz);
    // std::cout << std::endl;

    } /* end of gpu_assign_particle_force */

__global__ void gpu_mesh_green_kernel(const unsigned int n_wave_vectors,
                                      const Scalar4 * d_gridk,
                                      const Scalar3 * d_ymob,
                                      const Scalar2 * d_sf,
                                      hipfftComplex * d_mesh_Fx,
                                      hipfftComplex * d_mesh_Fy,
                                      hipfftComplex * d_mesh_Fz,
                                      hipfftComplex * d_fourier_mesh_Fx,
                                      hipfftComplex * d_fourier_mesh_Fy,
                                      hipfftComplex * d_fourier_mesh_Fz)
    {
    unsigned int kidx;
    kidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (kidx >= n_wave_vectors)
        return;
    
    hipfftComplex Fx = d_mesh_Fx[kidx];
    hipfftComplex Fy = d_mesh_Fy[kidx];
    hipfftComplex Fz = d_mesh_Fz[kidx];

    Scalar kx = d_gridk[kidx].x;
    Scalar ky = d_gridk[kidx].y;
    Scalar kz = d_gridk[kidx].z;

    Scalar ya = d_ymob[kidx].x;
    Scalar sf = d_sf[kidx].x;
    Scalar uf_fac = ya * sf * sf;

    hipfftComplex F_dot_k;
    F_dot_k.x = Fx.x * kx + Fy.x * ky + Fz.x * kz;
    F_dot_k.y = Fx.y * kx + Fy.y * ky + Fz.y * kz;

    hipfftComplex fourier_Fx;
    fourier_Fx.x = uf_fac * (Fx.x - kx * F_dot_k.x);
    fourier_Fx.y = uf_fac * (Fx.y - kx * F_dot_k.y);

    hipfftComplex fourier_Fy;
    fourier_Fy.x = uf_fac * (Fy.x - ky * F_dot_k.x);
    fourier_Fy.y = uf_fac * (Fy.y - ky * F_dot_k.y);

    hipfftComplex fourier_Fz;
    fourier_Fz.x = uf_fac * (Fz.x - kz * F_dot_k.x);
    fourier_Fz.y = uf_fac * (Fz.y - kz * F_dot_k.y);

    d_fourier_mesh_Fx[kidx] = fourier_Fx;
    d_fourier_mesh_Fy[kidx] = fourier_Fy;
    d_fourier_mesh_Fz[kidx] = fourier_Fz;
    } /* end of gpu_mesh_green_kernel */

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
                    unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_mesh_green_kernel);

    max_block_size = attr.maxThreadsPerBlock;
    // std::cout << "CU: In Mesh Green" << std::endl;
    // std::cout << "max block size = " << max_block_size << std::endl;
    unsigned int run_block_size = min(max_block_size, block_size);
    unsigned int n_blocks = n_wave_vectors / run_block_size;
    if (n_wave_vectors % run_block_size)
        {
        n_blocks += 1;
        }
    // std::cout << "number of blocks = " << n_blocks << std::endl;
    // std::cout << "number of threads per block = " << run_block_size << std::endl;
    // std::cout << std::endl;

    dim3 grid(n_blocks, 1, 1);
    hipLaunchKernelGGL(
        (gpu_mesh_green_kernel),
        dim3(grid),
        dim3(run_block_size),
        0,
        0,
        n_wave_vectors,
        d_gridk,
        d_ymob,
        d_sf,
        d_mesh_Fx,
        d_mesh_Fy,
        d_mesh_Fz,
        d_fourier_mesh_Fx,
        d_fourier_mesh_Fy,
        d_fourier_mesh_Fz);

    } /* end of gpu_mesh_green */

__global__ void gpu_interpolate_particle_velocity_kernel(const unsigned int work_size,
                                                         const uint3 mesh_dim,
                                                         const uint3 n_ghost_cells,
                                                         const unsigned int * d_index_array,
                                                         const Scalar4 * d_postype,
                                                         const int P,
                                                         const Scalar gauss_fac,
                                                         const Scalar gauss_exp,
                                                         const Scalar vk,
                                                         const Scalar3 h,
                                                         BoxDim box,
                                                         unsigned int offset,
                                                         const hipfftComplex * d_mesh_inv_Fx,
                                                         const hipfftComplex * d_mesh_inv_Fy,
                                                         const hipfftComplex * d_mesh_inv_Fz,
                                                         Scalar * d_vel)
    {
    unsigned int work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_idx >= work_size)
        return;
    
    unsigned int group_idx = work_idx + offset;
    unsigned int idx = d_index_array[group_idx];
    unsigned int i3 = idx * 3;

    Scalar4 postype = d_postype[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    Scalar3 dr = make_scalar3(0.0, 0.0, 0.0);

    int3 cell_coord = find_cell_id(pos,
                                   mesh_dim.x,
                                   mesh_dim.y,
                                   mesh_dim.z,
                                   n_ghost_cells,
                                   box,
                                   P,
                                  dr);
    
    if (cell_coord.x < 0 || cell_coord.x >= (int)mesh_dim.x ||
        cell_coord.y < 0 || cell_coord.y >= (int)mesh_dim.y ||
        cell_coord.z < 0 || cell_coord.z >= (int)mesh_dim.z)
        {
        return;
        }
    
    Scalar3 vel = make_scalar3(0.0, 0.0, 0.0);

    int nlower = - (static_cast<int>(P) - 1) / 2;
    int nupper = P / 2;

    int i = cell_coord.x;
    int j = cell_coord.y;
    int k = cell_coord.z;

    for (int l = nlower; l <= nupper; l++)
        {
        int neighi = i + l;
        if (neighi >= (int)mesh_dim.x)
            neighi -= mesh_dim.x;
        else if (neighi < 0)
            neighi += mesh_dim.x;
        Scalar rx = (Scalar(l) + dr.x) * h.x;

        for (int m = nlower; m <= nupper; m++)
            {
            int neighj = j + m;
            if (neighj >= (int)mesh_dim.y)
                neighj -= mesh_dim.y;
            else if (neighj < 0)
                neighj += mesh_dim.y;
            Scalar ry = (Scalar(m) + dr.y) * h.y;

            for (int n = nlower; n <= nupper; n++)
                {
                int neighk = k + n;
                if (neighk >= (int)mesh_dim.z)
                    neighk -= mesh_dim.z;
                else if (neighk < 0)
                    neighk += mesh_dim.z;
                Scalar rz = (Scalar(n) + dr.z) * h.z;

                unsigned int cell_idx = neighi + mesh_dim.x * (neighj + mesh_dim.y * neighk);
                Scalar r2 = rx * rx + ry * ry + rz * rz;
                Scalar fac = vk * gauss_fac * exp(- gauss_exp * r2);

                hipfftComplex inv_mesh_Fx = d_mesh_inv_Fx[cell_idx];
                hipfftComplex inv_mesh_Fy = d_mesh_inv_Fy[cell_idx];
                hipfftComplex inv_mesh_Fz = d_mesh_inv_Fz[cell_idx];

                vel.x += fac * inv_mesh_Fx.x;
                vel.y += fac * inv_mesh_Fy.x;
                vel.z += fac * inv_mesh_Fz.x;
                } // for n

            } // for m

        } // for l
    
    d_vel[i3    ] = vel.x;
    d_vel[i3 + 1] = vel.y;
    d_vel[i3 + 2] = vel.z;
    
    } /* end of gpu_interpolate_particle_velocity */

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
                                       unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_interpolate_particle_velocity_kernel);
    max_block_size = attr.maxThreadsPerBlock;
    // std::cout << "CU: In Particle Velocity Interpolation" << std::endl;
    // std::cout << "max block size = " << max_block_size << std::endl;

    unsigned int run_block_size = min(max_block_size, block_size);
    // std::cout << "number of threads per block = " << run_block_size << std::endl;
    
    // reset force for all particles
    hipMemsetAsync(d_vel, 0, 3 * sizeof(Scalar) * group_size);
    unsigned int n_blocks = group_size / run_block_size + 1;
    // std::cout << "number of blocks = " << n_blocks << std::endl;
    hipLaunchKernelGGL(
        (gpu_interpolate_particle_velocity_kernel),
        dim3(n_blocks),
        dim3(run_block_size),
        0,
        0,
        group_size,
        mesh_dim,
        n_ghost_bins,
        d_index_array,
        d_postype,
        P,
        gauss_fac,
        gauss_exp,
        vk,
        h,
        box,
        0,
        d_mesh_inv_Fx,
        d_mesh_inv_Fy,
        d_mesh_inv_Fz,
        d_vel);

    // std::cout << std::endl;

    } /* end of gpu_interpolate_particle_velocity */

__global__ void gpu_mobility_velocity_sum_kernel(const unsigned int group_size,
                                                 const unsigned int * d_index_array,
                                                 const Scalar * d_u1,
                                                 const Scalar * d_u2,
                                                 Scalar * d_u)
    {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx < group_size)
        {
        unsigned int idx = d_index_array[group_idx];
        unsigned int i3 = idx * 3;

        d_u[i3    ] = d_u1[i3    ] + d_u2[i3    ];
        d_u[i3 + 1] = d_u1[i3 + 1] + d_u2[i3 + 1];
        d_u[i3 + 2] = d_u1[i3 + 2] + d_u2[i3 + 2];
        }
    
    } /* end of gpu_mobility_velocity_sum_kernel */

void gpu_mobility_velocity_sum(const unsigned int group_size,
                               const unsigned int * d_index_array,
                               const Scalar * d_u1,
                               const Scalar * d_u2,
                               Scalar * d_u,
                               unsigned int block_size)
    {
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    hipLaunchKernelGGL(
        (gpu_mobility_velocity_sum_kernel),
        dim3(grid),
        dim3(threads),
        0,
        0,
        group_size,
        d_index_array,
        d_u1,
        d_u2,
        d_u);

    } /* end of gpu_mobility_velocity_sum */

__global__ void gpu_brownian_farfield_particle_rng_kernel(Scalar kT,
                                                          uint64_t timestep,
                                                          uint16_t seed,
                                                          Scalar dt,
                                                          const unsigned int group_size,
                                                          const unsigned int * d_index_array,
                                                          const unsigned int * d_tag,
                                                          Scalar * d_psi)
    {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        Scalar fac = sqrt(6.0 * kT / dt);
        unsigned int idx = d_index_array[group_idx];
        unsigned int i3 = idx * 3;
        unsigned int ptag = d_tag[idx];

        RandomGenerator rng(hoomd::Seed(RNGIdentifier::StokesMreal, timestep, seed),
                            hoomd::Counter(ptag));
        UniformDistribution<Scalar> uniform(Scalar(- 1.0), Scalar(1.0));
        Scalar rx = uniform(rng);
        Scalar ry = uniform(rng);
        Scalar rz = uniform(rng);

        d_psi[i3    ] = fac * rx;
        d_psi[i3 + 1] = fac * ry;
        d_psi[i3 + 2] = fac * rz;
        }
    
    } /* end of gpu_brownian_farfield_particle_rng_kernel */

void gpu_brownian_farfield_particle_rng(const Scalar kT,
                                        const uint64_t timestep,
                                        const uint16_t seed,
                                        const Scalar dt,
                                        const unsigned int group_size,
                                        const unsigned int * d_index_array,
                                        const unsigned int * d_tag,
                                        Scalar * d_psi,
                                        unsigned int block_size)
    {
    unsigned int n_blocks = group_size / block_size;
    if (group_size % block_size)
        {
        n_blocks += 1;
        }
    
    // std::cout << "CU: In Far-Field Particle RNG" << std::endl;
    // std::cout << "number of threads per block = " << block_size << std::endl;
    // std::cout << "number of blocks = " << n_blocks << std::endl;

    dim3 grid(n_blocks, 1, 1);
    dim3 threads(block_size, 1, 1);

    hipLaunchKernelGGL(
        (gpu_brownian_farfield_particle_rng_kernel),
        dim3(grid),
        dim3(threads),
        0,
        0,
        kT,
        timestep,
        seed,
        dt,
        group_size,
        d_index_array,
        d_tag,
        d_psi);
    // std::cout << std::endl;

    } /* end of gpu_brownian_farfield_particle_rng */

__global__ void gpu_brownian_mapback_kernel(Scalar * d_V,
                                            Scalar * d_Tm,
                                            unsigned int m,
                                            unsigned int group_size,
                                            const unsigned int * d_index_array,
                                            unsigned int numel,
                                            Scalar * d_x)
    {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx < group_size)
        {
        unsigned int idx = d_index_array[group_idx];
        unsigned int i3 = idx * 3;

        d_x[i3    ] = 0.0;
        d_x[i3 + 1] = 0.0;
        d_x[i3 + 2] = 0.0;

        for (unsigned int j = 0; j < m; j++)
            {
            Scalar fac = d_Tm[j];

            d_x[i3    ] += d_V[j * numel + i3    ] * fac;
            d_x[i3 + 1] += d_V[j * numel + i3 + 1] * fac;
            d_x[i3 + 2] += d_V[j * numel + i3 + 2] * fac;
            }
        
        } // if 

    } /* end of gpu_brownian_mapback_kernel */

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
                          unsigned int block_size)
    {
    unsigned int n_blocks = group_size / block_size;
    if (group_size % block_size)
        {
        n_blocks += 1;
        }
    // std::cout << "CU: In Far-Field Lanczos" << std::endl;
    // std::cout << "number of threads per block = " << block_size << std::endl;
    // std::cout << "number of blocks = " << n_blocks << std::endl;

    dim3 grid(n_blocks, 1, 1);
    dim3 threads(block_size, 1, 1);

    unsigned int numel = group_size * 3;
    unsigned int m_in = m_lanczos_ff;

    Scalar * d_v = d_iter_ff_v;
    Scalar * d_V = d_iter_ff_V;
    Scalar * d_vj = d_iter_ff_vj;
    Scalar * d_vjm1 = d_iter_ff_vjm1;
    Scalar * d_uold = d_iter_ff_uold;
    Scalar * d_Mpsi = d_iter_Mpsi;
    Scalar * d_temp;

    double * alpha = new double[mmax];
    double * alpha_save = new double[mmax];
    double * beta = new double[mmax + 1];
    double * beta_save = new double[mmax + 1];
    double * W = new double[mmax * mmax];
    double * W1 = new double[mmax];
    double * Tm = new double [mmax];
    
    // copy psi to v0
    cudaMemcpy(d_vj, d_psi, numel * sizeof(Scalar), cudaMemcpyDeviceToDevice);

    Scalar vnorm, psinorm;
    cublasDnrm2(blasHandle, 
                numel, 
                d_psi, 1,
                &vnorm);
    psinorm = vnorm;

    // compute psi . M . psi (for step norm)
    gpu_mobility_real_uf(group_size,
                         d_postype,
                         d_index_array,
                         d_nneigh,
                         d_nlist,
                         d_headlist,
                         self_a,
                         xi,
                         pisqrt,
                         xi_pisqrt_inv,
                         rcut,
                         box,
                         d_psi,
                         d_Mpsi,
                         block_size);
    Scalar psiMpsi;
    cublasDdot(blasHandle,
               numel,
               d_psi, 1,
               d_Mpsi, 1,
               &psiMpsi);
    psiMpsi = psiMpsi / (psinorm * psinorm);

    unsigned int m = m_in - 1;
    m = m < 1 ? 1 : m;

    Scalar alpha_temp;
    Scalar beta_temp = 0.0;

    // first iteration
    // scale v_{j-1} by 0 and v_{j} by 1/vnorm
    Scalar scale = 0.0;
    cublasDscal(blasHandle,
                numel,
                &scale,
                d_vjm1, 1);
    scale = 1.0 / vnorm;
    cublasDscal(blasHandle,
                numel,
                &scale,
                d_vj, 1);

    for (unsigned int j = 0; j < m; j++)
        {
        cudaMemcpy(&d_V[j * numel], d_vj, numel * sizeof(Scalar), cudaMemcpyDeviceToDevice);
        beta[j] = beta_temp;

        // v = M . v_{j} - beta_{j} * v_{j-1}
        gpu_mobility_real_uf(group_size,
                             d_postype,
                             d_index_array,
                             d_nneigh,
                             d_nlist,
                             d_headlist,
                             self_a,
                             xi,
                             pisqrt,
                             xi_pisqrt_inv,
                             rcut,
                             box,
                             d_vj,
                             d_v,
                             block_size);
        scale = - beta_temp;
        cublasDaxpy(blasHandle,
                    numel,
                    &scale,
                    d_vjm1, 1,
                    d_v, 1);

        // alpha_{j} = v_{j} . v
        cublasDdot(blasHandle,
                   numel,
                   d_v, 1,
                   d_vj, 1,
                   &alpha_temp);
        alpha[j] = alpha_temp;

        // v = v - alpha_{j} * v_{j}
        scale = - alpha_temp;
        cublasDaxpy(blasHandle,
                    numel,
                    &scale,
                    d_vj, 1,
                    d_v, 1);
        
        // beta_{j+1} = norm(v)
        cublasDnrm2(blasHandle,
                    numel,
                    d_v, 1,
                    &beta_temp);
        beta[j + 1] = beta_temp;

        if (beta_temp < 1E-8)
            {
            m = j + 1;
            break;
            }
        
        // v = v / beta_{j+1}
        scale = 1.0 / beta_temp;
        cublasDscal(blasHandle,
                    numel,
                    &scale,
                    d_v, 1);
        
        // swap pointers
        d_temp = d_vjm1;
        d_vjm1 = d_vj;
        d_vj = d_v;
        d_v = d_temp;
        } // for j
    
    // save alpha and beta (will be overwritten by LAPACK)
    for (unsigned int i = 0; i < m; i++)
        {
        alpha_save[i] = alpha[i];
        beta_save[i] = beta[i];
        }
    beta_save[m] = beta[m];

    // compute eigen-decomposition of tridiagonal matrix
    int INFO = LAPACKE_dpteqr(LAPACK_ROW_MAJOR, 'I', m, alpha, &beta[1], W, m);
    if (INFO != 0)
        {
        std::cout << "Eigenvalue decomposition #1 failed." << std::endl;
        std::cout << "INFO = " << INFO << std::endl;
        std::cout << "m = " << m << std::endl;
            std::cout << "alpha: " << std::endl;
            for (unsigned int k = 0; k < m; k++)
                {
                std::cout << alpha[k] << std::endl;
                }
            std::cout << std::endl;

            std::cout << "beta: " << std::endl;
            for (unsigned int k = 0; k < m; k++)
                {
                std::cout << beta[k] << std::endl;
                }
            std::cout << std::endl;
        exit(EXIT_FAILURE); 
        }
    for (unsigned int i = 0; i < m; i++)
        {
        W1[i] = sqrt(alpha[i]) * W[i];
        }
    double sum_temp;
    for (unsigned int i = 0; i < m; i++)
        {
        sum_temp = 0.0;
        for (unsigned int k = 0; k < m; k++)
            {
            unsigned int idx = i * m + k;
            sum_temp += W[idx] * W1[k];
            } // for j

            Tm[i] = sum_temp;
        }
    // copy Tm to GPU
    cudaMemcpy(d_Tm, Tm, m * sizeof(Scalar), cudaMemcpyHostToDevice);

    // multiply basis vectors by Tm to get x = A \cdot Tm
    hipLaunchKernelGGL(
        (gpu_brownian_mapback_kernel),
        dim3(grid),
        dim3(threads),
        0,
        0,
        d_V,
        d_Tm,
        m,
        group_size,
        d_index_array,
        numel,
        d_uslip_real);

    cudaMemcpy(d_uold,
               d_uslip_real,
               numel * sizeof(Scalar),
               cudaMemcpyDeviceToDevice);

    // recover alpha and beta
    for (unsigned int i = 0; i < m; i++)
        {
        alpha[i] = alpha_save[i];
        beta[i] = beta_save[i];
        }
    beta[m] = beta_save[m];

    // keep adding to basis vectors untill step norm is small enough
    Scalar stepnorm = 1.0;
    unsigned int j;

    while (stepnorm > error && m < mmax)
        {
        m++;
        j = m - 1;

        cudaMemcpy(&d_V[j * numel], d_vj, numel * sizeof(Scalar), cudaMemcpyDeviceToDevice);
        beta[j] = beta_temp;

        // v = M . v_{j} - beta_{j} * v_{j-1}
        gpu_mobility_real_uf(group_size,
                             d_postype,
                             d_index_array,
                             d_nneigh,
                             d_nlist,
                             d_headlist,
                             self_a,
                             xi,
                             pisqrt,
                             xi_pisqrt_inv,
                             rcut,
                             box,
                             d_vj,
                             d_v,
                             block_size);
        scale = - beta_temp;
        cublasDaxpy(blasHandle,
                    numel,
                    &scale,
                    d_vjm1, 1,
                    d_v, 1);

        // alpha_{j} = v_{j} . v
        cublasDdot(blasHandle,
                   numel,
                   d_v, 1,
                   d_vj, 1,
                   &alpha_temp);
        alpha[j] = alpha_temp;

        // v = v - alpha_{j} * v_{j}
        scale = - alpha_temp;
        cublasDaxpy(blasHandle,
                    numel,
                    &scale,
                    d_vj, 1,
                    d_v, 1);
        
        // beta_{j+1} = norm(v)
        cublasDnrm2(blasHandle,
                    numel,
                    d_v, 1,
                    &beta_temp);
        beta[j + 1] = beta_temp;

        if (beta_temp < 1E-8)
            {
            m = j + 1;
            break;
            }
        
        // v = v / beta_{j+1}
        scale = 1.0 / beta_temp;
        cublasDscal(blasHandle,
                    numel,
                    &scale,
                    d_v, 1);
        
        // swap pointers
        d_temp = d_vjm1;
        d_vjm1 = d_vj;
        d_vj = d_v;
        d_v = d_temp;

        // save alpha and beta (will be overwritten by LAPACK)
        for (unsigned int i = 0; i < m; i++)
            {
            alpha_save[i] = alpha[i];
            beta_save[i] = beta[i];
            }
        beta_save[m] = beta[m];

        // compute eigen-decomposition of tridiagonal matrix
        int INFO = LAPACKE_dpteqr(LAPACK_ROW_MAJOR, 'I', m, alpha, &beta[1], W, m);
        if (INFO != 0)
            {
            std::cout << "Eigenvalue decomposition #1 failed." << std::endl;
            std::cout << "INFO = " << INFO << std::endl;
            std::cout << "m = " << m << std::endl;
                std::cout << "alpha: " << std::endl;
                for (unsigned int k = 0; k < m; k++)
                    {
                    std::cout << alpha[k] << std::endl;
                    }
                std::cout << std::endl;

                std::cout << "beta: " << std::endl;
                for (unsigned int k = 0; k < m; k++)
                    {
                    std::cout << beta[k] << std::endl;
                    }
                std::cout << std::endl;
            exit(EXIT_FAILURE); 
            }
        for (unsigned int i = 0; i < m; i++)
            {
            W1[i] = sqrt(alpha[i]) * W[i];
            }
        double sum_temp;
        for (unsigned int i = 0; i < m; i++)
            {
            sum_temp = 0.0;
            for (unsigned int k = 0; k < m; k++)
                {
                unsigned int idx = i * m + k;
                sum_temp += W[idx] * W1[k];
                } // for j

                Tm[i] = sum_temp;
            }
        // copy Tm to GPU
        cudaMemcpy(d_Tm, Tm, m * sizeof(Scalar), cudaMemcpyHostToDevice);

        hipLaunchKernelGGL(
            (gpu_brownian_mapback_kernel),
            dim3(grid),
            dim3(threads),
            0,
            0,
            d_V,
            d_Tm,
            m,
            group_size,
            d_index_array,
            numel,
            d_uslip_real);

        // compute step norm error
        scale = - 1.0;
        cublasDaxpy(blasHandle,
                    numel,
                    &scale,
                    d_uslip_real, 1,
                    d_uold, 1);
        cublasDdot(blasHandle,
                   numel,
                   d_uold, 1,
                   d_uold, 1,
                   &stepnorm);
        stepnorm = sqrt(stepnorm / psiMpsi);
        cudaMemcpy(d_uold, d_uslip_real, numel * sizeof(Scalar), cudaMemcpyDeviceToDevice);
        
        for (unsigned int i = 0; i < m; i++)
            {
            alpha[i] = alpha_save[i];
            beta[i] = beta_save[i];
            }
        beta[m] = beta_save[m];
        } // while
    cublasDscal(blasHandle,
                numel,
                &psinorm,
                d_uslip_real, 1);

    d_v = NULL;
    d_V = NULL;
    d_vj = NULL;
    d_vjm1 = NULL;
    d_uold = NULL;
    d_Mpsi = NULL;
    d_temp = NULL;

    delete [] alpha;
    delete [] alpha_save;
    delete [] beta;
    delete [] beta_save;
    delete [] W;
    delete [] W1;
    delete [] Tm;
    } /* end of gpu_brownian_lanczos */

__global__ void gpu_brownian_farfield_grid1_rng_kernel(const unsigned int n_wave_vectors,
                                                       const uint64_t timestep,
                                                       const uint16_t seed,
                                                       const uint3 mesh_dim,
                                                       const Scalar vk,
                                                       const Scalar T,
                                                       const Scalar dt,
                                                       hipfftComplex * d_mesh_inv_Fx,
                                                       hipfftComplex * d_mesh_inv_Fy,
                                                       hipfftComplex * d_mesh_inv_Fz)
    {
    unsigned int kidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (kidx >= n_wave_vectors)
        {
        return;
        }
    Scalar fac = sqrt(3.0 * T / dt / vk);
    Scalar sqrt2 = sqrt(2.0);

    RandomGenerator rng(hoomd::Seed(RNGIdentifier::StokesMwave, timestep, seed), 
                        hoomd::Counter(kidx));
    UniformDistribution<Scalar> uniform(Scalar(- 1.0), Scalar(1.0));

    Scalar rex = fac * uniform(rng);
    Scalar rey = fac * uniform(rng);
    Scalar rez = fac * uniform(rng);

    Scalar imx = fac * uniform(rng);
    Scalar imy = fac * uniform(rng);
    Scalar imz = fac * uniform(rng);

    unsigned int nkx = mesh_dim.x;
    unsigned int nky = mesh_dim.y;
    unsigned int nkz = mesh_dim.z;

    unsigned int k = kidx / (nkx * nky);
    unsigned int j = (kidx - k * nky * nkx) / nkx;
    unsigned int i = kidx % nkx;

    if ( !(2 * k >= nkz + 1) &&
         !( (k == 0) && (2 * j >= nky + 1) ) &&
         !( (k == 0) && (j == 0) && (2 * i >= nkx + 1) ) &&
         !( (k == 0) && (j == 0) && (i == 0) )
        )
        {
        bool i_nyquist = ( (i == nkx / 2) && (nkx / 2 == (nkx + 1) / 2) );
        bool j_nyquist = ( (j == nky / 2) && (nky / 2 == (nky + 1) / 2) );
        bool k_nyquist = ( (k == nkz / 2) && (nkz / 2 == (nkz + 1) / 2) );

        // index of conjugate point
        unsigned int i_conj, j_conj, k_conj;
        if (i == 0 || i_nyquist)
            {
            i_conj = i;
            }
        else
            {
            i_conj = nkx - i;
            }

        if (j == 0 || j_nyquist)
            {
            j_conj = j;
            }
        else
            {
            j_conj = nky - j;
            }

        if (k == 0 || k_nyquist)
            {
            k_conj = k;
            }
        else
            {
            k_conj = nkz - k;
            }

        unsigned int id_conj = i_conj + nkx * (j_conj + nky * k_conj);
        if (kidx > id_conj)
            {
            return;
            }
        

        if ( (i == 0    && j_nyquist && k == 0)    ||
             (i_nyquist && j == 0    && k == 0)    ||
             (i_nyquist && j_nyquist && k == 0)    ||
             (i == 0    && j == 0    && k_nyquist) ||
             (i == 0    && j_nyquist && k_nyquist) ||
             (i_nyquist && j == 0    && k_nyquist) ||
             (i_nyquist && j_nyquist && k_nyquist)
            )
            {
            d_mesh_inv_Fx[kidx].x = sqrt2 * rex;
            d_mesh_inv_Fx[kidx].y = 0.0;

            d_mesh_inv_Fy[kidx].x = sqrt2 * rey;
            d_mesh_inv_Fy[kidx].y = 0.0;

            d_mesh_inv_Fz[kidx].x = sqrt2 * rez;
            d_mesh_inv_Fz[kidx].y = 0.0;
            }
        else
            {
            d_mesh_inv_Fx[kidx].x = rex;
            d_mesh_inv_Fx[kidx].y = imx;

            d_mesh_inv_Fy[kidx].x = rey;
            d_mesh_inv_Fy[kidx].y = imy;

            d_mesh_inv_Fz[kidx].x = rez;
            d_mesh_inv_Fz[kidx].y = imz;

            // conjugate points
            d_mesh_inv_Fx[id_conj].x = + rex;
            d_mesh_inv_Fx[id_conj].y = - imx;

            d_mesh_inv_Fy[id_conj].x = + rey;
            d_mesh_inv_Fy[id_conj].y = - imy;

            d_mesh_inv_Fz[id_conj].x = + rez;
            d_mesh_inv_Fz[id_conj].y = - imz;
            }

        } // if (half the grid)

    } /* end of gpu_brownian_farfield_grid1_rng_kernel */

// __global__ void gpu_brownian_farfield_grid1_rng_kernel(const unsigned int n_wave_vectors,
    //                                                    const uint64_t timestep,
    //                                                    const uint16_t seed,
    //                                                    const uint3 mesh_dim,
    //                                                    const Scalar vk,
    //                                                    const Scalar T,
    //                                                    const Scalar dt,
    //                                                    hipfftComplex * d_mesh_inv_Fx,
    //                                                    hipfftComplex * d_mesh_inv_Fy,
    //                                                    hipfftComplex * d_mesh_inv_Fz)
    // {
    // unsigned int kidx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (kidx >= n_wave_vectors)
    //     {
    //     return;
    //     }
    
    // unsigned int nkx = mesh_dim.x;
    // unsigned int nky = mesh_dim.y;
    // unsigned int nkz = mesh_dim.z;

    // unsigned int k = kidx / (nkx * nky);
    // unsigned int j = (kidx - k * nky * nkx) / nkx;
    // unsigned int i = kidx % nkx;

    // bool i_nyquist = ( (i == nkx / 2) && (nkx / 2 == (nkx + 1) / 2) );
    // bool j_nyquist = ( (j == nky / 2) && (nky / 2 == (nky + 1) / 2) );
    // bool k_nyquist = ( (k == nkz / 2) && (nkz / 2 == (nkz + 1) / 2) );

    // unsigned int i_conj = (i == 0 || i_nyquist) ? i : nkx - i;
    // unsigned int j_conj = (j == 0 || j_nyquist) ? j : nky - j;
    // unsigned int k_conj = (k == 0 || k_nyquist) ? k : nkz - k;

    // unsigned int id_conj = i_conj + nkx * (j_conj + nky * k_conj);
    // if (kidx > id_conj)
    //     return;

    // Scalar fac = sqrt(3.0 * T / dt / vk);
    // Scalar sqrt2 = sqrt(2.0);

    // RandomGenerator rng(hoomd::Seed(RNGIdentifier::StokesMwave, timestep, seed), 
    //                     hoomd::Counter(kidx));
    // UniformDistribution<Scalar> uniform(Scalar(- 1.0), Scalar(1.0));
    // Scalar rex = fac * uniform(rng);
    // Scalar rey = fac * uniform(rng);
    // Scalar rez = fac * uniform(rng);

    // Scalar imx = fac * uniform(rng);
    // Scalar imy = fac * uniform(rng);
    // Scalar imz = fac * uniform(rng);

    // if ( (i == 0    && j_nyquist && k == 0)    ||
    //      (i_nyquist && j == 0    && k == 0)    ||
    //      (i_nyquist && j_nyquist && k == 0)    ||
    //      (i == 0    && j == 0    && k_nyquist) ||
    //      (i == 0    && j_nyquist && k_nyquist) ||
    //      (i_nyquist && j == 0    && k_nyquist) ||
    //      (i_nyquist && j_nyquist && k_nyquist)
    //     )
    //     {
    //     d_mesh_inv_Fx[kidx].x = sqrt2 * rex;
    //     d_mesh_inv_Fx[kidx].y = 0.0;

    //     d_mesh_inv_Fy[kidx].x = sqrt2 * rey;
    //     d_mesh_inv_Fy[kidx].y = 0.0;

    //     d_mesh_inv_Fz[kidx].x = sqrt2 * rez;
    //     d_mesh_inv_Fz[kidx].y = 0.0;
    //     }
    // else
    //     {
    //     d_mesh_inv_Fx[kidx].x = rex;
    //     d_mesh_inv_Fx[kidx].y = imx;

    //     d_mesh_inv_Fy[kidx].x = rey;
    //     d_mesh_inv_Fy[kidx].y = imy;

    //     d_mesh_inv_Fz[kidx].x = rez;
    //     d_mesh_inv_Fz[kidx].y = imz;

    //     // conjugate points
    //     d_mesh_inv_Fx[id_conj].x = + rex;
    //     d_mesh_inv_Fx[id_conj].y = - imx;

    //     d_mesh_inv_Fy[id_conj].x = + rey;
    //     d_mesh_inv_Fy[id_conj].y = - imy;

    //     d_mesh_inv_Fz[id_conj].x = + rez;
    //     d_mesh_inv_Fz[id_conj].y = - imz;
    //     }

    // } /* end of gpu_brownian_farfield_grid1_rng_kernel */

__global__ void gpu_brownian_farfield_grid2_rng_kernel(const unsigned int n_wave_vectors,
                                                       const Scalar4 * d_gridk,
                                                       const Scalar3 * d_ymob,
                                                       const Scalar2 * d_sf,
                                                       hipfftComplex * d_mesh_inv_Fx,
                                                       hipfftComplex * d_mesh_inv_Fy,
                                                       hipfftComplex * d_mesh_inv_Fz)
    {
    unsigned int kidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (kidx >= n_wave_vectors)
        {
        return;
        }
    Scalar kx = d_gridk[kidx].x;
    Scalar ky = d_gridk[kidx].y;
    Scalar kz = d_gridk[kidx].z;

    Scalar B = (kidx == 0) ? 0.0 : sqrt(d_ymob[kidx].x);
    Scalar sf = (kidx == 0) ? 0.0 : d_sf[kidx].x;
    Scalar fac = B * sf;

    hipfftComplex Fx = d_mesh_inv_Fx[kidx];
    hipfftComplex Fy = d_mesh_inv_Fy[kidx];
    hipfftComplex Fz = d_mesh_inv_Fz[kidx];

    hipfftComplex F_dot_k;
    if (kidx == 0)
        {
        F_dot_k.x = 0.0;
        F_dot_k.y = 0.0;
        }
    else
        {
        F_dot_k.x = Fx.x * kx + Fy.x * ky + Fz.x * kz;
        F_dot_k.y = Fx.y * kx + Fy.y * ky + Fz.y * kz;
        }
    
    d_mesh_inv_Fx[kidx].x = fac * (Fx.x - F_dot_k.x * kx);
    d_mesh_inv_Fx[kidx].y = fac * (Fx.y - F_dot_k.y * kx);

    d_mesh_inv_Fy[kidx].x = fac * (Fy.x - F_dot_k.x * ky);
    d_mesh_inv_Fy[kidx].y = fac * (Fy.y - F_dot_k.y * ky);

    d_mesh_inv_Fz[kidx].x = fac * (Fz.x - F_dot_k.x * kz);
    d_mesh_inv_Fz[kidx].y = fac * (Fz.y - F_dot_k.y * kz);

    } /* end of gpu_brownian_farfield_grid2_rng_kernel */

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
                                    unsigned int block_size)
    {
    // zero mesh
    hipMemsetAsync(d_mesh_inv_Fx, 0, sizeof(hipfftComplex) * mesh_dim.x  * mesh_dim.y * mesh_dim.z);
    hipMemsetAsync(d_mesh_inv_Fy, 0, sizeof(hipfftComplex) * mesh_dim.x  * mesh_dim.y * mesh_dim.z);
    hipMemsetAsync(d_mesh_inv_Fz, 0, sizeof(hipfftComplex) * mesh_dim.x  * mesh_dim.y * mesh_dim.z);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_brownian_farfield_grid1_rng_kernel);
    max_block_size = attr.maxThreadsPerBlock;
    // std::cout << "CU: In Grid RNG" << std::endl;
    // std::cout << "max block size = " << max_block_size << std::endl;

    unsigned int run_block_size = min(max_block_size, block_size);
    unsigned int n_blocks = n_wave_vectors / run_block_size;
    if (n_wave_vectors % run_block_size)
        {
        n_blocks += 1;
        }
    // std::cout << "number of blocks = " << n_blocks << std::endl;
    // std::cout << "number of threads per block = " << run_block_size << std::endl;
    // std::cout << std::endl;

    dim3 grid(n_blocks, 1, 1);
    hipLaunchKernelGGL(
        (gpu_brownian_farfield_grid1_rng_kernel),
        dim3(grid),
        dim3(run_block_size),
        0,
        0,
        n_wave_vectors,
        timestep,
        seed,
        mesh_dim,
        vk,
        T,
        dt,
        d_mesh_inv_Fx,
        d_mesh_inv_Fy,
        d_mesh_inv_Fz);
    
    hipLaunchKernelGGL(
        (gpu_brownian_farfield_grid2_rng_kernel),
        dim3(grid),
        dim3(run_block_size),
        0,
        0,
        n_wave_vectors,
        d_gridk,
        d_ymob,
        d_sf,
        d_mesh_inv_Fx,
        d_mesh_inv_Fy,
        d_mesh_inv_Fz);
    } /* end of gpu_brownian_farfield_grid_rng */

__global__ void gpu_rpy_step_one_kernel(const unsigned int nwork,
                                        const unsigned int offset,
                                        const unsigned int * d_index_array,
                                        BoxDim box,
                                        int3 * d_image,
                                        const Scalar dt,
                                        Scalar4 * d_vel,
                                        Scalar4 * d_pos)
    {
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx < nwork)
        {
        unsigned int group_idx = local_idx + offset;
        unsigned int idx = d_index_array[group_idx];

        Scalar4 pos = d_pos[idx];
        Scalar4 vel = d_vel[idx];
        int3 image = d_image[idx];

        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;

        box.wrap(pos, image);

        d_pos[idx] = pos;
        d_image[idx] = image;
        } // if local_idx < nwork
    
    } /* end of gpu_rpy_step_one_kernel */

void gpu_rpy_step_one(const unsigned int group_size,
                      const unsigned int * d_index_array,
                      const BoxDim& box,
                      int3 * d_image,
                      const Scalar dt,
                      Scalar4 * d_vel,
                      Scalar4 * d_pos,
                      unsigned int block_size,
                      const GPUPartition& gpu_partition)
    {
    // std::cout << "CU: In integrator" << std::endl;
    // std::cout << "Number of GPUs: " << gpu_partition.getNumActiveGPUs() << std::endl;
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);
        unsigned int nwork = range.second - range.first;
        // std::cout << "Nwork = " << nwork << std::endl;

        // std::cout << "number of threads per block = " << block_size << std::endl;
        unsigned int n_blocks = nwork / block_size;
        if (nwork % block_size)
            {
            n_blocks += 1;
            }
        // std::cout << "number of blocks = " << n_blocks << std::endl;

        dim3 grid(n_blocks, 1, 1);
        dim3 threads(block_size, 1, 1);

        hipLaunchKernelGGL(
            (gpu_rpy_step_one_kernel),
            dim3(grid),
            dim3(threads),
            0,
            0,
            nwork,
            range.first,
            d_index_array,
            box,
            d_image,
            dt,
            d_vel,
            d_pos);
        } // for idev
    // std::cout << std::endl;

    } /* end of gpu_rpy_step_one */

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
