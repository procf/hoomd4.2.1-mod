//~ ########## Created by the Rheoinformatic research group ##########
//~ HOOMD-blue:
// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.
//~
//~ This file:
//~ Written by Mingyang Tan

/*! \file TwoStepRPY.h
    \brief Declares an integrator that updates the positions of particles while considering the hydrodynamic interactions (HI) with Rotne-Prager-Yamakawa tensor
*/

#ifndef __TWOSTEPRPY_H__
#define __TWOSTEPRPY_H__

#include "IntegrationMethodTwoStep.h"
#include "NeighborList.h"
#include "hoomd/Variant.h"
#include "hoomd/GlobalArray.h"
#include <unordered_map>
#include <utility>
#include <set>
#include <functional>

#ifdef ENABLE_MPI
#include "CommunicatorGrid.h"
#include "hoomd/extern/dfftlib/src/dfft_host.h"
#endif

#include "hoomd/extern/kiss_fftnd.h"

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {

struct PairHash
    {
    std::size_t operator()(const std::pair<unsigned int, unsigned int>& p)
    const
        {
        return std::hash<unsigned int>()(p.first) ^ (std::hash<unsigned int>()(p.second) << 1);
        }
    };


class PYBIND11_EXPORT TwoStepRPY : public IntegrationMethodTwoStep
    {
//~
    public:
    // constructor
    TwoStepRPY(std::shared_ptr<SystemDefinition> sysdef,
               std::shared_ptr<ParticleGroup> group,
               std::shared_ptr<Variant> T,
               std::shared_ptr<NeighborList> nlist, 
               Scalar xi,
               Scalar error);
    
    // Destructor
    virtual ~TwoStepRPY();

    // Set up parameters of the system
    virtual void setParams();

    // Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    // Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    //~
    protected:  
    std::shared_ptr<Variant> m_T;                   // temperature
    std::shared_ptr<NeighborList> m_nlist;          // neighbor list
    
    //~ parameters for Ewald summation 
    Scalar m_xi;                    // Ewald splitting parameter
    Scalar m_error;                 // tolerance of error (for both Ewald and all iterative solvers as default)
    Scalar m_xi2;                   // Ewald splitting parameter square
    Scalar m_pisqrt;                // pi^0.5
    Scalar m_xi_pisqrt_inv;         // xi / pi^0.5
    Scalar m_rcut_ewald;            // cutoff radius for Ewald short-ranged interaction
    Scalar2 m_self_func;            // self function, x for UF, y for OT, and z for ES

    //~ parameters for domain decomposition for Ewald summation in wave space
    uint3 m_global_dim;                 // number of mesh points of global box
    uint3 m_mesh_points;                // number of mesh points of local box (local processor)
    uint3 m_n_ghost_cells;              // number of ghost cells along every axis
    uint3 m_grid_dim;                   // number of mesh points of local box including ghost cells (m_mesh_points + m_n_ghost_cells)
    unsigned int m_NNN;                 // total number of mesh points of global box
    Scalar3 m_h;                        // width between two mesh points
    Scalar m_vk;                        // volume of a grid cell
    Scalar3 m_ghost_width;              // dimensions of the ghost layer
    unsigned int m_ghost_offset;        // offset in mesh due to ghost cells
    unsigned int m_n_cells;             // total number of inner cells
    unsigned int m_n_inner_cells;       // number of inner mesh points without ghost cells

     //~ work arrays for grid calculation
    GlobalArray<Scalar4> m_gridk;                           // wave values (normalzed kx, ky, and kz, k)                [m_n_inner_cells * Scalar4]
    GlobalArray<Scalar3> m_ymob;                            // mobility function ya, yb, and yc in reciprocal space     [m_n_inner_cells * Scalar3]
    GlobalArray<Scalar2> m_sf;                              // shape function for force and force-dipole distribution   [m_n_inner_cells * Scalar2]

    //~ parameters of spectral Ewald
    unsigned int m_P;               // number of supporting mesh
    unsigned int m_radius;          // stencil radius (in units of mesh size)
    Scalar m_eta;                   // Gaussian distribution parameter
    Scalar m_gauss_fac;
    Scalar m_gauss_exp;

    bool m_box_changed;             // true if box has changed since last timestep
    bool m_need_initialize;         // true if mesh grid needs set

    //~ work arrays for Brownian slip velocity
    GlobalArray<Scalar> m_uslip;                // total Brownian slip velocity [group_size * n], n = 3 for F, n = 6 for FT, n = 11 for FTS
    //~ work arrays for Lanczos iteration (M^0.5 \cdot psi = uslip_real)
    unsigned int m_m_lanczos_ff;                 // number of Lanczos iterations
    unsigned int m_mmax;                         // max number of iterations
    GlobalArray<Scalar> m_iter_ff_v;             // [group_size * n], n = 3 for F, n = 6 for FT, n = 11 for FTS
    GlobalArray<Scalar> m_iter_ff_vj;            // [group_size * n], n = 3 for F, n = 6 for FT, n = 11 for FTS
    GlobalArray<Scalar> m_iter_ff_vjm1;          // [group_size * n], n = 3 for F, n = 6 for FT, n = 11 for FTS
    GlobalArray<Scalar> m_iter_ff_V;             // [group_size * n * (mmax + 1)], n = 3 for F, n = 6 for FT, n = 11 for FTS
    GlobalArray<Scalar> m_iter_ff_uold;          // [group_size * n], n = 3 for F, n = 6 for FT, n = 11 for FTS
    GlobalArray<Scalar> m_iter_ff_u;                // Brownian slip velocity in real space [group_size * n], n = 3 for F, n = 6 for FT, n = 11 for FTS
    GlobalArray<Scalar> m_iter_psi;              // Random vector in real space [group_size * n], n = 3 for F, n = 6 for FT, n = 11 for FTS
    GlobalArray<Scalar> m_iter_Mpsi;

    //~ work arrays for Mr \cdot F = Ur and Mw \cdot F = Uw
    GlobalArray<Scalar> m_u_real;       // velocity in real space [n * group_size], n = 3 for F, n = 6 for FT, n = 11 for FTS
    GlobalArray<Scalar> m_u_wave;       // velocity in real space [n * group_size], n = 3 for F, n = 6 for FT, n = 11 for FTS

    GlobalArray<Scalar> m_u_determin;
    GlobalArray<Scalar> m_uslip_wave;

    //~ velocity and force of particles
    GlobalArray<Scalar> m_uoe;          // velocity of particles [n * group_size], n = 3 for F, n = 6 for FT, n = 11 for FTS
    GlobalArray<Scalar> m_fts;          // force of particles [n * group_size], n = 3 for F, n = 6 for FT, n = 11 for FTS

    // helper function to be called when box changes
    void setBoxChange()
        {
        m_box_changed = true;
        }

    // helper function to set up mesh
    virtual void setupMesh();

    // helper function to set up FFT and allocate memory
    virtual void initializeFFT();
    
    // helper function to set up work arrays
    virtual void initializeWorkArray();

    // helper function calculate wave value
    virtual void computeWaveValue();

    void mobilityRealFunc(Scalar r,
                          Scalar& xa, Scalar& ya,
                          Scalar& yb,
                          Scalar& xc, Scalar &yc);

    // helper function to calculate real-space Ur = Mr \cdot F
    virtual void mobilityRealUF(Scalar * force,
                                Scalar * ureal);

    // helper function to assign force moment of particles to mesh
    virtual void assignParticleForce(Scalar * force);

    // helper function to do FFT of mesh force moments
    virtual void forwardFFT();

    // helper function to scale the mesh in Fourier space by Green function
    virtual void meshGreen();

    // helper function to do inverse FFT of mesh velocity moments
    virtual void backwardFFT();

    // helper function to interpolate velocity moments of mesh to particle
    virtual void interpolateParticleVelocity(Scalar * uwave);

    // helper function to calculate wave-space Uw = Mw \cdot F
    void mobilityWaveUF(Scalar * force,
                        Scalar * uwave);

    // helper function to calculate U = M \cdot F
    virtual void mobilityGeneralUF(Scalar * force,
                                   Scalar * velocity);

    // helper function to generate random vector for Brownian far-field real-space calculation
    void brownianFarFieldRealRNG(uint64_t timestep,
                                 Scalar * psi);

    // helper function to use Lanczos algorithm to solve for M^0.5 \cdot \Psi
    void brownianLanczos(Scalar * psi,
                         Scalar * iter_ff_v,
                         Scalar * iter_ff_vj,
                         Scalar * iter_ff_vjm1,
                         Scalar * iter_ff_V,
                         Scalar * iter_ff_uold,
                         Scalar * iter_Mpsi,
                         Scalar * u);

    // helper function to solve the far-field Brownian slip velocity in real space
    virtual void brownianFarFieldSlipVelocityReal(uint64_t timestep,
                                                  Scalar * ureal);

    // helper function to generate random numbers on mesh with proper conjugacy
    void brownianFarFieldGridRNG(uint64_t timestep);

    // helper function to solve the far-field Brownian slip velocity in wave space
    virtual void brownianFarFieldSlipVelocityWave(uint64_t timestep,
                                                  Scalar * uwave);

    // helper function to wrap up the calculation of far-field Brownian slip velocity
    virtual void brownianFarFieldSlipVelocity(uint64_t timestep,
                                      Scalar * uslip);

    // helper function to solve for velocity if no lubrication or dissipative contact force
    virtual void solverMobilityUF(uint64_t timestep,
                                  Scalar * force,
                                  Scalar * uoe);

    // helper function construct mobility matrix
    void mobilityMatrix();

    // helper function to calculate M . F
    void mobilityMatrixUF(const Scalar * fts,
                          Scalar * uoe);

    // helper function to calculate M^0.5 . Psi
    void mobilityMatrixSqrtUF(const Scalar * psi,
                              Scalar * uslip);

    void brownianParticleRNG(uint64_t timestep,
                             Scalar * psi);

    // helper function to solve for velocity with RPY in matrix form
    void solverMobilityMatrixUF(uint64_t timestep,
                                const Scalar * fts,
                                Scalar * uoe);

//~
    private:
    kiss_fftnd_cfg m_kiss_fft = NULL;               // FFT configuration
    kiss_fftnd_cfg m_kiss_ifft = NULL;              // IFFT configuration
#ifdef ENABLE_MPI
    dfft_plan m_dfft_plan_forward;      // distributed FFT
    dfft_plan m_dfft_plan_inverse;      // distributed IFFT
    std::unique_ptr<CommunicatorGrid<kiss_fft_cpx>> m_grid_comm_forward;
    std::unique_ptr<CommunicatorGrid<kiss_fft_cpx>> m_grid_comm_reverse;
#endif
    bool m_kiss_fft_initialized;
    bool m_dfft_initialized;

    GlobalArray<Scalar> m_kron;
    GlobalArray<Scalar> m_mobmat_real;
    GlobalArray<Scalar> m_mobmat_wave;
    GlobalArray<Scalar> m_mobmat;
    GlobalArray<Scalar> m_mobmat_scratch;
    GlobalArray<Scalar> m_mobmatSqrt;


    //~ force/force-dipole of mesh in real space (padded with offset)
    GlobalArray<kiss_fft_cpx> m_mesh_Fx;                    // mesh force in x direction            [(m_n_cells + m_ghost_offset) * kiss_fft_cpx]
    GlobalArray<kiss_fft_cpx> m_mesh_Fy;                    // mesh force in y direction            [(m_n_cells + m_ghost_offset) * kiss_fft_cpx]
    GlobalArray<kiss_fft_cpx> m_mesh_Fz;                    // mesh force in z direction            [(m_n_cells + m_ghost_offset) * kiss_fft_cpx]
    
    //~ force/force-dipole of mesh in FFT space
    GlobalArray<kiss_fft_cpx> m_mesh_fft_Fx;                // FFTed mesh force in x direction      [m_n_inner_cells * kiss_fft_cpx]
    GlobalArray<kiss_fft_cpx> m_mesh_fft_Fy;                // FFTed mesh force in y direction      [m_n_inner_cells * kiss_fft_cpx]
    GlobalArray<kiss_fft_cpx> m_mesh_fft_Fz;                // FFTed mesh force in z direction      [m_n_inner_cells * kiss_fft_cpx]
    

    //~ scaled force/force-dipole of mesh in real space (padded with offset)
    GlobalArray<kiss_fft_cpx> m_mesh_inv_Fx;                // IFFTed mesh force in x direction     [(m_n_cells + m_ghost_offset) * kiss_fft_cpx]
    GlobalArray<kiss_fft_cpx> m_mesh_inv_Fy;                // IFFTed mesh force in y direction     [(m_n_cells + m_ghost_offset) * kiss_fft_cpx]
    GlobalArray<kiss_fft_cpx> m_mesh_inv_Fz;                // IFFTed mesh force in z direction     [(m_n_cells + m_ghost_offset) * kiss_fft_cpx]
    
    // helper function to compute number of ghost cells
    uint3 computeGhostCellNum();

    }; /* End of TwoStepRPY class */ 

    } // end namespace md
    } // end namespace hoomd

#endif
