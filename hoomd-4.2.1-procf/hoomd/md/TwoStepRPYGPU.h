//~ ########## Created by the Rheoinformatic research group ##########
//~ HOOMD-blue:
// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.
//~
//~ This file:
//~ Written by Mingyang Tan

/*! \file TwoStepRPYGPU.h
    \brief Declares an integrator that updates the positions of particles while considering the hydrodynamic interactions (HI) with Rotne-Prager-Yamakawa tensor
*/
#include "TwoStepRPY.h"

#ifndef __TWOSTEPRPY_GPU_H__
#define __TWOSTEPRPY_GPU_H__

#ifdef ENABLE_HIP

#if __HIP_PLATFORM_HCC__
#include <hipfft.h>
#elif __HIP_PLATFORM_NVCC__
#include <cufft.h>
typedef cufftComplex hipfftComplex;
typedef cufftHandle hipfftHandle;
#endif

#include "IntegrationMethodTwoStep.h"
#include "NeighborList.h"
#include "hoomd/Variant.h"
#include "hoomd/GlobalArray.h"
#include <cublas_v2.h>
#include "hoomd/Autotuner.h"

#ifdef ENABLE_MPI
#include "CommunicatorGridGPU.h"

#ifndef USE_HOST_DFFT
#include "hoomd/extern/dfftlib/src/dfft_cuda.h"
#else
#include "hoomd/extern/dfftlib/src/dfft_host.h"
#endif
#endif

// #define CHECK_HIPFFT_ERROR(status)                      
//         {                                               
//         handleHIPFFTResult(status, __FILE__, __LINE__); 
//         }

namespace hoomd
    {
namespace md
    {
class PYBIND11_EXPORT TwoStepRPYGPU : public TwoStepRPY
    {
    public:
    //! Constructor
    TwoStepRPYGPU(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<ParticleGroup> group,
                  std::shared_ptr<Variant> T,
                  std::shared_ptr<NeighborList> nlist,
                  Scalar xi,
                  Scalar error);
    virtual ~TwoStepRPYGPU();

    //! Helper function to set up mesh
    void setupMesh();

    //! Helper function to setup FFT and allocate the mesh arrays
    void initializeFFT();

    //! Helper function to allocate the work arrays
    void initializeWorkArray();

    //! Helper function to set up CUBLAS
    void setupCublas();

    //! Helper function to compute wave values and functions
    void computeWaveValue();

    //! Helper function to calculate real-space U = M . F
    void mobilityRealUF(Scalar * force,
                        Scalar * ureal);

    //! Helper function to assign particle coordinates to mesh
    void assignParticleForce(Scalar * force);

    //! Helper function to do forward FFT
    void forwardFFT();

    //! Helper function to multiply Green's function to mesh force
    void meshGreen();

    //! Helper function to do backward FFT
    void backwardFFT();

    //! Helper function to interpolate velocity of mesh to particle
    void interpolateParticleVelocity(Scalar * uwave);

    //! Helper function to wrap up the calculation of U = M . F in wave space
    void mobilityWaveUF(Scalar * force,
                        Scalar * uwave);

    //! Helper function to calculate U = M . F with real-space and wave-space combined
    void mobilityGeneralUF(Scalar * force,
                           Scalar * velocity);

    //! Helper function to calculate slip velocity in real space
    void brownianFarFieldSlipVelocityReal(uint64_t timestep,
                                          Scalar * uslip_real);

    //! Helper function to calculate slip velocity in wave space
    void brownianFarFieldSlipVelocityWave(uint64_t timestep,
                                          Scalar * uslip_wave);

    //! Helper function to calculate the overall slip velocity
    void brownianFarFieldSlipVelocity(uint64_t timestep,
                                      Scalar * uslip);

    //! Helper function to solve U = M . F + M^0.5 . Psi
    void solverMobilityUF(uint64_t timestep,
                          Scalar * force,
                          Scalar * uoe);

    // Perform the first step of integration
    virtual void integrateStepOne(uint64_t timestep);

    // Perform the second step of integration
    virtual void integrateStepTwo(uint64_t timestep);

    private:
    std::shared_ptr<Autotuner<1>> m_tuner_assign;
    std::shared_ptr<Autotuner<1>> m_tuner_green;
    std::shared_ptr<Autotuner<1>> m_tuner_interpolate;
    std::shared_ptr<Autotuner<1>> m_tuner_wavefunc;
    std::shared_ptr<Autotuner<1>> m_tuner_gridrng;
    std::shared_ptr<Autotuner<1>> m_tuner_integrate;

    hipfftHandle m_hipfft_plan;
    bool m_local_fft;
    bool m_cufft_initialized;
    bool m_cuda_dfft_initialized;

    unsigned int m_block_size;

#ifdef ENABLE_MPI
    typedef CommunicatorGridGPU<hipfftComplex> CommunicatorGridGPUComplex;
    std::shared_ptr<CommunicatorGridGPUComplex> m_gpu_grid_comm_forward;
    std::shared_ptr<CommunicatorGridGPUComplex> m_gpu_grid_comm_reverse;

    dfft_plan m_dfft_plan_forward;
    dfft_plan m_dfft_plan_inverse;
#endif
    GlobalArray<hipfftComplex> m_mesh_Fx;
    GlobalArray<hipfftComplex> m_mesh_Fy;
    GlobalArray<hipfftComplex> m_mesh_Fz;
    GlobalArray<hipfftComplex> m_mesh_inv_Fx;
    GlobalArray<hipfftComplex> m_mesh_inv_Fy;
    GlobalArray<hipfftComplex> m_mesh_inv_Fz;

    cublasHandle_t m_blasHandle;

    GlobalArray<Scalar> m_Tm;

    //! Helper function to compute number of ghost cells
    uint3 computeGhostCellNum();
    };

    } // end namespace md
    } // end namespace hoomd


#endif // ENABLE_HIP

#endif // __TWOSTEPRPY_GPU_H__
