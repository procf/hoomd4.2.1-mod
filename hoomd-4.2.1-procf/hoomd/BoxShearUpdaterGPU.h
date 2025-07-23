//~ ########## Created by the Rheoinformatic research group ##########
//~ HOOMD-blue:
// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.
//~
//~ This file:
//~ Written by Mingyang Tan

// ########## Created by Rheoinformatic //~ [RHEOINF] ##########

/*! \file BoxResizeUpdaterGPU.h
    \brief Declares an updater that resizes the simulation box of the system using the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BoxShearUpdater.h"

#ifndef __BOX_SHEAR_UPDATER_GPU_H__
#define __BOX_SHEAR_UPDATER_GPU_H__

namespace hoomd
    {
/// Updates the simulation box over time using the GPU
/** This simple updater gets the box lengths from specified variants and sets
 * those box sizes over time. As an option, particles can be rescaled with the
 * box lengths or left where they are. Note: rescaling particles does not work
 * properly in MPI simulations.
 * \ingroup updaters
 */
class PYBIND11_EXPORT BoxShearUpdaterGPU : public BoxShearUpdater
    {
    public:
    /// Constructor
    BoxShearUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<Trigger> trigger,
                       std::shared_ptr<Variant> vinf,
                       Scalar deltaT,
                       bool flip,
                       std::shared_ptr<ParticleGroup> m_group);

    /// Destructor
    virtual ~BoxShearUpdaterGPU();

    /// Update box interpolation based on provided timestep
    virtual void update(uint64_t timestep) override;

    private:
    /// Autotuner for block size (wrap kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_wrap;
    };

namespace detail
    {
/// Export the BoxResizeUpdaterGPU to python
void export_BoxShearUpdaterGPU(pybind11::module& m);
    }  // end namespace detail
    }  // end namespace hoomd
#endif // __BOX_Shear_UPDATER_GPU_H__