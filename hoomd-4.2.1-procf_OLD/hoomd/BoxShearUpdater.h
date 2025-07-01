//~ ########## Created by the Rheoinformatic research group ##########
//~ HOOMD-blue:
// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.
//~
//~ This file:
//~ Written by Dr. Deepak Mangal 
//~ Documentation by Rob Campbell (2024)

// ########## Created by Rheoinformatic //~ [RHEOINF] ##########


/*! \file BoxResizeUpdater.h
    \brief Declares an updater that resizes the simulation box of the system
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BoxDim.h"
#include "ParticleGroup.h"
#include "Updater.h"
#include "Variant.h"

#include <memory>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

#ifndef __BOXSHEARUPDATER_H__
#define __BOXSHEARUPDATER_H__

namespace hoomd
    {
/// Updates the simulation box over time
/** This simple updater gets the box lengths from specified variants and sets
 * those box sizes over time. As an option, particles can be rescaled with the
 * box lengths or left where they are. Note: rescaling particles does not work
 * properly in MPI simulations.
 * \ingroup updaters
 */
class PYBIND11_EXPORT BoxShearUpdater : public Updater
    {
    public:
    /// Constructor
    BoxShearUpdater(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<Trigger> trigger,
                     std::shared_ptr<Variant> vinf,
                     Scalar deltaT,
                     bool flip,
                     std::shared_ptr<ParticleGroup> group);

    /// Destructor
    virtual ~BoxShearUpdater();

    virtual void setdeltaT(Scalar deltaT){m_deltaT = deltaT;}
    virtual Scalar getdeltaT() const{return m_deltaT;}

    virtual void setFlip(bool flip){m_flip = flip;}
    virtual bool getFlip() const{return m_flip;}

    /// Set the variant for interpolation
    void setVinf(std::shared_ptr<Variant> vinf)
        {
        m_vinf = vinf;
        }

    /// Get the variant for interpolation
    std::shared_ptr<Variant> getVinf()
        {
        return m_vinf;
        }

    /// Update box interpolation based on provided timestep
    virtual void update(uint64_t timestep);

    protected:
#ifdef ENABLE_MPI
    std::shared_ptr<Communicator> m_comm;
#endif

    private:
    std::shared_ptr<Variant> m_vinf;     //!< Variant that interpolates between boxes
    Scalar m_deltaT;
    bool m_flip;
    std::shared_ptr<ParticleGroup> m_group;
    };

namespace detail
    {
/// Export the BoxResizeUpdater to python
void export_BoxShearUpdater(pybind11::module& m);
    } // end namespace detail
    } // end namespace hoomd
#endif
