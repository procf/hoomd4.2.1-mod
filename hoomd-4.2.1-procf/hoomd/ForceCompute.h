// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// ########## Modified by Rheoinformatic //~ [RHEOINF] ##########

#include "Compute.h"
#include "GlobalArray.h"
#include "Index1D.h"
#include "ParticleGroup.h"
#include "PythonLocalDataAccess.h"

#ifdef ENABLE_HIP
#include "ParticleData.cuh"
#endif

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>

/*! \file ForceCompute.h
    \brief Declares the ForceCompute class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __FORCECOMPUTE_H__
#define __FORCECOMPUTE_H__

namespace hoomd
    {
//! Handy structure for passing the force arrays around
/*! \c fx, \c fy, \c fz have length equal to the number of particles and store the x,y,z
    components of the force on that particle. \a pe is also included as the potential energy
    for each particle, if it can be defined for the force. \a virial is the per particle virial.

    The per particle potential energy is defined such that \f$ \sum_i^N \mathrm{pe}_i =
   V_{\mathrm{total}} \f$

    The per particle virial is a upper triangular 3x3 matrix that is defined such
    that
    \f$ \sum_k^N \left(\mathrm{virial}_{ij}\right)_k = \sum_k^N \sum_{l>k} \frac{1}{2} \left(
   \vec{f}_{kl,i} \vec{r}_{kl,j} \right) \f$

    \ingroup data_structs
*/

class PYBIND11_EXPORT ForceCompute : public Compute
    {
    public:
    //! Constructs the compute
    ForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~ForceCompute();

    //! Store the timestep size
    virtual void setDeltaT(Scalar dt)
        {
        m_deltaT = dt;
        }

    //~ add shear rate [RHEOINF]
    virtual void setSR(Scalar shear_rate)
        {
        m_SR = shear_rate;
        }
    //~

#ifdef ENABLE_MPI
    //! Pre-compute the forces
    /*! This method is called in MPI simulations BEFORE the particles are migrated
     * and can be used to overlap computation with communication
     */
    virtual void preCompute(uint64_t timestep) { }
#endif

    //! Computes the forces
    virtual void compute(uint64_t timestep);

    //! Total the potential energy
    Scalar calcEnergySum();

    //! Sum the potential energy of a group
    Scalar calcEnergyGroup(std::shared_ptr<ParticleGroup> group);

    //! Sum the all forces for a group
    vec3<double> calcForceGroup(std::shared_ptr<ParticleGroup> group);

    //! Sum all virial terms for a group
    std::vector<Scalar> calcVirialGroup(std::shared_ptr<ParticleGroup> group);

    //~ Sum all virial_ind terms for a group [RHEOINF]
    std::vector<Scalar> calcVirialIndGroup(std::shared_ptr<ParticleGroup> group);
    //~

    /** Get per particle energies

        @returns a Numpy array with per particle energies in increasing tag order.
    */
    pybind11::object getEnergiesPython();

    /** Get per particle forces

        @returns a Numpy array with per particle forces in increasing tag order.
    */
    pybind11::object getForcesPython();

    /** Get per particle torques

        @returns a Numpy array with per particle torques in increasing tag order.
    */
    pybind11::object getTorquesPython();

    /** Get per particle virials

        @returns a Numpy array with per particle virials in increasing tag order.
    */
    pybind11::object getVirialsPython();

    //~ add virial_ind [RHEOINF] 
    /** Get per particle virial_inds

        @returns a Numpy array with per particle virial_inds in increasing tag order.
    */
    pybind11::object getVirialsIndPython();
    //~

    //! Easy access to the torque on a single particle
    Scalar4 getTorque(unsigned int tag);

    //! Easy access to the force on a single particle
    Scalar3 getForce(unsigned int tag);

    //! Easy access to the virial on a single particle
    Scalar getVirial(unsigned int tag, unsigned int component);

    //~! Easy access to the virial on a single particle [RHEOINF]
    Scalar getVirialInd(unsigned int tag, unsigned int component);
    //~

    //! Easy access to the energy on a single particle
    Scalar getEnergy(unsigned int tag);

    //! Get the array of computed forces
    const GlobalArray<Scalar4>& getForceArray() const
        {
        return m_force;
        }

    //! Get the array of computed virials
    const GlobalArray<Scalar>& getVirialArray() const
        {
        return m_virial;
        }

    //~! Get the array of computed virial_inds [RHEOINF] 
    const GlobalArray<Scalar>& getVirialIndArray() const
        {
        return m_virial_ind;
        }
    //~

    //! Get the array of computed torques
    const GlobalArray<Scalar4>& getTorqueArray() const
        {
        return m_torque;
        }

    //! Get the contribution to the external virial
    virtual Scalar getExternalVirial(unsigned int dir)
        {
        assert(dir < 6);
        return m_external_virial[dir];
        }

    //! Get the contribution to the external potential energy
    virtual Scalar getExternalEnergy()
        {
        return m_external_energy;
        }

#ifdef ENABLE_MPI
    //! Get requested ghost communication flags
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        // by default, only request positions
        CommFlags flags(0);
        flags[comm_flag::position] = 1;
        flags[comm_flag::net_force] = 1; // only used if constraints are present
        return flags;
        }
#endif

    //! Returns true if this ForceCompute requires anisotropic integration
    virtual bool isAnisotropic()
        {
        // by default, only translational degrees of freedom are integrated
        return false;
        }

    bool getLocalBuffersWriteable() const
        {
        return m_buffers_writeable;
        }

    protected:
    bool m_particles_sorted; //!< Flag set to true when particles are resorted in memory

    //! Helper function called when particles are sorted
    /*! setParticlesSorted() is passed as a slot to the particle sort signal.
        It is used to flag \c m_particles_sorted so that a second call to compute
        with the same timestep can properly recalculate the forces, which are stored
        by index.
    */
    void setParticlesSorted()
        {
        m_particles_sorted = true;
        }

    //! Reallocate internal arrays
    void reallocate();

    //! Update GPU memory hints
    void updateGPUAdvice();

    Scalar m_deltaT; //!< timestep size (required for some types of non-conservative forces)

    Scalar m_SR; //!<~ shear rate [RHEOINF]

    GlobalArray<Scalar4> m_force; //!< m_force.x,m_force.y,m_force.z are the x,y,z components of the
                                  //!< force, m_force.u is the PE

    /*! per-particle virial, a 2D array with width=number
        of particles and height=6. The elements of the (upper triangular)
        3x3 virial matrix \f$ \left(\mathrm{virial}_{ij}\right),k \f$ for
        particle \f$k\f$ are stored in the rows and are indexed in the
        order xx, xy, xz, yy, yz, zz
     */
    GlobalArray<Scalar> m_virial;
    size_t m_virial_pitch;         //!< The pitch of the 2D virial array
    GlobalArray<Scalar> m_virial_ind; //~ add virial_ind [RHEOINF]
    size_t m_virial_ind_pitch; //~!< The pitch of the 2D virial_ind array [RHEOINF]
    GlobalArray<Scalar4> m_torque; //!< per-particle torque

    Scalar m_external_virial[6]; //!< Stores external contribution to virial
    Scalar m_external_energy;    //!< Stores external contribution to potential energy

    /// Store the particle data flags used during the last computation
    PDataFlags m_computed_flags;

    // whether the local force buffers exposed by this class should be read-only
    bool m_buffers_writeable;

    //! Actually perform the computation of the forces
    /*! This is pure virtual here. Sub-classes must implement this function. It will be called by
        the base class compute() when the forces need to be computed.
        \param timestep Current time step
    */
    virtual void computeForces(uint64_t timestep) { }
    };

/** Make the local particle data available to python via zero-copy access
 *
 * */
template<class Output>
class PYBIND11_EXPORT LocalForceComputeData : public GhostLocalDataAccess<Output, ForceCompute>
    {
    public:
    LocalForceComputeData(ForceCompute& data, ParticleData& pdata)
        : GhostLocalDataAccess<Output, ForceCompute>(data,
                                                     pdata.getN(),
                                                     pdata.getNGhosts(),
                                                     pdata.getNGlobal()),
          m_force_handle(), m_torque_handle(), m_virial_handle(),
          m_virial_ind_handle(), //~ add virial_ind [RHEOINF]
          m_virial_pitch(data.getVirialArray().getPitch()), 
          m_virial_ind_pitch(data.getVirialIndArray().getPitch()), //~ add virial_ind [RHEOINF]
          m_buffers_writeable(data.getLocalBuffersWriteable())
        {
        }

    virtual ~LocalForceComputeData() = default;

    Output getForce(GhostDataFlag flag)
        {
        return this->template getLocalBuffer<Scalar4, Scalar>(m_force_handle,
                                                              &ForceCompute::getForceArray,
                                                              flag,
                                                              m_buffers_writeable,
                                                              3);
        }

    Output getPotentialEnergy(GhostDataFlag flag)
        {
        return this->template getLocalBuffer<Scalar4, Scalar>(m_force_handle,
                                                              &ForceCompute::getForceArray,
                                                              flag,
                                                              m_buffers_writeable,
                                                              0,
                                                              3 * sizeof(Scalar));
        }

    Output getTorque(GhostDataFlag flag)
        {
        return this->template getLocalBuffer<Scalar4, Scalar>(m_torque_handle,
                                                              &ForceCompute::getTorqueArray,
                                                              flag,
                                                              m_buffers_writeable,
                                                              3);
        }

    Output getVirial(GhostDataFlag flag)
        {
        // we order the strides as (1, m_virial_pitch) because we need to expose
        // the array as having shape (N, 6) even though the underlying data has
        // shape (6, m_virial_pitch)
        return this->template getLocalBuffer<Scalar, Scalar>(
            m_virial_handle,
            &ForceCompute::getVirialArray,
            flag,
            m_buffers_writeable,
            6,
            0,
            std::vector<size_t>(
                {sizeof(Scalar), static_cast<size_t>(m_virial_pitch * sizeof(Scalar))}));
        }

    //~ add virial_ind [RHEOINF] 
    Output getVirialInd(GhostDataFlag flag)
        {
        // we order the strides as (1, m_virial_ind_pitch) because we need to expose
        // the array as having shape (N, 5) even though the underlying data has
        // shape (5, m_virial_ind_pitch)
        return this->template getLocalBuffer<Scalar, Scalar>(
            m_virial_ind_handle,
            &ForceCompute::getVirialIndArray,
            flag,
            m_buffers_writeable,
            5,
            0,
            std::vector<size_t>(
                {sizeof(Scalar), static_cast<size_t>(m_virial_ind_pitch * sizeof(Scalar))}));
        }
    //~

    protected:
    void clear()
        {
        m_force_handle.reset(nullptr);
        m_torque_handle.reset(nullptr);
        m_virial_handle.reset(nullptr);
        m_virial_ind_handle.reset(nullptr); //~ add virial_ind [RHEOINF]
        m_rtag_handle.reset(nullptr);
        }

    private:
    std::unique_ptr<ArrayHandle<Scalar4>> m_force_handle;
    std::unique_ptr<ArrayHandle<Scalar4>> m_torque_handle;
    std::unique_ptr<ArrayHandle<Scalar>> m_virial_handle;
    std::unique_ptr<ArrayHandle<Scalar>> m_virial_ind_handle; //~ add virial_ind [RHEOINF]
    std::unique_ptr<ArrayHandle<unsigned int>> m_rtag_handle;
    size_t m_virial_pitch;
    size_t m_virial_ind_pitch; //~ add virial_ind [RHEOINF]
    bool m_buffers_writeable;
    };

namespace detail
    {
//! Exports the ForceCompute class to python
#ifndef __HIPCC__
void export_ForceCompute(pybind11::module& m);
#endif

template<class Output> void export_LocalForceComputeData(pybind11::module& m, std::string name)
    {
    pybind11::class_<LocalForceComputeData<Output>, std::shared_ptr<LocalForceComputeData<Output>>>(
        m,
        name.c_str())
        .def(pybind11::init<ForceCompute&, ParticleData&>())
        .def("getForce", &LocalForceComputeData<Output>::getForce)
        .def("getPotentialEnergy", &LocalForceComputeData<Output>::getPotentialEnergy)
        .def("getTorque", &LocalForceComputeData<Output>::getTorque)
        .def("getVirial", &LocalForceComputeData<Output>::getVirial)
        .def("getVirialInd", &LocalForceComputeData<Output>::getVirialInd) //~ add virial_ind [RHEOINF]
        .def("enter", &LocalForceComputeData<Output>::enter)
        .def("exit", &LocalForceComputeData<Output>::exit);
    };
    } // end namespace detail

    } // end namespace hoomd

#endif // __FORCECOMPUTE_H__
