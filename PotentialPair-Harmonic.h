// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __POTENTIAL_PAIR_H__
#define __POTENTIAL_PAIR_H__

#include "Lifetime.h"

#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "NeighborList.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/managed_allocator.h"
#include "hoomd/md/EvaluatorPairLJ.h"

// paniz_write
#include "hoomd/HOOMDMPI.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
#include <cmath> 

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

/*! \file PotentialPair.h
    \brief Defines the template class for standard pair potentials
    \details The heart of the code that computes pair potentials is in this file.
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {
//! Template class for computing pair potentials
/*! <b>Overview:</b>
    PotentialPair computes standard pair potentials (and forces) between all particle pairs in the
   simulation. It employs the use of a neighbor list to limit the number of computations done to
   only those particles with the cutoff radius of each other. The computation of the actual V(r) is
   not performed directly by this class, but by an evaluator class (e.g. EvaluatorPairLJ) which is
   passed in as a template parameter so the computations are performed as efficiently as possible.

    PotentialPair handles most of the gory internal details common to all standard pair potentials.
     - A cutoff radius to be specified per particle type pair
     - The energy can be globally shifted to 0 at the cutoff
     - XPLOR switching can be enabled
     - Per type pair parameters are stored and a set method is provided
     - And all the details about looping through the particles, computing dr, computing the virial,
   etc. are handled

    A note on the design of XPLOR switching:
    We need to be able to handle smooth XPLOR switching in systems of mixed LJ/WCA particles. There
   are three modes to enable all of the various use-cases:
     - Mode 1: No shifting. All pair potentials are computed as is and not shifted to 0 at the
   cutoff.
     - Mode 2: Shift everything. All pair potentials (no matter what type pair) are shifted so they
   are 0 at the cutoff
     - Mode 3: XPLOR switching enabled. A r_on value is specified per type pair. When r_on is less
   than r_cut, normal XPLOR switching will be applied to the unshifted potential. When r_on is
   greater than r_cut, the energy will be shifted. In this manner, a valid r_on value can be given
   for the LJ interactions and r_on > r_cut can be set for WCA (which will then be shifted).

    XPLOR switching gets significantly more complicated for all pair potentials when shifted
   potentials are used. Thus, the combination of XPLOR switching + shifted potentials will not be
   supported to avoid slowing down the calculation for everyone.

    <b>Implementation details</b>

    rcutsq, ronsq, and the params are stored per particle type pair. It wastes a little bit of
   space, but benchmarks show that storing the symmetric type pairs and indexing with Index2D is
   faster than not storing redundant pairs and indexing with Index2DUpperTriangular. All of these
   values are stored in GlobalArray for easy access on the GPU by a derived class. The type of the
   parameters is defined by \a param_type in the potential evaluator class passed in. See the
   appropriate documentation for the evaluator for the definition of each element of the parameters.
*/
template<class evaluator> class PotentialPair : public ForceCompute
    {
    public:
    //! Param type from evaluator
    typedef typename evaluator::param_type param_type;

    //! Construct the pair potential
    PotentialPair(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<NeighborList> nlist, Scalar K = 0.0);
    //! Destructor
    virtual ~PotentialPair();

    //! Set and get the pair parameters for a single type pair
    virtual void setParams(unsigned int typ1, unsigned int typ2, const param_type& param);
    virtual void setParamsPython(pybind11::tuple typ, pybind11::dict params);
    /// Get params for a single type pair using a tuple of strings
    virtual pybind11::dict getParams(pybind11::tuple typ);
    //! Set the rcut for a single type pair
    virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);
    /// Get the r_cut for a single type pair
    Scalar getRCut(pybind11::tuple types);
    /// Set the rcut for a single type pair using a tuple of strings
    virtual void setRCutPython(pybind11::tuple types, Scalar r_cut);
    //! Set ron for a single type pair
    virtual void setRon(unsigned int typ1, unsigned int typ2, Scalar ron);
    /// Get the r_on for a single type pair
    Scalar getROn(pybind11::tuple types);
    /// Set the r_on for a single type using a tuple of string
    virtual void setROnPython(pybind11::tuple types, Scalar r_on);
    /// Validate that types are within Ntypes
    void validateTypes(unsigned int typ1, unsigned int typ2, std::string action);

    void saveAngleMap(const std::map<std::pair<int, int>, std::vector<double>>& angle_map, const std::string& filename);
    void loadAngleMap(std::map<std::pair<int, int>, std::vector<double>>& angle_map, const std::string& filename);

    //! Shifting modes that can be applied to the energy
    enum energyShiftMode
        {
        no_shift = 0,
        shift,
        xplor
        };

    //! Set the mode to use for shifting the energy
    void setShiftMode(energyShiftMode mode)
        {
        m_shift_mode = mode;
        }

    void setShiftModePython(std::string mode)
        {
        if (mode == "none")
            {
            m_shift_mode = no_shift;
            }
        else if (mode == "shift")
            {
            m_shift_mode = shift;
            }
        else if (mode == "xplor")
            {
            m_shift_mode = xplor;
            }
        else
            {
            throw std::runtime_error("Invalid energy shift mode.");
            }
        }

    /// Get the mode used for the energy shifting
    std::string getShiftMode()
        {
        switch (m_shift_mode)
            {
        case no_shift:
            return "none";
        case shift:
            return "shift";
        case xplor:
            return "xplor";
        default:
            throw std::runtime_error("Error setting shift mode.");
            }
        }

    virtual void notifyDetach()
        {
        if (m_attached)
            {
            m_nlist->removeRCutMatrix(m_r_cut_nlist);
            }
        m_attached = false;
        }

    //! Set whether analytical tail correction is enabled
    void setTailCorrectionEnabled(bool enable)
        {
        m_tail_correction_enabled = enable;
        }

    bool getTailCorrectionEnabled()
        {
        return m_tail_correction_enabled;
        }
    //S
    void setK(Scalar K) {
        m_K = K;
    }
    Scalar getK() const {
        return m_K;
    }
    std::shared_ptr<Lifetime> LTIME;
    std::map<unsigned int, Scalar> angle_map;
    std::map<unsigned int, Scalar> angle_map_temp2;
    //F
 #ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    virtual CommFlags getRequestedCommFlags(uint64_t timestep);
 #endif

    //! Calculates the energy between two lists of particles.
    template<class InputIterator>
    void computeEnergyBetweenSets(InputIterator first1,
                                  InputIterator last1,
                                  InputIterator first2,
                                  InputIterator last2,
                                  Scalar& energy);
    //! Calculates the energy between two lists of particles.
    Scalar
    computeEnergyBetweenSetsPythonList(pybind11::array_t<int, pybind11::array::c_style> tags1,
                                       pybind11::array_t<int, pybind11::array::c_style> tags2);

    std::vector<std::string> getTypeShapeMapping() const
        {
        std::vector<std::string> type_shape_mapping(m_pdata->getNTypes());
        for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
            {
            evaluator eval(Scalar(0.0), Scalar(0.0), this->m_params[m_typpair_idx(i, i)]);
            type_shape_mapping[i] = eval.getShapeSpec();
            }
        return type_shape_mapping;
        }

    /// Reset stats counters for children objects
    virtual void resetStats()
        {
        m_nlist->resetStats();
        }

    /// Start autotuning kernel launch parameters
    virtual void startAutotuning();

    /// Check if autotuning is complete.
    virtual bool isAutotuningComplete();

    protected:
    std::shared_ptr<NeighborList> m_nlist; //!< The neighborlist to use for the computation
    energyShiftMode m_shift_mode; //!< Store the mode with which to handle the energy shift at r_cut
    Scalar m_K;
    Index2D m_typpair_idx;        //!< Helper class for indexing per type pair arrays
    GlobalArray<Scalar> m_rcutsq; //!< Cutoff radius squared per type pair
    GlobalArray<Scalar> m_ronsq;  //!< ron squared per type pair

    /// Per type pair potential parameters
    std::vector<param_type, hoomd::detail::managed_allocator<param_type>> m_params;

    /// Track whether we have attached to the Simulation object
    bool m_attached = true;

    bool m_tail_correction_enabled = false;
    /// r_cut (not squared) given to the neighbor list
    std::shared_ptr<GlobalArray<Scalar>> m_r_cut_nlist;

    /// Keep track of number of each type of particle
    std::vector<unsigned int> m_num_particles_by_type;

#ifdef ENABLE_MPI
    /// The system's communicator.
    std::shared_ptr<Communicator> m_comm;
#endif

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! Compute the long-range corrections to energy and pressure to account for truncating the pair
    //! potentials
    virtual void computeTailCorrection()
        {
        // early exit if tail correction not enabled
        if (!m_tail_correction_enabled)
            {
            return;
            }

        // tail correction only valid with no potential shifting, throw error if shift and tail
        // correction enabled
        if (m_shift_mode != no_shift)
            {
            throw std::runtime_error(
                "Pair potential shift mode must be \"none\" to apply tail corrections.");
            }

        // Only compute pressure correction if we need pressure on this timestep
        PDataFlags flags = this->m_pdata->getFlags();
        bool compute_virial = flags[pdata_flag::pressure_tensor];

        BoxDim box = m_pdata->getGlobalBox();
        int dimension = m_sysdef->getNDimensions();
        bool is_two_dimensions = dimension == 2;
        Scalar volume = box.getVolume(is_two_dimensions);
        ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);

        // compute energy correction and store in m_external_energy; this is done on every step
        m_external_energy = Scalar(0.0);
        const unsigned int my_rank = m_exec_conf->getRank();
        if (my_rank == 0)
            {
            for (unsigned int type_i = 0; type_i < m_pdata->getNTypes(); type_i++)
                {
                for (unsigned int type_j = 0; type_j < m_pdata->getNTypes(); type_j++)
                    {
                    // rho is the number density
                    Scalar rho_j = m_num_particles_by_type[type_j] / volume;
                    evaluator eval(Scalar(0.0),
                                   h_rcutsq.data[m_typpair_idx(type_i, type_j)],
                                   m_params[m_typpair_idx(type_i, type_j)]);
                    m_external_energy += Scalar(2.0) * m_num_particles_by_type[type_i] * M_PI
                                         * rho_j * eval.evalEnergyLRCIntegral();
                    }
                }
            }

        // compute the virial
        if (compute_virial)
            {
            // zero out the entire external virial tensor
            for (unsigned int i = 0; i < 6; i++)
                {
                m_external_virial[i] = Scalar(0.0);
                }

            if (my_rank == 0)
                {
                for (unsigned int type_i = 0; type_i < m_pdata->getNTypes(); type_i++)
                    {
                    // rho is the number density
                    Scalar rho_i = m_num_particles_by_type[type_i] / volume;
                    for (unsigned int type_j = 0; type_j < m_pdata->getNTypes(); type_j++)
                        {
                        Scalar rho_j = m_num_particles_by_type[type_j] / volume;
                        evaluator eval(Scalar(0.0),
                                       h_rcutsq.data[m_typpair_idx(type_i, type_j)],
                                       m_params[m_typpair_idx(type_i, type_j)]);
                        // The pressure LRC, where
                        // P = \frac{2 \cdot K_{trans} + W}{D \cdot  V}
                        Scalar delta_pressure = Scalar(4.0) / Scalar(6.0) * rho_i * rho_j * M_PI
                                                * eval.evalPressureLRCIntegral();
                        // \Delta W = \Delta P (D \cdot V)
                        // We will assume that the contribution to pressure is equal
                        // in x, y, and z, so we will add 1/3 \Delta W on the diagonal
                        // Note that 0, 3, and 5 are the indices of m_external_virial corresponding
                        // to the diagonal elements
                        Scalar delta_virial = dimension * volume * delta_pressure / Scalar(3.0);
                        m_external_virial[0] += delta_virial;
                        m_external_virial[3] += delta_virial;
                        m_external_virial[5] += delta_virial;
                        }
                    }
                }

            } // end if (compute_virial)

        } // end void computeTailCorrection()
        // To save and load the angle map
        //S
        void saveAngleMap(const std::map<unsigned int, Scalar>& angle_map, const std::string& filename) {
            std::ofstream outFile(filename, std::ios::out | std::ios::binary);
            if (!outFile) {
                std::cerr << "Error opening file for writing: " << filename << std::endl;
                return;
            }

            for (const auto& entry : angle_map) {
                // Write the key (unsigned int)
                outFile.write(reinterpret_cast<const char*>(&entry.first), sizeof(unsigned int));

                // Write the value (Scalar)
                outFile.write(reinterpret_cast<const char*>(&entry.second), sizeof(Scalar));
            }

            outFile.close();
        }

        void loadAngleMap(std::map<unsigned int, Scalar>& angle_map, const std::string& filename) {
            std::ifstream inFile(filename, std::ios::in | std::ios::binary);
            if (!inFile) {
                std::cerr << "Error opening file for reading: " << filename << std::endl;
                return;
            }

            while (inFile.peek() != EOF) {
                unsigned int key;
                Scalar value;

                // Read the key (unsigned int)
                inFile.read(reinterpret_cast<char*>(&key), sizeof(unsigned int));

                // Read the value (Scalar)
                inFile.read(reinterpret_cast<char*>(&value), sizeof(Scalar));

                // Insert into the map
                angle_map[key] = value;
            }

            inFile.close();
        }
        //F
    }; // end class PotentialPair

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
*/
template<class evaluator>
PotentialPair<evaluator>::PotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                                        std::shared_ptr<NeighborList> nlist,
                                        Scalar K)
    : ForceCompute(sysdef), m_nlist(nlist), m_shift_mode(no_shift),m_K(K),
      m_typpair_idx(m_pdata->getNTypes())
    {
    m_exec_conf->msg->notice(5) << "Constructing PotentialPair<" << evaluator::getName() << ">"
                                << std::endl;

    assert(m_pdata);
    assert(m_nlist);

    //~ add bond_calc flag [PROCF2023]
    if(m_K != 0.0)
	{
        LTIME = std::shared_ptr<Lifetime>(new Lifetime(sysdef));
	}


    //~

    GlobalArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_rcutsq.swap(rcutsq);
    GlobalArray<Scalar> ronsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_ronsq.swap(ronsq);
    m_params = std::vector<param_type, hoomd::detail::managed_allocator<param_type>>(
        m_typpair_idx.getNumElements(),
        param_type(),
        hoomd::detail::managed_allocator<param_type>(m_exec_conf->isCUDAEnabled()));

    m_r_cut_nlist
        = std::make_shared<GlobalArray<Scalar>>(m_typpair_idx.getNumElements(), m_exec_conf);
    nlist->addRCutMatrix(m_r_cut_nlist);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_pdata->getExecConf()->isCUDAEnabled())
        {
        // m_params is _always_ in unified memory, so memadvise and prefetch
        cudaMemAdvise(m_params.data(),
                      m_params.size() * sizeof(param_type),
                      cudaMemAdviseSetReadMostly,
                      0);
        auto& gpu_map = m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemPrefetchAsync(m_params.data(),
                                 sizeof(param_type) * m_params.size(),
                                 gpu_map[idev]);
            }

        // m_rcutsq and m_ronsq only in unified memory if allConcurrentManagedAccess
        if (m_exec_conf->allConcurrentManagedAccess())
            {
            cudaMemAdvise(m_rcutsq.get(),
                          m_rcutsq.getNumElements() * sizeof(Scalar),
                          cudaMemAdviseSetReadMostly,
                          0);
            cudaMemAdvise(m_ronsq.get(),
                          m_ronsq.getNumElements() * sizeof(Scalar),
                          cudaMemAdviseSetReadMostly,
                          0);
            for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
                {
                // prefetch data on all GPUs
                cudaMemPrefetchAsync(m_rcutsq.get(),
                                     sizeof(Scalar) * m_rcutsq.getNumElements(),
                                     gpu_map[idev]);
                cudaMemPrefetchAsync(m_ronsq.get(),
                                     sizeof(Scalar) * m_ronsq.getNumElements(),
                                     gpu_map[idev]);
                }
            }
        }
#endif

    // get number of each type of particle, needed for energy and pressure correction
    m_num_particles_by_type.resize(m_pdata->getNTypes());
    std::fill(m_num_particles_by_type.begin(), m_num_particles_by_type.end(), 0);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int typeid_i = __scalar_as_int(h_postype.data[i].w);
        m_num_particles_by_type[typeid_i] += 1;
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // reduce number of each type of particle on all processors
        MPI_Allreduce(MPI_IN_PLACE,
                      m_num_particles_by_type.data(),
                      m_pdata->getNTypes(),
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm_weak = m_sysdef->getCommunicator();
        assert(comm_weak.lock());
        m_comm = comm_weak.lock();
        }
#endif
    }

template<class evaluator> PotentialPair<evaluator>::~PotentialPair()
    {
    m_exec_conf->msg->notice(5) << "Destroying PotentialPair<" << evaluator::getName() << ">"
                                << std::endl;

    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param param Parameter to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
template<class evaluator>
void PotentialPair<evaluator>::setParams(unsigned int typ1,
                                         unsigned int typ2,
                                         const param_type& param)
    {
    validateTypes(typ1, typ2, "setting params");
    m_params[m_typpair_idx(typ1, typ2)] = param;
    m_params[m_typpair_idx(typ2, typ1)] = param;
    }

template<class evaluator>
void PotentialPair<evaluator>::setParamsPython(pybind11::tuple typ, pybind11::dict params)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    setParams(typ1, typ2, param_type(params, m_exec_conf->isCUDAEnabled()));
    }

template<class evaluator> pybind11::dict PotentialPair<evaluator>::getParams(pybind11::tuple typ)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    validateTypes(typ1, typ2, "setting params");

    return m_params[m_typpair_idx(typ1, typ2)].asDict();
    }

template<class evaluator>
void PotentialPair<evaluator>::validateTypes(unsigned int typ1,
                                             unsigned int typ2,
                                             std::string action)
    {
    auto n_types = this->m_pdata->getNTypes();
    if (typ1 >= n_types || typ2 >= n_types)
        {
        throw std::runtime_error("Error in" + action + " for pair potential. Invalid type");
        }
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param rcut Cutoff radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
template<class evaluator>
void PotentialPair<evaluator>::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    validateTypes(typ1, typ2, "setting r_cut");
        {
        // store r_cut**2 for use internally
        ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::readwrite);
        h_rcutsq.data[m_typpair_idx(typ1, typ2)] = rcut * rcut;
        h_rcutsq.data[m_typpair_idx(typ2, typ1)] = rcut * rcut;

        // store r_cut unmodified for so the neighbor list knows what particles to include
        ArrayHandle<Scalar> h_r_cut_nlist(*m_r_cut_nlist,
                                          access_location::host,
                                          access_mode::readwrite);
        h_r_cut_nlist.data[m_typpair_idx(typ1, typ2)] = rcut;
        h_r_cut_nlist.data[m_typpair_idx(typ2, typ1)] = rcut;
        }

    // notify the neighbor list that we have changed r_cut values
    m_nlist->notifyRCutMatrixChange();
    }

template<class evaluator>
void PotentialPair<evaluator>::setRCutPython(pybind11::tuple types, Scalar r_cut)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    setRcut(typ1, typ2, r_cut);
    }

template<class evaluator> Scalar PotentialPair<evaluator>::getRCut(pybind11::tuple types)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "getting r_cut.");
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    return sqrt(h_rcutsq.data[m_typpair_idx(typ1, typ2)]);
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param ron XPLOR r_on radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
template<class evaluator>
void PotentialPair<evaluator>::setRon(unsigned int typ1, unsigned int typ2, Scalar ron)
    {
    validateTypes(typ1, typ2, "setting r_on");
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::readwrite);
    h_ronsq.data[m_typpair_idx(typ1, typ2)] = ron * ron;
    h_ronsq.data[m_typpair_idx(typ2, typ1)] = ron * ron;
    }

template<class evaluator> Scalar PotentialPair<evaluator>::getROn(pybind11::tuple types)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "getting r_on");
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::read);
    return sqrt(h_ronsq.data[m_typpair_idx(typ1, typ2)]);
    }

template<class evaluator>
void PotentialPair<evaluator>::setROnPython(pybind11::tuple types, Scalar r_on)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    setRon(typ1, typ2, r_on);
    }

/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is
   called to ensure that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/

template<class evaluator> void PotentialPair<evaluator>::computeForces(uint64_t timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);
    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    //     Index2D nli = m_nlist->getNListIndexer();
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    // force arrays
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    const BoxDim box = m_pdata->getGlobalBox();
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    //S
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(),
                                   access_location::host,
                                   access_mode::read);
                                   
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                    access_location::host,
                                    access_mode::read);

    ArrayHandle<Scalar> h_current_neighbor_list( m_pdata->getParticleNList(),
                                                    access_location::host,
                                                    access_mode::readwrite);                        
    assert(h_current_neighbor_list.data);
    size_t p_neighbor_pitch = m_pdata->getParticleNList().getPitch();
    //std::cout<<"new 2"<<std::endl;
    // Copy the previous neighbor list to a new variable
    int tot_particles = (int)(m_pdata->getN());  
    std::vector<Scalar> h_previous_neighbor_temp(20 * tot_particles );
    std::vector<Scalar> h_previous_neighbor_list;
    if (m_K != 0.0 ){
        //std::cout<<"new 3"<<std::endl;
        for ( int i = 0; i < tot_particles; i++) {
            if(h_current_neighbor_list.data[i] != -2){
                for (size_t j = 0; j < 20; j++) { // 20 is the count of neighbors used
                    h_previous_neighbor_temp[j * tot_particles + i] = h_current_neighbor_list.data[j * p_neighbor_pitch + i];
                }
            }
        }
        //std::cout<<"new 4"<<std::endl;
        //std::cout<<"local: "<<tot_particles<<std::endl;
        #ifdef ENABLE_MPI
            if (m_sysdef->isDomainDecomposed()) {
                // Calculate the total number of particles across all ranks
                MPI_Allreduce(MPI_IN_PLACE,
                            &tot_particles,
                            1,
                            MPI_INT,
                            MPI_SUM,
                            m_exec_conf->getMPICommunicator());
            }
        #endif
        
        #ifdef ENABLE_MPI
        //std::cout<<"1"<<std::endl;
        //std::cout<<"totals: "<<tot_particles<<std::endl;
        unsigned int num_ranks = LTIME->num_rank;
        // Initialize a buffer to store gathered data on the root rank   
        std::vector<std::vector<Scalar>> gathered_previous_neighbor_lists(num_ranks);
        std::vector<int> displacements(gathered_previous_neighbor_lists.size());
        if (m_sysdef->isDomainDecomposed()) 
            {   
                // Gather data onto the root rank
                all_gather_v(h_previous_neighbor_temp, gathered_previous_neighbor_lists, m_exec_conf->getMPICommunicator());
                // 
                h_previous_neighbor_list.resize(20 * tot_particles);
                int offset = 0;
                for (unsigned int i = 0; i < num_ranks; i++) {
                    unsigned int ni = static_cast<int>(gathered_previous_neighbor_lists[i].size() / 20); // Number of particles in the current rank
                    for (unsigned int p = 0; p < ni; ++p) {
                        for (int j = 0; j < 20; j++) {
                            h_previous_neighbor_list[j * tot_particles + offset + p] = gathered_previous_neighbor_lists[i][j * ni + p];
                        }
                    }
                offset += ni;
                }
        }else{
                h_previous_neighbor_list.resize(h_previous_neighbor_temp.size());
                std::copy(h_previous_neighbor_temp.begin(), h_previous_neighbor_temp.end(), h_previous_neighbor_list.begin());
            } 
        #endif 
    
    }

    // start the connected neighbors with -2
    //memset((void*)h_current_neighbor_list.data, -2, sizeof(Scalar) * m_pdata->getParticleNList().getNumElements());
    for (size_t i = 0; i <  m_pdata->getParticleNList().getNumElements(); ++i) {
        h_current_neighbor_list.data[i] = -2 ;
    }
    // start the neighbors with tag of the particle so that the particle can be found
    for (int i = 0; i < (int)m_pdata->getN(); i++){
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        if(typei==0){
        h_current_neighbor_list.data[i] = h_tag.data[i];
        }
    }

    //F
    // need to start from a zero force, energy and virial
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());
    size_t idx_pi = -1; 
    // for each particle
            
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        /*
        if (std::isnan(h_pos.data[i].z)) {
            std::cout << "timestep:" <<timestep << std::endl;
            std::cout<<"my rank :"<<m_exec_conf->getRank()<<std::endl;
            std::cout << "Error: Position of particle tagi:" << h_tag.data[i] << " has NaN coordinates." << std::endl;
            continue; // Skip processing this particle or handle the error
        }
        
        std::cout<<"total particles:"<<(int)m_pdata->getN()<<std::endl;
        std::cout<<"tagi:"<<h_tag.data[i]<<std::endl;
        std::cout<<"previous neighbors"<<std::endl;
        for (int j = 0; j < 20; j++) {
            std::cout<<h_previous_neighbor_list[j * tot_particles + i ] <<",";
        }
        std::cout<<std::endl;
        std::cout<<"current neighbors"<<std::endl;
        for (int j = 0; j < 20; j++) {
            std::cout<<h_current_neighbor_list.data[j * p_neighbor_pitch + i] <<",";
        }
        std::cout<<std::endl;*/

        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        if (m_K != 0.0){
            //find the particle in the previous time step particle neighbor list
            if(typei == 0 ){
                //bool wasnt_found = true;
                for(int p_i = 0; p_i < tot_particles; p_i++){
                    if (h_previous_neighbor_list[p_i] == h_tag.data[i]) {
                        idx_pi = p_i;
                        //wasnt_found = false;
                        break;
                    }else{idx_pi = -3;}
                }
            }
            /*
            if (timestep >= 100000 && m_K != 0.0 && idx_pi == -3 ) {
                std::cout<<"load timestep:"<<timestep<<std::endl;
                loadAngleMap(angle_map, "angle_map.bin");
            }*/
        }
        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access charge (if needed)
        Scalar qi = Scalar(0.0);
        if (evaluator::needsCharge())
            qi = h_charge.data[i];

        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar pei = 0.0;
        Scalar virialxxi = 0.0;
        Scalar virialxyi = 0.0;
        Scalar virialxzi = 0.0;
        Scalar virialyyi = 0.0;
        Scalar virialyzi = 0.0;
        Scalar virialzzi = 0.0;

        // loop over all of the neighbors of this particle
        const size_t myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());

            // access charge (if needed)
            Scalar qj = Scalar(0.0);
            if (evaluator::needsCharge())
                qj = h_charge.data[j];

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            // get parameters for this type pair
            unsigned int typpair_idx = m_typpair_idx(typei, typej);
            const param_type& param = m_params[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];
            Scalar ronsq = Scalar(0.0);
            if (m_shift_mode == xplor)
                ronsq = h_ronsq.data[typpair_idx];

            // design specifies that energies are shifted if
            // 1) shift mode is set to shift
            // or 2) shift mode is explor and ron > rcut
            bool energy_shift = false;
            if (m_shift_mode == shift)
                energy_shift = true;
            else if (m_shift_mode == xplor)
                {
                if (ronsq > rcutsq)
                    energy_shift = true;
                }

            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);
            evaluator eval(rsq, rcutsq, param);
            if (evaluator::needsCharge())
                eval.setCharge(qi, qj);

            bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);
            //S
            unsigned int tagi = h_tag.data[i];
            unsigned int tagj = h_tag.data[j];
            
            if (m_K != 0.0){
                if (typei == typej)
                {   
                    Scalar rsq_root = std::sqrt(rsq) - Scalar(2.0);
                    if (rsq_root < Scalar(0.1))
                    {
                        // Check and update neighbors for particle i
                        size_t idx_i = 0;
                        bool not_saved = true;
                        while (idx_i < 20){
                        
                            // Check if tag[j] already exists
                            if (h_current_neighbor_list.data[idx_i * p_neighbor_pitch + i] == h_tag.data[j])
                            {
                                not_saved = false;
                                break; 
                            }

                            // If current entry is empty, save tag[j]
                            if (h_current_neighbor_list.data[idx_i * p_neighbor_pitch + i] == -2)
                            {
                                h_current_neighbor_list.data[idx_i * p_neighbor_pitch + i] = h_tag.data[j];
                                break; 
                            }

                            idx_i++; 
                        }

                        if (not_saved)
                        {
                            // Check and update neighbors for particle j
                            size_t idx_j = 0;
                            while (idx_j < 20)
                            {
                                // Check if tag[i] already exists
                                if (h_current_neighbor_list.data[idx_j * p_neighbor_pitch + j] == h_tag.data[i])
                                {
                                    break; 
                                }

                                // If current entry is empty, save tag[i]
                                if (h_current_neighbor_list.data[idx_j * p_neighbor_pitch + j] == -2)
                                {
                                    h_current_neighbor_list.data[idx_j * p_neighbor_pitch + j] = h_tag.data[i];
                                    break; 
                                }

                                idx_j++; 
                            }
                        }

                            
                            // save angles 
                            bool new_connection = true;
                            size_t is_idx = 0;
                            while ( is_idx < 20)
                            {
                                if (h_previous_neighbor_list[is_idx * tot_particles + idx_pi] == h_tag.data[j])
                                {
                                    new_connection = false;
                                    break;
                                }
                                is_idx++;
                            }
                            
                            if (new_connection && idx_pi != -3)
                            {   
                                // Calculate angle between j and all previous neighbors of i
                                
                                for (size_t idx_si = 1; idx_si < 20 ; ++idx_si)
                                {
                                    if (h_previous_neighbor_list[idx_si * tot_particles + idx_pi] != -2 && h_previous_neighbor_list[idx_si * tot_particles + idx_pi] != tagi)
                                    {  
                                        unsigned int tagk = static_cast<unsigned int>(h_previous_neighbor_list[idx_si * tot_particles + idx_pi]);
                                        Scalar3 pk;
                                        //bool ff = false;
                                        for (int M = 0; M < (int)(m_pdata->getN()+m_pdata->getNGhosts()); M++) 
                                        {
                                            // Check if the tag matches the desired tag
                                            if (h_tag.data[M] == tagk) {
                                                // Access the position of the particle with the matching tag
                                                pk = make_scalar3(h_pos.data[M].x, h_pos.data[M].y, h_pos.data[M].z);
                                                //ff = true;
                                                break; 
                                            }
                                        }
                                        // Calculate vectors between particles
                                        Scalar3 rij = pj-pi;
                                        Scalar3 rik = pk-pi;
                                        rij = box.minImage(rij);
                                        rik = box.minImage(rik);

                                        // Calculate magnitudes of vectors
                                        Scalar rij_mag = std::sqrt(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z);
                                        Scalar rik_mag = std::sqrt(rik.x * rik.x + rik.y * rik.y + rik.z * rik.z);

                                        // Calculate dot product
                                        Scalar dot_product = rij.x * rik.x + rij.y * rik.y + rij.z * rik.z;

                                        // Calculate angle in radians
                                        Scalar cos_theta = dot_product / (rij_mag * rik_mag);

                                        if (cos_theta > 1.0)
                                            cos_theta = 1.0;
                                        if (cos_theta < -1.0)
                                            cos_theta= -1.0;

                                        unsigned int vari = tagi ;
                                        unsigned int varj = tagj ;
                                        unsigned int vark = tagk;

                                        // sort them so var3>var3>var 
                                        unsigned int var1 = vari;
                                        unsigned int var3 = std::max({varj, vark});
                                        unsigned int var2 = std::min({varj, vark});
                                        

                                        unsigned int n = LTIME->num_solvent; 
                                        unsigned int angle_index = (var1 * n*(n-1)/2) + (2*var2*n - var2*var2 + 2*var3 - 3*var2 -2)/2;
                                        

                                        angle_map[angle_index] = acos(cos_theta);

                                        

                                    }
                                } 

                                
                                
                                // Calculate angle between j and all current neighbors of i
                                for (size_t idx_si = 1; idx_si < 20 ; ++idx_si)
                                {
                                    if (h_current_neighbor_list.data[idx_si * p_neighbor_pitch + i] != -2 && h_current_neighbor_list.data[idx_si * p_neighbor_pitch + i] != h_tag.data[j] 
                                    && h_current_neighbor_list.data[idx_si * p_neighbor_pitch + i] != h_tag.data[i])
                                        {
                                        unsigned int tagk = static_cast<unsigned int>(h_current_neighbor_list.data[idx_si * p_neighbor_pitch + i]);
                                        Scalar3 pk;
                                        //bool f2 =false;

                                        for (int M = 0; M < (int)(m_pdata->getN()+m_pdata->getNGhosts()); M++) 
                                        {
                                            // Check if the tag matches the desired tag
                                            if (h_tag.data[M] == tagk) {
                                                // Access the position of the particle with the matching tag
                                                pk = make_scalar3(h_pos.data[M].x, h_pos.data[M].y, h_pos.data[M].z);
                                                //f2= true;
                                                break; 
                                            }
                                        }
                                        // Check if the current neighbor is also a previous neighbor
                                        bool is_previous_neighbor = false;
                                        for (size_t idx_prev = 1; idx_prev < 20 ; ++idx_prev)
                                        {
                                            if (h_previous_neighbor_list[idx_prev * tot_particles + idx_pi] == tagk)
                                            {
                                                is_previous_neighbor = true;
                                                break;
                                            }
                                        }
                                        if (!is_previous_neighbor)
                                        {
                                            // Calculate vectors between particles
                                            Scalar3 rij = pj-pi;
                                            Scalar3 rik = pk-pi;
                                            rij = box.minImage(rij);
                                            rik = box.minImage(rik);

                                            // Calculate magnitudes of vectors
                                            Scalar rij_mag = std::sqrt(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z);
                                            Scalar rik_mag = std::sqrt(rik.x * rik.x + rik.y * rik.y + rik.z * rik.z);

                                            // Calculate dot product
                                            Scalar dot_product = rij.x * rik.x + rij.y * rik.y + rij.z * rik.z;

                                            // Calculate angle in radians
                                            Scalar cos_theta = dot_product / (rij_mag * rik_mag);

                                            if (cos_theta > 1.0)
                                                cos_theta = 1.0;
                                            if (cos_theta < -1.0)
                                                cos_theta= -1.0;

                                            // Calculate angle index
                                            unsigned int vari = tagi ;
                                            unsigned int varj = tagj ;
                                            unsigned int vark = tagk ;

                                            unsigned int var1 = vari;
                                            unsigned int var3 = std::max({varj, vark});
                                            unsigned int var2 = std::min({varj, vark});

                                            unsigned int n = LTIME->num_solvent; 
                                            unsigned int angle_index = (var1 * n*(n-1)/2) + (2*var2*n - var2*var2 + 2*var3 - 3*var2 -2)/2;

                                            // Save the angle
                                            angle_map[angle_index] = acos(cos_theta);

                                            
                                        }

                                    }

                                }
                                //bool j_wasnt_found = true;
                                size_t idx_pj = -1;
                                for(int p_j = 0; p_j < tot_particles; p_j++){
                                    if (h_previous_neighbor_list[p_j] == h_tag.data[j]) {
                                        idx_pj = p_j;
                                        //j_wasnt_found = false;
                                        break;
                                    }else{idx_pj = -3;}
                                }

                                
                                // Calculate angle between i and all previous neighbors of j

                        
                                for (size_t idx_sj = 1; idx_sj < 20 ; ++idx_sj)
                                {
                                    if (h_previous_neighbor_list[idx_sj* tot_particles + idx_pj] != -2 && h_previous_neighbor_list[idx_sj* tot_particles + idx_pj]!= h_tag.data[j] 
                                    && h_previous_neighbor_list[idx_sj* tot_particles + idx_pj]!= h_tag.data[i])
                                    {  
                                      
                                        unsigned int tagk = static_cast<unsigned int>(h_previous_neighbor_list[idx_sj* tot_particles + idx_pj]);
                                        Scalar3 pk;
                                        bool f3=false;

                                        for (int M = 0; M < (int)(m_pdata->getN()+ m_pdata->getNGhosts()); M++) 
                                        {
                                            // Check if the tag matches the desired tag
                                            if (h_tag.data[M] == tagk) {
                                                // Access the position of the particle with the matching tag
                                                pk = make_scalar3(h_pos.data[M].x, h_pos.data[M].y, h_pos.data[M].z);
                                                f3 = true;
                                                break; 
                                            } 
                                            
                                        }
                                        if (f3){

                                            // Calculate vectors between particles
                                            Scalar3 rji = pi-pj;
                                            Scalar3 rjk = pk-pj;
                                            rji = box.minImage(rji);
                                            rjk = box.minImage(rjk);

                                            // Calculate magnitudes of vectors
                                            Scalar rji_mag = std::sqrt(rji.x * rji.x + rji.y * rji.y + rji.z * rji.z);
                                            Scalar rjk_mag = std::sqrt(rjk.x * rjk.x + rjk.y * rjk.y + rjk.z * rjk.z);

                                            // Calculate dot product
                                            Scalar dot_product = rji.x * rjk.x + rji.y * rjk.y + rji.z * rjk.z;

                                            // Calculate angle in radians
                                            Scalar cos_theta = dot_product / (rji_mag * rjk_mag);

                                            if (cos_theta > 1.0)
                                                cos_theta = 1.0;
                                            if (cos_theta < -1.0)
                                                cos_theta= -1.0;

                                            unsigned int vari = tagi ;
                                            unsigned int varj = tagj ;
                                            unsigned int vark = tagk ;

                                            // sort them so var3>var3>var1
                                            unsigned int var1 = varj;
                                            unsigned int var3 = std::max({vari, vark});
                                            unsigned int var2 = std::min({vari, vark});

                                            unsigned int n = LTIME->num_solvent; 
                                            unsigned int angle_index = (var1 * n*(n-1)/2) + (2*var2*n - var2*var2 + 2*var3 - 3*var2 -2)/2;

                                            angle_map[angle_index] = acos(cos_theta);
 
                                        }
                                    }
                                }


                                
                                // Calculate angle between i and all current neighbors of j

                                for (size_t idx_sj = 1; idx_sj < 20 ; ++idx_sj)
                                {
                                    if (h_current_neighbor_list.data[idx_sj* p_neighbor_pitch + j] != -2 && h_current_neighbor_list.data[idx_sj* p_neighbor_pitch + j] != h_tag.data[i]
                                    && h_current_neighbor_list.data[idx_sj* p_neighbor_pitch + j]!= h_tag.data[j] )
                                        {
                                        unsigned int tagk = static_cast<unsigned int>(h_current_neighbor_list.data[idx_sj* p_neighbor_pitch + j]);
                                        Scalar3 pk;
                                        //bool f4=false;

                                        for (int M = 0; M < (int)(m_pdata->getN()+ m_pdata->getNGhosts()); M++) 
                                        {
                                            // Check if the tag matches the desired tag
                                            if (h_tag.data[M] == tagk) {
                                                // Access the position of the particle with the matching tag
                                                pk = make_scalar3(h_pos.data[M].x, h_pos.data[M].y, h_pos.data[M].z);
                                                //f4=true;
                                                break; 
                                            }
                                        }
                                        // Check if the current neighbor is also a previous neighbor
                                        bool is_previous_neighbor = false;
                                        for (size_t idx_prev = 1; idx_prev < 20 ; ++idx_prev)
                                        {
                                            if (h_previous_neighbor_list[idx_prev* tot_particles + idx_pj] == tagk)
                                            {
                                                is_previous_neighbor = true;
                                                break;
                                            }
                                        }
                                        if (!is_previous_neighbor)
                                        {
                                            // Calculate vectors between particles
                                            Scalar3 rji = pi-pj;
                                            Scalar3 rjk = pk-pj;
                                            rji = box.minImage(rji);
                                            rjk = box.minImage(rjk);

                                            // Calculate magnitudes of vectors
                                            Scalar rji_mag = std::sqrt(rji.x * rji.x + rji.y * rji.y + rji.z * rji.z);
                                            Scalar rjk_mag = std::sqrt(rjk.x * rjk.x + rjk.y * rjk.y + rjk.z * rjk.z);

                                            // Calculate dot product
                                            Scalar dot_product = rji.x * rjk.x + rji.y * rjk.y + rji.z * rjk.z;

                                            // Calculate angle in radians
                                            Scalar cos_theta = dot_product / (rji_mag * rjk_mag);

                                            if (cos_theta > 1.0)
                                                cos_theta = 1.0;
                                            if (cos_theta < -1.0)
                                                cos_theta= -1.0;

                                            unsigned int vari = tagi ;
                                            unsigned int varj = tagj ;
                                            unsigned int vark = tagk ;

                                            // sort them so var3>var3>var 
                                            unsigned int var1 = varj;
                                            unsigned int var3 = std::max({vari, vark});
                                            unsigned int var2 = std::min({vari, vark});
                                            

                                            unsigned int n = LTIME->num_solvent; 
                                            unsigned int angle_index = (var1 * n*(n-1)/2) + (2*var2*n - var2*var2 + 2*var3 - 3*var2 -2)/2;
                                            

                                            angle_map[angle_index] = acos(cos_theta);

                                           
                                        }

                                    }

                                }
                                
                            }
                        
                    }
                }
            }
             //F






            if (evaluated)
                {
                // modify the potential for xplor shifting
                if (m_shift_mode == xplor)
                    {
                    if (rsq >= ronsq && rsq < rcutsq)
                        {
                        // Implement XPLOR smoothing (FLOPS: 16)
                        Scalar old_pair_eng = pair_eng;
                        Scalar old_force_divr = force_divr;

                        // calculate 1.0 / (xplor denominator)
                        Scalar xplor_denom_inv
                            = Scalar(1.0)
                              / ((rcutsq - ronsq) * (rcutsq - ronsq) * (rcutsq - ronsq));

                        Scalar rsq_minus_r_cut_sq = rsq - rcutsq;
                        Scalar s = rsq_minus_r_cut_sq * rsq_minus_r_cut_sq
                                   * (rcutsq + Scalar(2.0) * rsq - Scalar(3.0) * ronsq)
                                   * xplor_denom_inv;
                        Scalar ds_dr_divr
                            = Scalar(12.0) * (rsq - ronsq) * rsq_minus_r_cut_sq * xplor_denom_inv;

                        // make modifications to the old pair energy and force
                        pair_eng = old_pair_eng * s;
                        // note: I'm not sure why the minus sign needs to be there: my notes have a
                        // + But this is verified correct via plotting
                        force_divr = s * old_force_divr - ds_dr_divr * old_pair_eng;
                        }
                    }

                Scalar force_div2r = force_divr * Scalar(0.5);
                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fi += dx * force_divr;
                pei += pair_eng * Scalar(0.5);
                if (compute_virial)
                    {
                    virialxxi += force_div2r * dx.x * dx.x;
                    virialxyi += force_div2r * dx.x * dx.y;
                    virialxzi += force_div2r * dx.x * dx.z;
                    virialyyi += force_div2r * dx.y * dx.y;
                    virialyzi += force_div2r * dx.y * dx.z;
                    virialzzi += force_div2r * dx.z * dx.z;
                    }
                // add the force to particle j if we are using the third law (MEM TRANSFER: 10
                // scalars / FLOPS: 8) only add force to local particles
                if (third_law && j < m_pdata->getN())
                    {
                    unsigned int mem_idx = j;
                    h_force.data[mem_idx].x -= dx.x * force_divr;
                    h_force.data[mem_idx].y -= dx.y * force_divr;
                    h_force.data[mem_idx].z -= dx.z * force_divr;
                    h_force.data[mem_idx].w += pair_eng * Scalar(0.5);

                    /*if ((timestep-100000) % 10000 == 0){
                    Scalar fj_mag = std::sqrt(h_force.data[mem_idx].x * h_force.data[mem_idx].x 
                    + h_force.data[mem_idx].y * h_force.data[mem_idx].y + h_force.data[mem_idx].z * h_force.data[mem_idx].z);
                    std::cout << h_force.data[mem_idx].w << "," << fj_mag << "," << (rsq-Scalar(2.0)) << std::endl;
                    }*/
                    if (compute_virial)
                        {
                        h_virial.data[0 * m_virial_pitch + mem_idx] += force_div2r * dx.x * dx.x;
                        h_virial.data[1 * m_virial_pitch + mem_idx] += force_div2r * dx.x * dx.y;
                        h_virial.data[2 * m_virial_pitch + mem_idx] += force_div2r * dx.x * dx.z;
                        h_virial.data[3 * m_virial_pitch + mem_idx] += force_div2r * dx.y * dx.y;
                        h_virial.data[4 * m_virial_pitch + mem_idx] += force_div2r * dx.y * dx.z;
                        h_virial.data[5 * m_virial_pitch + mem_idx] += force_div2r * dx.z * dx.z;
                        }
                    }
                }
            }

        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = i;
        h_force.data[mem_idx].x += fi.x;
        h_force.data[mem_idx].y += fi.y;
        h_force.data[mem_idx].z += fi.z;
        h_force.data[mem_idx].w += pei;
        if (compute_virial)
            {
            h_virial.data[0 * m_virial_pitch + mem_idx] += virialxxi;
            h_virial.data[1 * m_virial_pitch + mem_idx] += virialxyi;
            h_virial.data[2 * m_virial_pitch + mem_idx] += virialxzi;
            h_virial.data[3 * m_virial_pitch + mem_idx] += virialyyi;
            h_virial.data[4 * m_virial_pitch + mem_idx] += virialyzi;
            h_virial.data[5 * m_virial_pitch + mem_idx] += virialzzi;
            }
        }

    if (m_K != 0.0){
        //MPI_Barrier(m_exec_conf->getMPICommunicator());
    #ifdef ENABLE_MPI
        unsigned int num_rank = LTIME->num_rank;
        //unsigned int my_rank = m_exec_conf->getRank();
        vector<map<unsigned int, Scalar>> angle_map_temp1(num_rank);

        if (m_sysdef->isDomainDecomposed()) 
        {
            all_gather_v(angle_map, angle_map_temp1, m_exec_conf->getMPICommunicator()); 
            // Iterate over the vector of maps from different ranks and merge into angle_map_temp2
            for (unsigned int i = 0; i < num_rank; ++i) {
                for (const auto& entry : angle_map_temp1[i]) {
                    angle_map_temp2[entry.first] = entry.second;
                }
            }

            angle_map_temp1.clear();
            angle_map.clear(); 
            /*   
            if (m_K != 0.0 && (timestep-100000) % 10000 == 0 && m_exec_conf->getRank()==0) {
            std::cout<<"save timestep:"<<timestep<<std::endl;
            saveAngleMap(angle_map_temp2, "angle_map2.bin");
            }*/ 
        }
    
   

    //~ add bond_calc [PROCF2023] 
    
    	LTIME->updatebondtime(timestep);

    //
    #endif
    if (m_K != 0.0 && (timestep-100000) % 10000 == 0 && m_exec_conf->getRank()==0) {
        std::cout<<"timestep: "<< timestep <<std::endl;
    }
    for (int i = 0; i < (int)m_pdata->getN(); i++){
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        if (typei==0) {
                //Scalar kk =0;
                size_t nonzero_count = 0;
                unsigned int tagi = h_tag.data[i]; 
                for (size_t idx_i = 1; idx_i < 20 ; ++idx_i)
                {
                    // check if i has more than 1 neighbor
                    if (h_current_neighbor_list.data[idx_i* p_neighbor_pitch +i] != -2 
                    && h_current_neighbor_list.data[idx_i* p_neighbor_pitch +i] != tagi)
                    {
                        nonzero_count++; //count the neighbors
                    }
                }
                if (nonzero_count > 1)
                {
                    #define SMALL Scalar(0.001)
                    // go over each angle between two different neighbors of i and apply angle force 
                    // find the first neighbor a
                    for (size_t idx_a = 1; idx_a < (nonzero_count); ++idx_a)
                    {
                        unsigned int taga = static_cast<unsigned int>(h_current_neighbor_list.data[idx_a* p_neighbor_pitch +i]);
                        Scalar3 pa;
                        int a = -1;
                        //bool f5=false;
                        for (int M = 0; M < (int)(m_pdata->getN()+ m_pdata->getNGhosts()); M++)
                        {
                            if (h_tag.data[M] == taga) {
                            // Access the position of the particle with the matching tag
                                pa = make_scalar3(h_pos.data[M].x, h_pos.data[M].y, h_pos.data[M].z);
                                a = M; 
                                //f5=true;
                                break; 
                                }
                        }

                        unsigned int typea = __scalar_as_int(h_pos.data[a].w);
                        // find the second neighbor b
                        for (size_t idx_b = (idx_a+1) ; idx_b < (nonzero_count+1); ++idx_b)
                        {
                            unsigned int tagb = static_cast<unsigned int>(h_current_neighbor_list.data[idx_b* p_neighbor_pitch +i]);
                            Scalar3 pb;
                            int b = -1;
                            //bool f6=false;
                            for (int N = 0; N < (int)(m_pdata->getN()+ m_pdata->getNGhosts()); N++)
                            {
                                if (h_tag.data[N] == tagb) {
                                    // Access the position of the particle with the matching tag
                                    pb = make_scalar3(h_pos.data[N].x, h_pos.data[N].y, h_pos.data[N].z);
                                    b = N;
                                    //f6 = true;
                                    break; 
                                    }

                            }
                            unsigned int typeb = __scalar_as_int(h_pos.data[b].w);

                            

                            if (tagb != taga  && taga != tagi && tagb != tagi && typea==0 && typeb==0)
                            {
                                // Calculate angle index
                                unsigned int vari = tagi ;
                                unsigned int vara = taga ;
                                unsigned int varb = tagb ;

                                unsigned int var1 = vari;
                                unsigned int var3 = std::max({vara, varb});
                                unsigned int var2 = std::min({vara, varb});

                                unsigned int n = LTIME->num_solvent; 
                                unsigned int current_angle_index = (var1 * n*(n-1)/2) + (2*var2*n - var2*var2 + 2*var3 - 3*var2 -2)/2;

                                Scalar3 ria = make_scalar3(pa.x - pi.x, pa.y - pi.y, pa.z - pi.z);
                                Scalar3 rib = make_scalar3(pb.x - pi.x, pb.y - pi.y, pb.z - pi.z);
                                ria = box.minImage(ria);
                                rib = box.minImage(rib);

                                // Calculate magnitudes of vectors
                                Scalar ria_mag = std::sqrt(ria.x * ria.x + ria.y * ria.y + ria.z * ria.z);
                                Scalar rib_mag = std::sqrt(rib.x * rib.x + rib.y * rib.y + rib.z * rib.z);

                                // Calculate dot product
                                Scalar dot_product = ria.x * rib.x + ria.y * rib.y + ria.z * rib.z;

                                // Calculate cosine of the angle 
                                Scalar current_cos_theta = dot_product / (ria_mag * rib_mag);
                                

                                if (current_cos_theta > 1.0)
                                    current_cos_theta = 1.0;
                                if (current_cos_theta < -1.0)
                                    current_cos_theta = -1.0;
                                
                                Scalar current_sin_theta = std::sqrt(1.0 - current_cos_theta * current_cos_theta);
                                if (current_sin_theta < SMALL)
                                    current_sin_theta = SMALL;
                            
                                current_sin_theta = 1.0 / current_sin_theta;


                                
                                assert(angle_map);
                                // Calculate deviation from equilibrium angle which was saved previously 
                                Scalar dth;
                                if (m_sysdef->isDomainDecomposed()){
                                    dth = acos(current_cos_theta) - angle_map_temp2[current_angle_index];
                                }else{
                                    dth = acos(current_cos_theta) - angle_map[current_angle_index];
                                }

                                //Scalar eq_theta = 180.0 * M_PI / 180.0;
                                //Scalar tk = m_K * (acos(current_cos_theta) - eq_theta);
                                Scalar tk = m_K * dth;


                                
                                // Calculate force magnitude for K(thera - theta_0)^2
                                Scalar vab = -1.0 * tk * current_sin_theta;

                                /*
                                // Calculate force magnitude for Emanuela equation
                                Scalar B = m_K;
                                Scalar w = 0.3;
                                Scalar eq_theta = 120.0 * M_PI / 180.0;
                                //Scalar eq_theta = angle_map_temp2[current_angle_index];

                                Scalar Aia = pow(ria_mag/2, -10) * pow(1 - pow(ria_mag / 4, 10), 2);
                                Scalar Aib = pow(rib_mag/2, -10) * pow(1 - pow(rib_mag / 4, 10), 2);

                                Scalar exp_theta = exp(-pow((current_cos_theta - cos(eq_theta)) / w, 2));

                                Scalar vab = (-2 * B * current_sin_theta / pow(w, 2)) * Aia * Aib * exp_theta * (current_cos_theta - cos(eq_theta));
                                */

                                Scalar a11 = vab * current_cos_theta / (ria_mag * ria_mag);
                                Scalar a12 = -vab / (ria_mag * rib_mag);
                                Scalar a22 = vab * current_cos_theta / (rib_mag * rib_mag);

                                // Calculate forces
                                Scalar3 fia = make_scalar3(a11 * ria.x + a12 * rib.x,
                                                            a11 * ria.y + a12 * rib.y,
                                                            a11 * ria.z + a12 * rib.z);

                                Scalar3 fib = make_scalar3(a22 * ria.x + a12 * rib.x,
                                                            a22 * ria.y + a12 * rib.y,
                                                            a22 * ria.z + a12 * rib.z);

                                // compute the energy, for each atom in the angle for K(thera - theta_0)^2
                                Scalar angle_eng = (tk * dth) * Scalar(1.0 / 6.0);


                                // compute the energy, for each atom in the angle for Emanuela equation
                                //Scalar angle_eng = (B * Aia * Aib * exp_theta)/3;
                                
                                
                                if ((timestep-100000) % 10000 == 0){
                                Scalar fia_mag = std::sqrt(fia.x * fia.x + fia.y * fia.y + fia.z * fia.z);
                                std::cout << angle_eng << "," << fia_mag << "," << current_cos_theta <<","<<angle_map_temp2[current_angle_index]<<","<< (ria_mag-Scalar(2.0)) <<","<< (ria_mag-Scalar(2.0)) << std::endl;
                                }

                                Scalar angle_virial[6];
                                angle_virial[0] = Scalar(1. / 3.) * (ria.x * fia.x + rib.x * fib.x);
                                angle_virial[1] = Scalar(1. / 3.) * (ria.y * fia.x + rib.y * fib.x);
                                angle_virial[2] = Scalar(1. / 3.) * (ria.z * fia.x + rib.z * fib.x);
                                angle_virial[3] = Scalar(1. / 3.) * (ria.y * fia.y + rib.y * fib.y);
                                angle_virial[4] = Scalar(1. / 3.) * (ria.z * fia.y + rib.z * fib.y);
                                angle_virial[5] = Scalar(1. / 3.) * (ria.z * fia.z + rib.z * fib.z);

                                
                                // Update forces and virials for particle i, a and b
                                
                                if (a < (int)(m_pdata->getN()+ m_pdata->getNGhosts())) {
                                    h_force.data[a].x += fia.x;
                                    h_force.data[a].y += fia.y;
                                    h_force.data[a].z += fia.z;
                                    h_force.data[a].w += angle_eng;
                                    for (int l = 0; l < 6; l++)
                                        h_virial.data[l * m_virial_pitch + a] += angle_virial[l]; 
                                }
                                
                                if (i < (int)m_pdata->getN()) {
                                    h_force.data[i].x -= fia.x + fib.x; 
                                    h_force.data[i].y -= fia.y + fib.y;
                                    h_force.data[i].z -= fia.z + fib.z;
                                    h_force.data[i].w += angle_eng;
                                    for (int l = 0; l < 6; l++)
                                        h_virial.data[l * m_virial_pitch + i] += angle_virial[l]; 
                                }

                                if (b < (int)(m_pdata->getN()+ m_pdata->getNGhosts())) {
                                    h_force.data[b].x += fib.x;
                                    h_force.data[b].y += fib.y;
                                    h_force.data[b].z += fib.z;
                                    h_force.data[b].w += angle_eng;
                                    for (int l = 0; l < 6; l++)
                                        h_virial.data[l * m_virial_pitch + b] += angle_virial[l]; 
                                }
                                /*
                                if (timestep >= 100660){
                                    if (taga == 36963 || tagb == 36963 || tagi==36963){
                                    std::cout << "timestep:"<<timestep<<std::endl;
                                    std::cout << "taga:"<<taga<<", tagb:"<<tagb<<", tagi:"<<tagi<<std::endl;
                                    std::cout<<"current angle:"<<acos(current_cos_theta)<<", Equilibrium angle:" <<angle_map_temp2[current_angle_index]<<std::endl;
                                    std::cout<<"fab:"<<fia.x<<","<<fia.y<<","<<fia.z<<std::endl;
                                    std::cout<<"total force on a:"<<h_force.data[a].x<<","<<h_force.data[a].y<<","<<h_force.data[a].z<<std::endl;
                                    std::cout<<"total force on b:"<<h_force.data[b].x<<","<<h_force.data[b].y<<","<<h_force.data[b].z<<std::endl;
                                    std::cout<<"total force on i:"<<h_force.data[i].x<<","<<h_force.data[i].y<<","<<h_force.data[i].z<<std::endl;
                                    }

                                    if (taga == 3371 || tagb == 3371 || tagi==3371){
                                    std::cout << "timestep:"<<timestep<<std::endl;
                                    std::cout << "taga:"<<taga<<", tagb:"<<tagb<<", tagi:"<<tagi<<std::endl;
                                    std::cout<<"current angle:"<<acos(current_cos_theta)<<", Equilibrium angle:" <<angle_map_temp2[current_angle_index]<<std::endl;
                                    std::cout<<"fab:"<<fia.x<<","<<fia.y<<","<<fia.z<<std::endl;
                                    std::cout<<"total force on a:"<<h_force.data[a].x<<","<<h_force.data[a].y<<","<<h_force.data[a].z<<std::endl;
                                    std::cout<<"total force on b:"<<h_force.data[b].x<<","<<h_force.data[b].y<<","<<h_force.data[b].z<<std::endl;
                                    std::cout<<"total force on i:"<<h_force.data[i].x<<","<<h_force.data[i].y<<","<<h_force.data[i].z<<std::endl;
                                    }
                                }   */                  
                            }
                        }
                    }
                } 
            }
        }
  
    //F   
    }

    computeTailCorrection();
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template<class evaluator>
CommFlags PotentialPair<evaluator>::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = CommFlags(0);

    if (evaluator::needsCharge())
        flags[comm_flag::charge] = 1;

    flags |= ForceCompute::getRequestedCommFlags(timestep);

    return flags;
    }
#endif

template<class evaluator> void PotentialPair<evaluator>::startAutotuning()
    {
    ForceCompute::startAutotuning();

    // Start autotuning the neighbor list.
    m_nlist->startAutotuning();
    }

template<class evaluator> bool PotentialPair<evaluator>::isAutotuningComplete()
    {
    bool result = ForceCompute::isAutotuningComplete();
    return result && m_nlist->isAutotuningComplete();
    }

//! function to compute the energy between two lists of particles.
//! strictly speaking tags1 and tags2 should be disjoint for the result to make any sense.
//! \param energy is the sum of the energies between all particles in tags1 and tags2, U = \sum_{i
//! \in tags1, j \in tags2} u_{ij}.
template<class evaluator>
template<class InputIterator>
inline void PotentialPair<evaluator>::computeEnergyBetweenSets(InputIterator first1,
                                                               InputIterator last1,
                                                               InputIterator first2,
                                                               InputIterator last2,
                                                               Scalar& energy)
    {
    if (first1 == last1 || first2 == last2)
        return;

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // temporarily add tag comm flag
        CommFlags old_flags = m_comm->getFlags();
        CommFlags new_flags = old_flags;
        new_flags[comm_flag::tag] = 1;
        m_comm->setFlags(new_flags);

        // force communication
        m_comm->migrateParticles();
        m_comm->exchangeGhosts();

        // reset the old flags
        m_comm->setFlags(old_flags);


        // Paniz
        CommFlags flags = CommFlags(0);
        flags[comm_flag::position] = 1;
        flags[comm_flag::net_force] = 1;
        flags[comm_flag::net_virial] = 1;
        flags[comm_flag::particle_n_list] = 1;
        flags[comm_flag::tag] = 1;
        }
#endif

    energy = Scalar(0.0);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtags(m_pdata->getRTags(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    const BoxDim box = m_pdata->getGlobalBox();
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);

    // for each particle in tags1
    while (first1 != last1)
        {
        unsigned int i = h_rtags.data[*first1];
        first1++;
        if (i >= m_pdata->getN()) // not owned by this processor.
            continue;
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);

        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access charge (if needed)
        Scalar qi = Scalar(0.0);
        if (evaluator::needsCharge())
            qi = h_charge.data[i];

        // loop over all particles in tags2
        for (InputIterator iter = first2; iter != last2; ++iter)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_rtags.data[*iter];
            if (j >= m_pdata->getN() + m_pdata->getNGhosts()) // not on this processor at all
                continue;
            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());

            // access charge (if needed)
            Scalar qj = Scalar(0.0);
            if (evaluator::needsCharge())
                qj = h_charge.data[j];

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            // get parameters for this type pair
            unsigned int typpair_idx = m_typpair_idx(typei, typej);
            const param_type& param = m_params[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];
            Scalar ronsq = Scalar(0.0);
            if (m_shift_mode == xplor)
                ronsq = h_ronsq.data[typpair_idx];

            // design specifies that energies are shifted if
            // 1) shift mode is set to shift
            // or 2) shift mode is explor and ron > rcut
            bool energy_shift = false;
            if (m_shift_mode == shift)
                energy_shift = true;
            else if (m_shift_mode == xplor)
                {
                if (ronsq > rcutsq)
                    energy_shift = true;
                }

            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);
            evaluator eval(rsq, rcutsq, param);
            if (evaluator::needsCharge())
                eval.setCharge(qi, qj);

            bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);

            if (evaluated)
                {
                // modify the potential for xplor shifting
                if (m_shift_mode == xplor)
                    {
                    if (rsq >= ronsq && rsq < rcutsq)
                        {
                        // Implement XPLOR smoothing (FLOPS: 16)
                        Scalar old_pair_eng = pair_eng;
                        Scalar old_force_divr = force_divr;

                        // calculate 1.0 / (xplor denominator)
                        Scalar xplor_denom_inv
                            = Scalar(1.0)
                              / ((rcutsq - ronsq) * (rcutsq - ronsq) * (rcutsq - ronsq));

                        Scalar rsq_minus_r_cut_sq = rsq - rcutsq;
                        Scalar s = rsq_minus_r_cut_sq * rsq_minus_r_cut_sq
                                   * (rcutsq + Scalar(2.0) * rsq - Scalar(3.0) * ronsq)
                                   * xplor_denom_inv;
                        Scalar ds_dr_divr
                            = Scalar(12.0) * (rsq - ronsq) * rsq_minus_r_cut_sq * xplor_denom_inv;

                        // make modifications to the old pair energy and force
                        pair_eng = old_pair_eng * s;
                        // note: I'm not sure why the minus sign needs to be there: my notes have a
                        // + But this is verified correct via plotting
                        force_divr = s * old_force_divr - ds_dr_divr * old_pair_eng;
                        }
                    }
                energy += pair_eng;
                }
            }
        }
#ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &energy,
                      1,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif
    }

//! Calculates the energy between two lists of particles.
template<class evaluator>
Scalar PotentialPair<evaluator>::computeEnergyBetweenSetsPythonList(
    pybind11::array_t<int, pybind11::array::c_style> tags1,
    pybind11::array_t<int, pybind11::array::c_style> tags2)
    {
    Scalar eng = 0.0;
    if (tags1.ndim() != 1)
        throw std::domain_error("error: ndim != 2");
    unsigned int* itags1 = (unsigned int*)tags1.mutable_data();

    if (tags2.ndim() != 1)
        throw std::domain_error("error: ndim != 2");
    unsigned int* itags2 = (unsigned int*)tags2.mutable_data();
    computeEnergyBetweenSets(itags1, itags1 + tags1.size(), itags2, itags2 + tags2.size(), eng);
    return eng;
    }

namespace detail
    {
//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_PotentialPair(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialPair<T>, ForceCompute, std::shared_ptr<PotentialPair<T>>>
        potentialpair(m, name.c_str());
    potentialpair
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar>())
        .def("setParams", &PotentialPair<T>::setParamsPython)
        .def("getParams", &PotentialPair<T>::getParams)
        .def("setRCut", &PotentialPair<T>::setRCutPython)
        .def("getRCut", &PotentialPair<T>::getRCut)
        .def("setROn", &PotentialPair<T>::setROnPython)
        .def("getROn", &PotentialPair<T>::getROn)
        .def_property("mode",
                      &PotentialPair<T>::getShiftMode,
                      &PotentialPair<T>::setShiftModePython)
        .def_property("tail_correction",
                      &PotentialPair<T>::getTailCorrectionEnabled,
                      &PotentialPair<T>::setTailCorrectionEnabled)
        .def("computeEnergyBetweenSets", &PotentialPair<T>::computeEnergyBetweenSetsPythonList)
        .def_property("K", &PotentialPair<T>::getK, &PotentialPair<T>::setK);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // __POTENTIAL_PAIR_H__
