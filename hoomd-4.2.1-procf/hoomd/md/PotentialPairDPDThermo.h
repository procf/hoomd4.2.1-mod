// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// ########## Modified by PRO-CF //~ [PROCF2023] ##########

#ifndef __POTENTIAL_PAIR_DPDTHERMO_H__
#define __POTENTIAL_PAIR_DPDTHERMO_H__

#include "PotentialPair.h"
#include "hoomd/Variant.h"
#include "Lifetime.h" //~ add Lifetime.h [PROCF2023]

// paniz_write
#include "hoomd/HOOMDMPI.h"
#include <memory>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>

// _write

/*! \file PotentialPairDPDThermo.h
    \brief Defines the template class for a dpd thermostat and LJ pair potential
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {
//! Template class for computing dpd thermostat and LJ pair potential
/*! <b>Overview:</b>
    TODO - Revise Documentation Below

    PotentialPairDPDThermo computes a dpd thermostat and Lennard Jones pair potentials (and forces)
   between all particle pairs in the simulation. It employs the use of a neighbor list to limit the
   number of computations done to only those particles with the cutoff radius of each other. The
   computation of the actual V(r) is not performed directly by this class, but by an evaluator class
   (e.g. EvaluatorPairDPDLJThermo) which is passed in as a template parameter so the computations
    are performed as efficiently as possible.

    PotentialPairDPDThermo handles most of the gory internal details common to all standard pair
   potentials.
     - A cutoff radius to be specified per particle type pair for the conservative and stochastic
   potential
     - Per type pair parameters are stored and a set method is provided
     - And all the details about looping through the particles, computing dr, computing the virial,
   etc. are handled

    \sa export_PotentialPairDPDThermo()
*/
template<class evaluator> class PotentialPairDPDThermo : public PotentialPair<evaluator>
    {
    public:
    //! Param type from evaluator
    typedef typename evaluator::param_type param_type;

    //! Construct the pair potential
    PotentialPairDPDThermo(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<NeighborList> nlist,
                           bool bond_calc,
                           Scalar K); //~ add bond_calc [PROCF2023]
    //! Destructor
    virtual ~PotentialPairDPDThermo() {};

    //! Set the temperature
    virtual void setT(std::shared_ptr<Variant> T);

    //! Get the temperature
    virtual std::shared_ptr<Variant> getT();


    //~! Check the bond_calc flag [PROCF2023]
    void setBondCalcEnabled(bool bond_calc)
	{
	m_bond_calc = bond_calc;
	}
    bool getBondCalcEnabled()
	{
	return m_bond_calc;
	}


    void setK(Scalar K) {
        m_K = K;
    }
    Scalar getK() const {
        return m_K;
    }

    //~
    //~ add Lifetime [PROCF2023] 
    std::shared_ptr<Lifetime> LTIME;
    //~

    std::map<unsigned int, Scalar> angle_map;
    std::map<unsigned int, Scalar> angle_map_temp2;
    



#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    virtual CommFlags getRequestedCommFlags(uint64_t timestep);
#endif

    protected:
    std::shared_ptr<Variant> m_T; //!< Temperature for the DPD thermostat
    bool m_bond_calc; //= false;      //~!< bond_calc flag (default false) [PROCF2023]
    Scalar m_K;
    //! Actually compute the forces (overwrites PotentialPair::computeForces())
    virtual void computeForces(uint64_t timestep);

    const std::shared_ptr<SystemDefinition> m_sysdef;
    //std::shared_ptr<ParticleData> m_p;
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
    
    };
// Initialize the static member variable outside the class definition
//template<class evaluator>
//std::vector<std::vector<Scalar>> PotentialPairDPDThermo<evaluator>::accumulated_neighbor_lists;


/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
*/
template<class evaluator>
PotentialPairDPDThermo<evaluator>::PotentialPairDPDThermo(std::shared_ptr<SystemDefinition> sysdef,
                                                          std::shared_ptr<NeighborList> nlist,
                                                          bool bond_calc,
                                                          Scalar K) //~ add bond_calc [PROCF2023]
    : PotentialPair<evaluator>(sysdef, nlist), m_bond_calc(bond_calc), m_K(K), m_sysdef(sysdef)//~ add bond_calc [PROCF2023],
    {
    m_exec_conf = m_sysdef->getParticleData()->getExecConf();

    //~ add bond_calc flag [PROCF2023]
    if(m_K != 0.0)
	{
        LTIME = std::shared_ptr<Lifetime>(new Lifetime(sysdef));
	}
    //~
    }
    
/*! \param T the temperature the system is thermostated on this time step.
 */
template<class evaluator> void PotentialPairDPDThermo<evaluator>::setT(std::shared_ptr<Variant> T)
    {
    m_T = T;
    }


/*! Gets the K*/
template<class evaluator> std::shared_ptr<Variant> PotentialPairDPDThermo<evaluator>::getT() 
    {
    return m_T;
    }



/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is
   called to ensure that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template<class evaluator> void PotentialPairDPDThermo<evaluator>::computeForces(uint64_t timestep)
    {

    // start by updating the neighborlist
    this->m_nlist->compute(timestep);

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(this->m_nlist->getNNeighArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<unsigned int> h_nlist(this->m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<size_t> h_head_list(this->m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);

    ArrayHandle<Scalar4> h_pos(this->m_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);
    ArrayHandle<Scalar4> h_vel(this->m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::read);
    ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(),
                                    access_location::host,
                                    access_mode::read);
    //~ access particle diameter [PROCF2023] 
    ArrayHandle<Scalar> h_diameter(this->m_pdata->getDiameters(),
                                   access_location::host,
                                   access_mode::read);
    //~ 
    //S
    ArrayHandle<Scalar> h_current_neighbor_list( this->m_pdata->getParticleNList(),
                                                 access_location::host,
                                                 access_mode::readwrite);                        
    assert(h_current_neighbor_list.data);
    size_t p_neighbor_pitch = this->m_pdata->getParticleNList().getPitch();

    // Copy the previous neighbor list to a new variable
    int tot_particles = (int)(this->m_pdata->getN());  
    std::vector<Scalar> h_previous_neighbor_temp(20 * tot_particles );
    std::vector<Scalar> h_previous_neighbor_list;

    if (m_K != 0.0 ){
        for ( int i = 0; i < tot_particles; i++) {
            if(h_current_neighbor_list.data[i] != -2){
                for (size_t j = 0; j < 20; j++) { // 20 is the count of neighbors used
                    h_previous_neighbor_temp[j * tot_particles + i] = h_current_neighbor_list.data[j * p_neighbor_pitch + i];
                }
            }
        }

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
        unsigned int num_ranks = this->LTIME->num_rank;
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

    // start the connected neighbors with zero
    //memset((void*)h_current_neighbor_list.data, 0, sizeof(Scalar) * this->m_pdata->getParticleNList().getNumElements());
    for (size_t i = 0; i <  this->m_pdata->getParticleNList().getNumElements(); ++i) {
        h_current_neighbor_list.data[i] = -2 ;
    }
    // start the neighbors with tag of the particle so that the particle can be found
    for (int i = 0; i < (int)this->m_pdata->getN(); i++){
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        if(typei){
        h_current_neighbor_list.data[i] = h_tag.data[i];
        }
    }
    //F
    // force arrays
    ArrayHandle<Scalar4> h_force(this->m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(this->m_virial, access_location::host, access_mode::overwrite);
    //~ add virial_ind [PROCF2023]
    ArrayHandle<Scalar> h_virial_ind(this->m_virial_ind, access_location::host, access_mode::overwrite);
    //~

    const BoxDim box = this->m_pdata->getBox();
    //~ get additional box details [PROCF2023] 
    Scalar3 L2 = box.getL();
    uchar3 per_ = box.getPeriodic();
    //~
    Scalar shear_rate = this->m_SR; //~ add shear rate [PROCF2023] 
    ArrayHandle<Scalar> h_ronsq(this->m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(this->m_rcutsq, access_location::host, access_mode::read);

    
    // need to start from a zero force, energy and virial
    memset((void*)h_force.data, 0, sizeof(Scalar4) * this->m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * this->m_virial.getNumElements());
    //~ add virial_ind [PROCF2023] 
    memset((void*)h_virial_ind.data, 0, sizeof(Scalar) * this->m_virial_ind.getNumElements());
    //~
    
    uint16_t seed = this->m_sysdef->getSeed();
    // index of the particle neighbor in previous time step particle neighbor list
    // for each particle
    size_t idx_pi = -1; 
    for (int i = 0; i < (int)this->m_pdata->getN(); i++)
        {

        //find the particle in the previous time step particle neighbor list
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        if (m_K != 0.0){
            if(typei){
                //bool wasnt_found = true;
                for(int p_i = 0; p_i < tot_particles; p_i++){
                    if (h_previous_neighbor_list[p_i] == h_tag.data[i]) {
                        idx_pi = p_i;
                        //wasnt_found = false;
                        break;
                    }else{idx_pi = 0;}
                }
            
            }
        }
        // access the particle's position, velocity, and type (MEM TRANSFER: 7 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        Scalar3 vi = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);

        //unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const size_t head_i = h_head_list.data[i];

        // sanity check
        assert(typei < this->m_pdata->getNTypes());

        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar pei = 0.0;
        Scalar viriali[6];
        for (unsigned int l = 0; l < 6; l++)
            viriali[l] = 0.0;
        //~ initialize the current virial_ind to zero [PROCF2023]
        Scalar viriali_ind[5];
        for (unsigned int l = 0; l < 5; l++)
            viriali_ind[l] = 0.0;
        //~

        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int)h_n_neigh.data[i];

        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[head_i + k];
            assert(j < this->m_pdata->getN() + this->m_pdata->getNGhosts());

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;

            // calculate dv_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 vj = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
            Scalar3 dv = vi - vj;

	        //~ add shear rate [PROCF2023] 
            if(shear_rate != Scalar(0.0) && (int)per_.y){
               if (abs(dx.y) > Scalar(0.5)*L2.y){
                   if(dx.y > Scalar(0.0)) dv.x -= shear_rate;
                   else dv.x += shear_rate;
                   }
               }
	        //~

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < this->m_pdata->getNTypes());

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            // calculate the drag term r \dot v
            Scalar rdotv = dot(dx, dv);

            // get parameters for this type pair
            unsigned int typpair_idx = this->m_typpair_idx(typei, typej);
            const param_type& param = this->m_params[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];

            // design specifies that energies are shifted if
            // 1) shift mode is set to shift
            bool energy_shift = false;
            if (this->m_shift_mode == this->shift)
                energy_shift = true;

            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar force_divr_cons = Scalar(0.0);
            //~ add virial_ind terms [PROCF2023] 
            Scalar cons_divr = Scalar(0.0);
            Scalar disp_divr = Scalar(0.0);
            Scalar rand_divr = Scalar(0.0);
            Scalar sq_divr = Scalar(0.0);
            Scalar cont_divr = Scalar(0.0);
	        //~
            Scalar pair_eng = Scalar(0.0);
            evaluator eval(rsq, rcutsq, param);

            // Special Potential Pair DPD Requirements
            const Scalar currentTemp = m_T->operator()(timestep);

            // set seed using global tags
            unsigned int tagi = h_tag.data[i];
            unsigned int tagj = h_tag.data[j];
            eval.set_seed_ij_timestep(seed, tagi, tagj, timestep);
            eval.setDeltaT(this->m_deltaT);
            eval.setRDotV(rdotv);
            eval.setT(currentTemp);


            h_diameter.data[i] = Scalar (2.0);
            h_diameter.data[j] = Scalar (2.0);

            //S
            if (m_K != 0.0){
                if (typei && typej)
                {   
                    Scalar rsq_root = std::sqrt(rsq) - Scalar(0.5) * (h_diameter.data[i] + h_diameter.data[j]);
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
                            
                            if (new_connection)
                            {   
                                // Calculate angle between j and all previous neighbors of i
                                for (size_t idx_si = 1; idx_si < 20 ; ++idx_si)
                                {
                                    if (h_previous_neighbor_list[idx_si * tot_particles + idx_pi] != -2)
                                    {  
                                        unsigned int tagk = static_cast<unsigned int>(h_previous_neighbor_list[idx_si * tot_particles + idx_pi]);
                                        Scalar3 pk;
                                        //bool ff = false;
                                        for (int M = 0; M < (int)(this->m_pdata->getN()+this->m_pdata->getNGhosts()); M++) 
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

                                        unsigned int vari = tagi - this->LTIME->num_solvent;
                                        unsigned int varj = tagj - this->LTIME->num_solvent;
                                        unsigned int vark = tagk - this->LTIME->num_solvent;

                                        // sort them so var3>var3>var 
                                        unsigned int var1 = vari;
                                        unsigned int var3 = std::max({varj, vark});
                                        unsigned int var2 = std::min({varj, vark});
                                        

                                        unsigned int n = this->LTIME->num_colloid; 
                                        unsigned int angle_index = (var1 * n*(n-1)/2) + (2*var2*n - var2*var2 + 2*var3 - 3*var2 -2)/2;
                                        

                                        angle_map[angle_index] = acos(cos_theta);

                                    }
                                }

                                
                                
                                // Calculate angle between j and all current neighbors of i
                                for (size_t idx_si = 1; idx_si < 20 ; ++idx_si)
                                {
                                    if (h_current_neighbor_list.data[idx_si * p_neighbor_pitch + i] != -2 && h_current_neighbor_list.data[idx_si * p_neighbor_pitch + i] != h_tag.data[j])
                                        {
                                        unsigned int tagk = static_cast<unsigned int>(h_current_neighbor_list.data[idx_si * p_neighbor_pitch + i]);
                                        Scalar3 pk;
                                        //bool f2 =false;

                                        for (int M = 0; M < (int)(this->m_pdata->getN()+ this->m_pdata->getNGhosts()); M++) 
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
                                            unsigned int vari = tagi - this->LTIME->num_solvent;
                                            unsigned int varj = tagj - this->LTIME->num_solvent;
                                            unsigned int vark = tagk - this->LTIME->num_solvent;

                                            unsigned int var1 = vari;
                                            unsigned int var3 = std::max({varj, vark});
                                            unsigned int var2 = std::min({varj, vark});

                                            unsigned int n = this->LTIME->num_colloid; 
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
                                    }
                                }
                                // Calculate angle between i and all previous neighbors of j

                                for (size_t idx_sj = 1; idx_sj < 20 ; ++idx_sj)
                                {
                                    if (h_previous_neighbor_list[idx_sj* tot_particles + idx_pj] != -2)
                                    {  
                                    
                                        unsigned int tagk = static_cast<unsigned int>(h_previous_neighbor_list[idx_sj* tot_particles + idx_pj]);
                                        Scalar3 pk;
                                        bool f3=false;
                                        for (int M = 0; M < (int)(this->m_pdata->getN()+ this->m_pdata->getNGhosts()); M++) 
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

                                            unsigned int vari = tagi - this->LTIME->num_solvent;
                                            unsigned int varj = tagj - this->LTIME->num_solvent;
                                            unsigned int vark = tagk - this->LTIME->num_solvent;

                                            // sort them so var3>var3>var1
                                            unsigned int var1 = varj;
                                            unsigned int var3 = std::max({vari, vark});
                                            unsigned int var2 = std::min({vari, vark});

                                            unsigned int n = this->LTIME->num_colloid; 
                                            unsigned int angle_index = (var1 * n*(n-1)/2) + (2*var2*n - var2*var2 + 2*var3 - 3*var2 -2)/2;

                                            angle_map[angle_index] = acos(cos_theta);

                                        }
                                    }
                                }


                                
                                // Calculate angle between i and all current neighbors of j

                                for (size_t idx_sj = 1; idx_sj < 20 ; ++idx_sj)
                                {
                                    if (h_current_neighbor_list.data[idx_sj* p_neighbor_pitch + j] != -2 && h_current_neighbor_list.data[idx_sj* p_neighbor_pitch + j] != h_tag.data[i])
                                        {
                                        unsigned int tagk = static_cast<unsigned int>(h_current_neighbor_list.data[idx_sj* p_neighbor_pitch + j]);
                                        Scalar3 pk;
                                        //bool f4=false;

                                        for (int M = 0; M < (int)(this->m_pdata->getN()+ this->m_pdata->getNGhosts()); M++) 
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

                                            unsigned int vari = tagi - this->LTIME->num_solvent;
                                            unsigned int varj = tagj - this->LTIME->num_solvent;
                                            unsigned int vark = tagk - this->LTIME->num_solvent;

                                            // sort them so var3>var3>var 
                                            unsigned int var1 = varj;
                                            unsigned int var3 = std::max({vari, vark});
                                            unsigned int var2 = std::min({vari, vark});
                                            

                                            unsigned int n = this->LTIME->num_colloid; 
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

            bool evaluated
                = eval.evalForceEnergyThermo(force_divr, force_divr_cons, 
                  //~ add virial_ind terms [PROCF2023]
                  cons_divr, disp_divr, rand_divr, sq_divr, cont_divr, 
                  //~
                  pair_eng, energy_shift);

            if (evaluated)
                {
                // compute the virial (FLOPS: 2)
                Scalar pair_virial[6];
                pair_virial[0] = Scalar(0.5) * dx.x * dx.x * force_divr_cons;
                pair_virial[1] = Scalar(0.5) * dx.x * dx.y * force_divr_cons;
                pair_virial[2] = Scalar(0.5) * dx.x * dx.z * force_divr_cons;
                pair_virial[3] = Scalar(0.5) * dx.y * dx.y * force_divr_cons;
                pair_virial[4] = Scalar(0.5) * dx.y * dx.z * force_divr_cons;
                pair_virial[5] = Scalar(0.5) * dx.z * dx.z * force_divr_cons;

                //~ compute the virial_ind [PROCF2023]
                Scalar virial_ind_prefix = Scalar(0.5) * dx.x * dx.y;
                Scalar pair_virial_ind[5];
                pair_virial_ind[0] = virial_ind_prefix * cons_divr;
                pair_virial_ind[1] = virial_ind_prefix * disp_divr;
                pair_virial_ind[2] = virial_ind_prefix * rand_divr;
                pair_virial_ind[3] = virial_ind_prefix * sq_divr;
                pair_virial_ind[4] = virial_ind_prefix * cont_divr;
                //~

                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fi += dx * force_divr;
                pei += pair_eng * Scalar(0.5);
                for (unsigned int l = 0; l < 6; l++)
                    viriali[l] += pair_virial[l];
                //~ add virial_ind [PROCF2023]
                for (unsigned int l = 0; l < 5; l++)
                    viriali_ind[l] += pair_virial_ind[l];
		//~

                // add the force to particle j if we are using the third law (MEM TRANSFER: 10
                // scalars / FLOPS: 8)
                if (third_law)
                    {
                    unsigned int mem_idx = j;
                    h_force.data[mem_idx].x -= dx.x * force_divr;
                    h_force.data[mem_idx].y -= dx.y * force_divr;
                    h_force.data[mem_idx].z -= dx.z * force_divr;
                    h_force.data[mem_idx].w += pair_eng * Scalar(0.5);
                    for (unsigned int l = 0; l < 6; l++)
                        h_virial.data[l * this->m_virial_pitch + mem_idx] += pair_virial[l];
                    //~ add virial_ind [PROCF2023] 
                    for (unsigned int l = 0; l < 5; l++)
                        h_virial_ind.data[l * this->m_virial_ind_pitch + mem_idx] += pair_virial_ind[l];
		    //~
                    }
                }
            }

        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = i;
        h_force.data[mem_idx].x += fi.x;
        h_force.data[mem_idx].y += fi.y;
        h_force.data[mem_idx].z += fi.z;
        h_force.data[mem_idx].w += pei;
        for (unsigned int l = 0; l < 6; l++)
            h_virial.data[l * this->m_virial_pitch + mem_idx] += viriali[l];
        //~ add virial_ind [PROCF2023] 
        for (unsigned int l = 0; l < 5; l++)
            h_virial_ind.data[l * this->m_virial_ind_pitch + mem_idx] += viriali_ind[l];
	    //~ 


        }
    if (m_K != 0.0){
    //MPI_Barrier(m_exec_conf->getMPICommunicator());
    #ifdef ENABLE_MPI
        unsigned int num_rank = this->LTIME->num_rank;
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

        }
    
   

    //~ add bond_calc [PROCF2023] 

    	this->LTIME->updatebondtime(timestep);
	
    //
    #endif
    for (int i = 0; i < (int)this->m_pdata->getN(); i++){
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        if (typei) {
                //Scalar kk =0;
                size_t nonzero_count = 0;
                unsigned int tagi = h_tag.data[i]; 
                for (size_t idx_i = 1; idx_i < 20 ; ++idx_i)
                {
                    // check if i has more than 1 neighbor
                    if (h_current_neighbor_list.data[idx_i* p_neighbor_pitch +i] != -2)
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
                        for (int M = 0; M < (int)(this->m_pdata->getN()+ this->m_pdata->getNGhosts()); M++)
                        {
                            if (h_tag.data[M] == taga) {
                            // Access the position of the particle with the matching tag
                                pa = make_scalar3(h_pos.data[M].x, h_pos.data[M].y, h_pos.data[M].z);
                                a = M; 
                                //f5=true;
                                break; 
                                }
                        }

                        //unsigned int typea = __scalar_as_int(h_pos.data[a].w);
                        // find the second neighbor b
                        for (size_t idx_b = (idx_a+1) ; idx_b < (nonzero_count+1); ++idx_b)
                        {
                            unsigned int tagb = static_cast<unsigned int>(h_current_neighbor_list.data[idx_b* p_neighbor_pitch +i]);
                            Scalar3 pb;
                            int b = -1;
                            //bool f6=false;
                            for (int N = 0; N < (int)(this->m_pdata->getN()+ this->m_pdata->getNGhosts()); N++)
                            {
                                if (h_tag.data[N] == tagb) {
                                    // Access the position of the particle with the matching tag
                                    pb = make_scalar3(h_pos.data[N].x, h_pos.data[N].y, h_pos.data[N].z);
                                    b = N;
                                    //f6 = true;
                                    break; 
                                    }

                            }
                            //unsigned int typeb = __scalar_as_int(h_pos.data[b].w);

                            

                            //if (tagb != taga  && taga != tagi && tagb != tagi && typea && typeb){}
                            
                                // Calculate angle index
                                unsigned int vari = tagi - this->LTIME->num_solvent;
                                unsigned int vara = taga - this->LTIME->num_solvent;
                                unsigned int varb = tagb - this->LTIME->num_solvent;

                                unsigned int var1 = vari;
                                unsigned int var3 = std::max({vara, varb});
                                unsigned int var2 = std::min({vara, varb});

                                unsigned int n = this->LTIME->num_colloid; 
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
                                Scalar tk = m_K * dth;


                                
                                // Calculate force magnitude
                                Scalar vab = -1.0 * tk * current_sin_theta;
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

                                // compute the energy, for each atom in the angle
                                Scalar angle_eng = (tk * dth) * Scalar(1.0 / 6.0);

                                Scalar angle_virial[6];
                                angle_virial[0] = Scalar(1. / 3.) * (ria.x * fia.x + rib.x * fib.x);
                                angle_virial[1] = Scalar(1. / 3.) * (ria.y * fia.x + rib.y * fib.x);
                                angle_virial[2] = Scalar(1. / 3.) * (ria.z * fia.x + rib.z * fib.x);
                                angle_virial[3] = Scalar(1. / 3.) * (ria.y * fia.y + rib.y * fib.y);
                                angle_virial[4] = Scalar(1. / 3.) * (ria.z * fia.y + rib.z * fib.y);
                                angle_virial[5] = Scalar(1. / 3.) * (ria.z * fia.z + rib.z * fib.z);

                                
                                // Update forces and virials for particle i, a and b
                                
                                if (a < (int)(this->m_pdata->getN()+ this->m_pdata->getNGhosts())) {
                                    h_force.data[a].x += fia.x;
                                    h_force.data[a].y += fia.y;
                                    h_force.data[a].z += fia.z;
                                    h_force.data[a].w += angle_eng;
                                    for (int l = 0; l < 6; l++)
                                        h_virial.data[l * this->m_virial_pitch + a] += angle_virial[l]; 
                                }
                                
                                if (i < (int)this->m_pdata->getN()) {
                                    h_force.data[i].x -= fia.x + fib.x; 
                                    h_force.data[i].y -= fia.y + fib.y;
                                    h_force.data[i].z -= fia.z + fib.z;
                                    h_force.data[i].w += angle_eng;
                                    for (int l = 0; l < 6; l++)
                                        h_virial.data[l * this->m_virial_pitch + i] += angle_virial[l]; 
                                }

                                if (b < (int)(this->m_pdata->getN()+ this->m_pdata->getNGhosts())) {
                                    h_force.data[b].x += fib.x;
                                    h_force.data[b].y += fib.y;
                                    h_force.data[b].z += fib.z;
                                    h_force.data[b].w += angle_eng;
                                    for (int l = 0; l < 6; l++)
                                        h_virial.data[l * this->m_virial_pitch + b] += angle_virial[l]; 
                                }                       
                            
                        }
                    }
                } 
            }
        }
  
 //F 
    }
      
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template<class evaluator>
CommFlags PotentialPairDPDThermo<evaluator>::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = CommFlags(0);

    // DPD needs ghost particle velocity
    flags[comm_flag::velocity] = 1;
    // DPD needs tags for RNG
    flags[comm_flag::tag] = 1;
    //~ add particle diameter [PROCF2023]
    flags[comm_flag::diameter]=1; 
    //~

    // Paniz
    flags[comm_flag::position] = 1;
    flags[comm_flag::net_force] = 1;
    flags[comm_flag::net_virial] = 1;
    flags[comm_flag::particle_n_list] = 1;

    flags |= PotentialPair<evaluator>::getRequestedCommFlags(timestep);

    return flags;
    }
#endif

namespace detail
    {
//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T>
void export_PotentialPairDPDThermo(pybind11::module& m, const std::string& name) {
    pybind11::class_<PotentialPairDPDThermo<T>,
                     PotentialPair<T>,
                     std::shared_ptr<PotentialPairDPDThermo<T>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<NeighborList>,
                            bool, Scalar>()) // Corrected initialization of constructor
        .def_property("bond_calc", &PotentialPairDPDThermo<T>::getBondCalcEnabled, &PotentialPairDPDThermo<T>::setBondCalcEnabled)
        .def_property("kT", &PotentialPairDPDThermo<T>::getT, &PotentialPairDPDThermo<T>::setT)
        .def_property("K", &PotentialPairDPDThermo<T>::getK, &PotentialPairDPDThermo<T>::setK);

    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // __POTENTIAL_PAIR_DPDTHERMO_H__
