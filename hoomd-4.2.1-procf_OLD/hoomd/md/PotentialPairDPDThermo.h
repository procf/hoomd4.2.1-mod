// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// ########## Modified by Rheoinformatic //~ [RHEOINF] [RHEOINF] ##########

#ifndef __POTENTIAL_PAIR_DPDTHERMO_H__
#define __POTENTIAL_PAIR_DPDTHERMO_H__

#include "PotentialPair.h"
#include "hoomd/Variant.h"
#include "Lifetime.h" //~ add Lifetime.h [RHEOINF]

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
                           bool bond_calc); //~ add bond_calc [RHEOINF]
    //! Destructor
    virtual ~PotentialPairDPDThermo() {};

    //! Set the temperature
    virtual void setT(std::shared_ptr<Variant> T);

    //! Get the temperature
    virtual std::shared_ptr<Variant> getT();

    //~! Check the bond_calc flag [RHEOINF]
    void setBondCalcEnabled(bool bond_calc)
	{
	m_bond_calc = bond_calc;
	}
    bool getBondCalcEnabled()
	{
	return m_bond_calc;
	}
    //~

    //~ add Lifetime [RHEOINF] 
    std::shared_ptr<Lifetime> LTIME;
    //~

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    virtual CommFlags getRequestedCommFlags(uint64_t timestep);
#endif

    protected:
    std::shared_ptr<Variant> m_T; //!< Temperature for the DPD thermostat

    bool m_bond_calc; //= false;      //~!< bond_calc flag (default false) [RHEOINF]

   //ofstream DiameterFile; //~ print diameters [RHEOINF]

    //! Actually compute the forces (overwrites PotentialPair::computeForces())
    virtual void computeForces(uint64_t timestep);
    };

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
*/
template<class evaluator>
PotentialPairDPDThermo<evaluator>::PotentialPairDPDThermo(std::shared_ptr<SystemDefinition> sysdef,
                                                          std::shared_ptr<NeighborList> nlist,
                                                          bool bond_calc) //~ add bond_calc [RHEOINF]
    : PotentialPair<evaluator>(sysdef, nlist), m_bond_calc(bond_calc) //~ add bond_calc [RHEOINF]
    {
    //~ add bond_calc flag [RHEOINF]
    if(m_bond_calc)
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

/*! Gets the temperature variant*/
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
    //~ access particle diameter [RHEOINF] 
    ArrayHandle<Scalar> h_diameter(this->m_pdata->getDiameters(),
                                   access_location::host,
                                   access_mode::read);
    //~

    //~ access particle charges [RHEOINF]
    ArrayHandle<Scalar> h_charge(this->m_pdata->getCharges(),
                                   access_location::host,
                                   access_mode::read);
    //~

    // force arrays
    ArrayHandle<Scalar4> h_force(this->m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(this->m_virial, access_location::host, access_mode::overwrite);
    //~ add virial_ind [RHEOINF]
    ArrayHandle<Scalar> h_virial_ind(this->m_virial_ind, access_location::host, access_mode::overwrite);
    //~

    const BoxDim box = this->m_pdata->getBox();
    //~ get box dims and shear rate [RHEOINF]
    Scalar3 L2 = box.getL();
    uchar3 per_ = box.getPeriodic();
    Scalar shear_rate = this->m_SR;
    //~
    ArrayHandle<Scalar> h_ronsq(this->m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(this->m_rcutsq, access_location::host, access_mode::read);

    // need to start from a zero force, energy and virial
    memset((void*)h_force.data, 0, sizeof(Scalar4) * this->m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * this->m_virial.getNumElements());
    //~ add virial_ind [RHEOINF] 
    memset((void*)h_virial_ind.data, 0, sizeof(Scalar) * this->m_virial_ind.getNumElements());
    //~

    uint16_t seed = this->m_sysdef->getSeed();

    // for each particle
    for (int i = 0; i < (int)this->m_pdata->getN(); i++)
        {
        // access the particle's position, velocity, and type (MEM TRANSFER: 7 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        Scalar3 vi = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);

        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const size_t head_i = h_head_list.data[i];

        // sanity check
        assert(typei < this->m_pdata->getNTypes());

        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar pei = 0.0;
        Scalar viriali[6];
        for (unsigned int l = 0; l < 6; l++)
            viriali[l] = 0.0;
        //~ initialize the current virial_ind to zero [RHEOINF]
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

            //~ calculate shear rate [RHEOINF]
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

            //~ store the typeIDs of the current pair [RHEOINF]
            unsigned int pair_typeids[2] = {0, 0};
            pair_typeids[0] = typei;
            pair_typeids[1] = typej;
            //~

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            //~ calculate the center-center distance equal to particle-particle contact (AKA r0) [RHEOINF]
            //~ the calculation is only used if there is polydispersity
            Scalar radcontact = 0.0;
            {
            radcontact = Scalar(0.5) * (h_diameter.data[i] + h_diameter.data[j]);
            }
            //~ Print or save diameters to check that values are correct 
            //std::cout << "contact = " << radcontact << std::endl;
            //string diameter_file = "potential_pair_diameters.csv";
            //DiameterFile.open("potential_pair_diameters.csv", ios::out | ios::app);
            //DiameterFile << h_tag.data[i] << "," << h_tag.data[j] << "," << typei << "," << typej << "," << h_diameter.data[i] << "," << h_diameter.data[j] << "," << radcontact << "\n";
            //DiameterFile.close();
            //~

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
            //~ add virial_ind terms [RHEOINF] 
            Scalar cons_divr = Scalar(0.0);
            Scalar disp_divr = Scalar(0.0);
            Scalar rand_divr = Scalar(0.0);
            Scalar sq_divr = Scalar(0.0);
            Scalar cont_divr = Scalar(0.0);
	    //~
            Scalar pair_eng = Scalar(0.0);  
            evaluator eval(rsq, radcontact, pair_typeids, rcutsq, param); //~ add radcontact, pair_typeids for polydispersity [RHEOINF] 

            // Special Potential Pair DPD Requirements
            const Scalar currentTemp = m_T->operator()(timestep);

            // set seed using global tags
            unsigned int tagi = h_tag.data[i];
            unsigned int tagj = h_tag.data[j];
            eval.set_seed_ij_timestep(seed, tagi, tagj, timestep);
            eval.setDeltaT(this->m_deltaT);
            eval.setRDotV(rdotv);
            eval.setT(currentTemp);

	    //~ add bond_calc [RHEOINF]
	    if(m_bond_calc)
		{
           	if(typei && typej) // if both are NOT zero (solvents are type zero)
               	    {
   	       	    Scalar rsq_root = fast::sqrt(rsq) - Scalar(0.5)*(h_diameter.data[i]+h_diameter.data[j]);
   	            if(rsq_root < Scalar(0.10)) // assumes the cut-off is 0.1
   	                {
   	                unsigned int var1 = tagi - this->LTIME->num_solvent;
   	                unsigned int var2 = tagj - this->LTIME->num_solvent;
   	                if(var1 > var2)
   	                    {
   	                    var1 = tagj - this->LTIME->num_solvent;
   	                    var2 = tagi - this->LTIME->num_solvent;
   	                    }
   	                unsigned int bond_index = (this->LTIME->num_colloid * var1) - (var1 * (var1+1) / 2) + var2 - var1 - 1;
   	                if(rsq_root < Scalar(0.08))
			    {
   	                    this->LTIME->Bond_check[bond_index] = 2;
			    }
   	                else
			    {
   	                    this->LTIME->Bond_check[bond_index] = 1;
			    }
   	                }
		    }
		}
	    //~

            bool evaluated
                = eval.evalForceEnergyThermo(force_divr, force_divr_cons, 
                  //~ add virial_ind terms [RHEOINF]
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

                //~ compute the virial_ind [RHEOINF]
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
                //~ add virial_ind [RHEOINF]
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
                    //~ add virial_ind [RHEOINF] 
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
        //~ add virial_ind [RHEOINF] 
        for (unsigned int l = 0; l < 5; l++)
            h_virial_ind.data[l * this->m_virial_ind_pitch + mem_idx] += viriali_ind[l];
	//~
        }

    //~ add bond_calc [RHEOINF] 
    if(m_bond_calc)
	{
    	this->LTIME->updatebondtime(timestep);
    	if(timestep%10000 == 0) // assumes the recording period is 10000
	    {
            this->LTIME->writeBondtime();
	    }
	}
    //~

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
    //~ add particle diameter [RHEOINF]
    flags[comm_flag::diameter]=1; 
    //~

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
template<class T> void export_PotentialPairDPDThermo(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialPairDPDThermo<T>,
                     PotentialPair<T>,
                     std::shared_ptr<PotentialPairDPDThermo<T>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, bool>()) //~ add bool for bond_calc [RHEOINF]
        .def_property("bond_calc",  
		&PotentialPairDPDThermo<T>::getBondCalcEnabled, &PotentialPairDPDThermo<T>::setBondCalcEnabled)  //~ add bond_calc [RHEOINF]
        .def_property("kT", &PotentialPairDPDThermo<T>::getT, &PotentialPairDPDThermo<T>::setT);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // __POTENTIAL_PAIR_DPDTHERMO_H__
