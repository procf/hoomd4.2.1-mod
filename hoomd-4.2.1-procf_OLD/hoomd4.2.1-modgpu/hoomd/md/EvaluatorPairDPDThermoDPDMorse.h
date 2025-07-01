//~ ########## Created by the Rheoinformatic research group ##########
//~ HOOMD-blue:
// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.
//~
//~ This file:
///~ Written by Mohammad (Nabi) Nabizadeh and Dr. Deepak Mangal
///~ Documentation by Rob Campbell (2022)


#ifndef __PAIR_EVALUATOR_DPDMORSE_H__
#define __PAIR_EVALUATOR_DPDMORSE_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"


/*! \file EvaluatorPairDPDMorseThermo.h
 \brief Defines the pair evaluator class for different particle interactions:
		   the DPD conservative potential : solvent-solvent and solvent-colloid
		   the DPD Morse potential : colloid-colloid
		   the short range lubrication (squeezing) force : colloid-colloid 
				(when too close for solvent particles to fit between them)
		   the semi-hard potential (contact force) : colloid-colloid (when particles overlap)
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {
//! Class for evaluating the DPD Thermostat pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ


    <b>DPD Thermostat, Conservative, Morse, Lubrication, and Contact Force specifics</b>

    There are two functions, EvaluatorPairDPDThermoMorse::evalForceAndEnergy and 
    the thermostat potential EvaluatorPairDPDThermoMorse::evalForceEnergyThermo


    EvaluatorPairDPDThermoMorse::evalForceAndEnergy evaluates different functions depending on 
    the details of the particles involved.

    [A] If one of the particles has a radius of zero OR the bond-dissociation energy (D0) is zero--

    EvaluatorPairDPDThermoMorse::evalForceAndEnergy evaluates the DPD Conservative force function:
    \f[ V_{\mathrm{DPD-C}}(r) = A_{\mathrm{0}} \cdot \left( r_{\mathrm{cut}} - \left( (r_{ij} 
                        - a_{i} - a_{j} \right) \right)
    - \frac{1}{2} \cdot \frac{A_{\mathrm{0}}}{r_{\mathrm{cut}}} \cdot \left(r_{\mathrm{cut}}^2 
    - \left( (r_{ij} - a_{i} - a_{j} \right)^2 \right)\f]
    
    [B] If both particles have a non-zero radius AND the bond-dissociation energy (D0) is non-zero--

    EvaluatorPairDPDThermoMorse::evalForceAndEnergy evaluates the Morse function:
    \f[ V_{\mathrm{Morse}}(r) = D_{\mathrm{0}} \cdot \left( \exp(-2 \cdot \alpha \cdot \left( (r_{ij} 
                        - a_{i} - a_{j} \right) - r_{\mathrm{0}})
    - 2 \cdot \exp(-\alpha  \cdot \left( (r_{ij} - a_{i} - a_{j} \right) - r_{\mathrm{0}}) \right)\f]


    The Morse potential does not need charge. Nine parameters are specified and
    stored in a Scalar. \a A0 is placed in \a params.A0, \a D0 is placed in \a params.D0, 
    \a alpha is placed in \a params.alpha, \a r0 is placed in \a params.r0, 
    \a eta is placed in \a params.eta, \a f_contact is placed in \a params.f_contact, 
    \a a1 is placed in \a params.a1, \a a2 is placed in \a params.a2, 
    \a rcut is placed in \a params.rcut
    and \a scaled_D0 is placed in \a params.scaled_D0


    The thermostat potential EvaluatorPairDPDThermoMorse::evalForceEnergyThermo evaluates includes 
    the Conservative Force, Dissipative Force, Random Force, Hydrodynamic (squeezing) lubrication force, 
    Morse force, and contact force--evaluating the function:
    \f{eqnarray*}
    F =   F_{\mathrm{C}}(r) + F_{\mathrm{R,ij}}(r_{ij}) +  F_{\mathrm{D,ij}}(v_{ij}) 
    + F_{\mathrm{Hydrodynamics}}(r) + F{\mathrm{Morse}}(r) + F{\mathrm{Contact}}(r)\f}

    \f{eqnarray*}
    F_{\mathrm{C}}(r) = & A_{\mathrm{0}} \cdot  w(r_{ij}) \\
    F_{\mathrm{R, ij}}(r_{ij}) = & - \theta_{ij}\sqrt{3} \sqrt{\frac{2k_b\gamma T}{\Delta t}}\cdot
    w(r_{ij})  \\
    F_{\mathrm{D, ij}}(r_{ij}) = & - \gamma w^2(r_{ij})\left( \hat r_{ij} \circ v_{ij} \right)  \\
    \f}
    F_{\mathrm{Hydrodynamics}}(r_{ij}) = & F_{\mathrm{squeezing}}(r_{ij}) = & a_{sq} \left( v_ij \cdot e_ij \right) \cdot e_ij\\
    F_{\mathrm{Morse}}(r_{ij}) = & 2 \cdot D_{\mathrm{0}} \cdot \alpha \cdot \left( \exp(-2 \cdot 
    \alpha \cdot \left( (r_{ij} - a_{i} - a_{j} \right) - r_{\mathrm{0}}) - 2 \cdot \exp(\alpha \cdot 
    \left( (r_{ij} - a_{i} - a_{j} \right) - r_{\mathrm{0}})\\
    F_{\mathrm{contact}}(r_{ij}) = & f_contact \cdot  \left( 1 - \left( r_{ij} - a_{i} 
    - a_{j} \right) \right) \cdot \left( frac{a_{i} + a_{j}}{2} \right)^3\\

    \f{eqnarray*}
    w(r_{ij}) = &\left( 1 - r/r_{\mathrm{cut}} \right)  & r < r_{\mathrm{cut}} \\
                     = & 0 & r \ge r_{\mathrm{cut}} \\
    \f}
    where \f$\hat r_{ij} \f$ is a normalized vector from particle i to particle j, \f$ v_{ij} = v_i
    - v_j \f$, and \f$ \theta_{ij} \f$ is a uniformly distributed random number in the range [-1, 1].


    EvaluatorPairDPDThermoMorse::evalForceEnergyThermo also evaluates different functions 
    depending on the details of the particles involved.

    [A] If one of the particles has a radius of zero OR the bond-dissociation energy (D0) is zero--

    EvaluatorPairDPDThermoMorse::evalForceEnergyThermo evaluates the DPD Conservative force function and
    F_{\mathrm{Morse}}(r_{ij}) = 0
    F_{\mathrm{contact}}(r_{ij}) = 0


    [B] If both particles have a non-zero radius AND their position causes the radii to overlap AND f_contact!=0
    EvaluatorPairDPDThermoMorse::evalForceEnergyThermo evaluates the contact force function and
    F_{\mathrm{C}}(r) = 0
    F_{\mathrm{Morse}}(r_{ij}) = 0

    If f_contact is 0, then either Morse (if the bond-disoociation energy D0 is non-zero) or conservative force
    (if D0=0) is used to resolve overlaps

    [C] If both particles have a non-zero radius AND the bond-dissociation energy (D0) is non-zero
    EvaluatorPairDPDThermoMorse::evalForceEnergyThermo evaluates the Morse function and
    F_{\mathrm{C}}(r) = 0
    F_{\mathrm{contact}}(r_{ij}) = 0
 
 
    The Thermostat potential does not need charge. Nine parameters are specified and
    stored in a Scalar. \a A0 is placed in \a params.A0, \a D0 is placed in \a params.D0, 
    \a alpha is placed in \a params.alpha, \a r0 is placed in \a params.r0, 
    \a eta is placed in \a params.eta, \a f_contact is placed in \a params.f_contact, 
    \a a1 is placed in \a params.a1, \a a2 is placed in \a params.a2, 
    \a rcut is placed in \a params.rcut
    and \a scaled_D0 is placed in \a params.scaled_D0

    These are related to the standard lj parameters sigma and epsilon by:
    - \a A0 = \f$ A \f$
    - \a gamma = \f$ \gamma \f$

*/
class EvaluatorPairDPDThermoDPDMorse
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
	Scalar A0;
	Scalar gamma;
        Scalar D0;
	Scalar alpha;
	Scalar r0;
	Scalar eta;
	Scalar f_contact;
	Scalar a1;
	Scalar a2;
	Scalar rcut;
	bool scaled_D0;
        Scalar sys_kT;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        // CUDA memory hints
        void set_memory_hints() const { }
#endif
#ifndef __HIPCC__
        param_type() : A0(0), gamma(0), D0(0), alpha(0), r0(0), eta(0), f_contact(0), a1(0), a2(0), rcut(0), scaled_D0(false), sys_kT(0) { }

        param_type(pybind11::dict v, bool managed = false, bool scaled_D0 = false)
            {
	    A0 = v["A0"].cast<Scalar>();
	    gamma = v["gamma"].cast<Scalar>();
            D0 = v["D0"].cast<Scalar>();
	    alpha = v["alpha"].cast<Scalar>();
	    r0 = v["r0"].cast<Scalar>();
	    eta = v["eta"].cast<Scalar>();
	    f_contact = v["f_contact"].cast<Scalar>();
	    a1 = v["a1"].cast<Scalar>();
	    a2 = v["a2"].cast<Scalar>();
	    rcut = v["rcut"].cast<Scalar>();
	    this->scaled_D0 = scaled_D0;
            sys_kT = v["sys_kT"].cast<Scalar>();
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
	    v["A0"] = A0;
	    v["gamma"] = gamma;
            v["D0"] = D0;
	    v["alpha"] = alpha;
	    v["r0"] = r0;
	    v["eta"] = eta;
	    v["f_contact"] = f_contact;
	    v["a1"] = a1;
	    v["a2"] = a2;
	    v["rcut"] = rcut;
	    v["scaled_D0"] = scaled_D0;
	    v["sys_kT"] = sys_kT;
            return v;
            }
#endif
        }
#ifdef SINGLE_PRECISION
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _pair_typeids the typeIDs of the interacting particles [RHEOINF]
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairDPDThermoDPDMorse(Scalar _rsq, Scalar _radsum, unsigned int _pair_typeids[2], Scalar _rcutsq, const param_type& _params) //~ add radsum, pair_typeIDs [RHEOINF]
        : rsq(_rsq), radsum(_radsum), rcutsq(_rcutsq), A0(_params.A0), gamma(_params.gamma), D0(_params.D0), alpha(_params.alpha), 
	r0(_params.r0), eta(_params.eta), f_contact(_params.f_contact), a1(_params.a1), a2(_params.a2), rcut(_params.rcut), scaled_D0(_params.scaled_D0), sys_kT(_params.sys_kT) // add radsum, scaled_D0, kT [RHEOINF]
        {
        typei = _pair_typeids[0]; //~ add typei [RHEOINF]
        typej = _pair_typeids[1]; //~ add typej [RHEOINF]  
        }

    //! Set i and j, (particle tags), and the timestep
    DEVICE void
    set_seed_ij_timestep(uint16_t seed, unsigned int i, unsigned int j, uint64_t timestep)
        {
        m_seed = seed;
        m_i = i;
        m_j = j;
        m_timestep = timestep;
        }

    //! Set the timestep size
    DEVICE void setDeltaT(Scalar dt)
        {
        m_deltaT = dt;
        }

    //! Set the velocity term
    DEVICE void setRDotV(Scalar dot)
        {
        m_dot = dot;
        }

    //! Set the temperature
    DEVICE void setT(Scalar Temp)
        {
        m_T = Temp;
        }
    
    //~ add diameter [RHEOINF]
    //~ NOTE: for some reason diameters are not passed to this file correctly... which is why we use radcontact instead
    DEVICE static bool needsDiameter()
        {
        return false;
        }
    //! Accept the optional diameter values
    /*! \param di Diameter of particle i
        \param dj Diameter of particle j
    */
    DEVICE void setDiameter(Scalar di, Scalar dj) { }
    //    {
    //    diameter_i = di;
    //    diameter_j = dj;
    //    }
    //~

    //! DPDMorse doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }
    //! Accept the optional charge values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy using the conservative force only
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param pair_eng Output parameter to write the computed pair energy
        \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the
       cutoff \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are
       performed in PotentialPair.

        \return True if they are evaluated or false if they are not because we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {

        //~ but remove solvent diameters (should be treated as zero) using typeid
        //~ ASSUMING SOLVENTS WERE INITIALIZED WITH R_S=0.5 
        if (typei == 0)
          {
          radsum -= Scalar(0.5);
          }
        if (typej == 0)
          {
          radsum -= Scalar(0.5);
          }

        //~ Scale attraction strength by particle size
        if (scaled_D0)
          {
          D0 = D0 * (0.5*radsum);
          }   
        //~ 

        Scalar rinv = fast::rsqrt(rsq);
        // convert to h_ij (surface-surface distance)
	Scalar r = (Scalar(1.0) / rinv) - radsum;
	//Scalar rcut = fast::sqrt(rcutsq) - radsum;
	Scalar rcutinv = Scalar(1.0) / rcut;
	Scalar w_factor = (Scalar(1.0) - r * rcutinv);
        // compute the force divided by r in force_divr
        if(r < rcut)
	   {
	   if(typei == 0 || typej == 0) //~ switch to using typeID
	   //if(a1 == Scalar(0.0) or a2 == Scalar(0.0))
	      {
	      force_divr = A0 * w_factor * rinv;
	      pair_eng = A0 * (rcut - r) - Scalar(1.0 / 2.0) * A0 * rcutinv * (rcut * rcut - r * r);
	      }
	   else
	      {
	      if(D0 != Scalar(0.0))
	         {
	         Scalar Exp_factor = fast::exp(-alpha * (r - r0));
	         pair_eng = D0 * Exp_factor * (Exp_factor - Scalar(2.0));
	         force_divr = Scalar(2.0) * D0 * alpha * Exp_factor * (Exp_factor - Scalar(1.0)) * rinv;
	         //~ energy shift is ignored: This was legacy from using the LJ potential as a template.
		 //~ if(energy_shift)
		 //~   {
		 //~   Scalar Exp_factor_cut = fast::exp(-alpha * (rcut - r0));
		 //~   pair_eng -= D0 * Exp_factor_cut * (Exp_factor_cut - Scalar(2.0));
		 //~   }
		 }
	      else
		 {
		 force_divr = A0 * w_factor * rinv;
		 pair_eng = A0 * (rcut - r) - Scalar(1.0 / 2.0) * A0 * rcutinv * (rcut * rcut - r * r);
		 }
	      }
	   return true;
	   }
	else
	   return false; 
        }

    //! Evaluate the force and energy using the thermostat
    /*! \param force_divr Output parameter to write the computed total force divided by r.
        \param force_divr_cons Output parameter to write the computed sum of: Morse force OR conservative force,
       dissipative force, and squeezing (lubrication) divided by r.
        \param cons_divr Output parameter to write the computed conservative OR Morse force divided by r.
        \param disp_divr Output parameter to write the computed dissipative force divided by r.
        \param rand_divr Output parameter to write the computed random force divided by r.
        \param sq_divr Output parameter to write the computed squeezing (lubrication) force divided by r.
        \param cont_divr Output parameter to write the computed contact force divided by r.
        \param pair_eng Output parameter to write the computed pair energy \param energy_shift
       Ignored. DPD always goes to 0 at the cutoff. \note There is no need to check if rsq < rcutsq
       in this method. Cutoff tests are performed in PotentialPair.

        \note The portion of the force \b excluding random and contact force components must be output to \a force_divr_cons 
       so that the virial may be computed correctly.

        \return True if they are evaluated or false if they are not because we are beyond the cutoff
    */
    DEVICE bool evalForceEnergyThermo(Scalar& force_divr,
                                      Scalar& force_divr_cons,
                                      Scalar& cons_divr,
                                      Scalar& disp_divr,
                                      Scalar& rand_divr,
                                      Scalar& sq_divr,
                                      Scalar& cont_divr,
                                      Scalar& pair_eng,
                                      bool energy_shift)
        {

        //~ but remove solvent diameters (should be treated as zero) using typeid
        //~ ASSUMING SOLVENTS WERE INITIALIZED WITH R_S=0.5 
        if (typei == 0)
          {
          radsum -= Scalar(0.5);
          }
        if (typej == 0)
          {
          radsum -= Scalar(0.5);
          }

        //~ Scale attraction strength by particle size
        if (scaled_D0)
          {
          D0 = D0 * (0.5*radsum);
          }
        //~ 
	
	// get 1/r_ij
        Scalar rinv = fast::rsqrt(rsq);
	// convert r_ij (center-center) to h_ij (surface-surface)
        Scalar r = (Scalar(1.0) / rinv) - radsum;
        Scalar rcutinv = Scalar(1.0) / rcut;

        // compute the force divided by r in force_divr
        if(r < rcut)
	   {
	   Scalar w_factor = (Scalar(1.0) - r * rcutinv);

	   // if at least one particle has a radius = 0 (solvent-solvent or solvent-colloid)
	   if(typei == 0 || typej == 0) //~ switch to using typeID
	   //if(a1 == Scalar(0.0) or a2 == Scalar(0.0))
	      {

	      // conservative DPD
	      cons_divr = A0 * w_factor * rinv;

	      // Drag term
              disp_divr = -gamma * m_dot * rinv * w_factor * w_factor * rinv;


	      unsigned int m_oi, m_oj;
	      // initialize the RNG
	      if (m_i > m_j)
	          {
	          m_oi = m_j;
	          m_oj = m_i;
	          }
	      else
	          {
	          m_oi = m_i;
	          m_oj = m_j;
	          }
 
	      hoomd::RandomGenerator rng(
	          hoomd::Seed(hoomd::RNGIdentifier::EvaluatorPairDPDThermo, m_timestep, m_seed),
	          hoomd::Counter(m_oi, m_oj));

	      // Generate a single random number theta
	      Scalar theta = hoomd::UniformDistribution<Scalar>(-1, 1)(rng);

	      // Random force
	      rand_divr = fast::rsqrt(m_deltaT / (m_T * gamma * Scalar(6.0))) * w_factor * theta * rinv;

              // conservative energy only
	      pair_eng = A0 * (rcut - r) - Scalar(1.0/2.0) * A0 * rcutinv * (rcut * rcut - r*r);
	      }

	   // if both particles do NOT both have radius = 0 (colloid-colloid)
	   else
	      { 

	      Scalar w_factor = (Scalar(1.0) - r * rcutinv);
              Scalar Exp_factor = fast::exp(-alpha * (r - r0));

	      // if particles overlap
	      if(r <= Scalar(0.0))
	         {
		 // resolve overlap with CONTACT FORCE, if a contact force is provided [RHEOINF]
    		 if (f_contact != 0.0)
                    {
    	            cont_divr = f_contact * (Scalar(1.0) - r) * pow((Scalar(0.50)*radsum),3) * rinv;
    	            }
	         // if no contact force provided, resolve overlap with other forces [RHEOINF]
	         else
    	            {
    	            // if D0 is provided, use this to calculate Morse repulsion [RHEOINF]
    	            if (D0 != 0.0)
        	       {
        	       //Scalar Exp_factor = fast::exp(-alpha * (r - r0));
        	       cons_divr = Scalar(2.0) * D0 * alpha * Exp_factor * (Exp_factor - Scalar(1.0)) * rinv;
        	       pair_eng = D0 * Exp_factor * (Exp_factor - Scalar(2.0));
        	       }
    	            // if no D0 is provided use repulsion of D0=10*sys_kT
                    // kT defaults to 0.1 but can be changed with the system
                    else
                       {
                       Scalar repulse_D0 = 10*sys_kT;
                       cons_divr = Scalar(2.0) * repulse_D0 * alpha * Exp_factor * (Exp_factor - Scalar(1.0)) * rinv;
                       pair_eng = repulse_D0 * Exp_factor * (Exp_factor - Scalar(2.0));
                       // use conservative force
                       //force_divr = A0 * w_factor * rinv;
                       //pair_eng = A0 * (rcut - r) - Scalar(1.0 / 2.0) * A0 * rcutinv * (rcut * rcut - r * r);
        	       }
    	            }
	         }

	      // if no overlap
	      else
	         {

		 // if there is a Morse potential given in the simulation
	         if( D0 != Scalar(0.0))
	            {
	            //Scalar Exp_factor = fast::exp(-alpha * (r - r0));

		    // use Morse force NOT conservative force
	            cons_divr = Scalar(2.0) * D0 * alpha * Exp_factor * (Exp_factor - Scalar(1.0)) * rinv;

		    // Morse potential
	            pair_eng = D0 * Exp_factor * (Exp_factor - Scalar(2.0));		
		    //~ energy shift is ignored: This was legacy from using the LJ potential as a template.
	            //~if (energy_shift)
	            //~    {
	            //~    Scalar Exp_factor_cut = fast::exp(-alpha * (rcut - r0));
	            //~    pair_eng -= D0 * Exp_factor_cut * (Exp_factor_cut - Scalar(2.0));
	            //~    }
	            }

		 // if there is NOT a Morse potential given in the simulation
	         else 
	            {
		    // use conservative force
	            cons_divr = A0 * w_factor * rinv;
	            pair_eng = A0 * (rcut - r) - Scalar(1.0/2.0) * A0 * rcutinv * (rcut * rcut - r*r);
	            }

		 // for ALL OTHER forces in the case of NO overlaps

		 // Drag term
		 disp_divr = -gamma * m_dot * rinv * w_factor * w_factor * rinv;

		 // minimum distance for using the lubrication approximation (squeezing/lubrication force)
	         Scalar del_min = Scalar(0.001);
		 // maximum distance for using contact force
	         Scalar Del_max = Scalar(0.001);
		 // drag coefficient asq = mu_ij
		 Scalar asq = Scalar(0.0);

		 // if using the lubrication approximation
	         if (r <= del_min)
		    {
	            asq = Scalar(1.178225) * eta * radsum * radsum / del_min; //3.0*pi/8.0=1.178225
		    }
	         else
		    {
	            asq = Scalar(1.178225) * eta * radsum * radsum / r;
		    }

		 // hydrodynamic squeezing) force
	         sq_divr = -asq * m_dot * rinv * rinv;
		 // update total force
		 force_divr_cons = force_divr;

	         // if using Contact force
	         if(r <= Del_max)
		    {
                    if (f_contact != 0.0)
                      {
  		      // Contact force
	              cont_divr = f_contact * pow((Scalar(1.0) - r/Del_max), 3) * pow((Scalar(0.50)*radsum),3) * rinv;
                      }
		    }

	         unsigned int m_oi, m_oj;
	         // initialize the RNG
	         if (m_i > m_j)
	             {
	             m_oi = m_j;
	             m_oj = m_i;
	             }
	         else
	             {
	             m_oi = m_i;
	             m_oj = m_j;
	             }
	         hoomd::RandomGenerator rng(
	             hoomd::Seed(hoomd::RNGIdentifier::EvaluatorPairDPDThermo, m_timestep, m_seed),
	             hoomd::Counter(m_oi, m_oj));

		 // Generate a single random number theta
	         Scalar theta = hoomd::UniformDistribution<Scalar>(-1, 1)(rng);

		 // Random force
		 rand_divr = fast::rsqrt(m_deltaT / (m_T * (asq+gamma) * Scalar(6.0))) * theta * w_factor * rinv;
	         }
	      }

	   // Caluclate the total forces
           force_divr_cons = cons_divr + disp_divr + sq_divr;
           force_divr = force_divr_cons + rand_divr + cont_divr;

	   return true;
	   }
	else
	   {
           return false;
	   }
	}

    DEVICE Scalar evalPressureLRCIntegral()
        {
        return 0;
        }

    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return 0;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("dpdmorse");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;          //!< Stored rsq from the constructor
    Scalar radsum;       //!< Stored contact-distance from the constructor [RHEOINF]
    unsigned int pair_typeids;//!< Stored pair typeIDs from the constructor [RHEOINF]
    unsigned int typei;  //!<~ Stored typeID of particle i from the constructor [RHEOINF]
    unsigned int typej;  //!<~ Stored typeID of particle j from the constructor [RHEOINF]
    Scalar rcutsq;       //!< Stored rcutsq from the constructor
    // parameters for potential extracted from the params by constructor
    Scalar diameter_i;  //!< the diameter of particle i [RHEOINF]
    Scalar diameter_j;  //!< the diameter of particle j [RHEOINF]
    Scalar A0;		 //!< the conservative force scaling parameter
    Scalar gamma;        //!< the viscous dissipation parameter
    Scalar D0;           //!< the depth of the Morse potential well
    Scalar alpha;	 //!< the width of the Morse potential wel
    Scalar r0;		 //!< the position of the center of the Morse potential well
    Scalar eta;		 //!< the background viscosity
    Scalar f_contact;	 //!< the contact force scaling parameter
    Scalar a1;		 //!< the radius of particle i
    Scalar a2;		 //!< the radius of particle j
    Scalar rcut;	 //!< the cut-off radius for particle interaction
    bool scaled_D0;	 //!< on/off bool for scaling D0 by particle size 
    Scalar sys_kT;	 //!< the system temperature (typically 0.1 or 1.0)
    uint16_t m_seed;     //!< User set seed for thermostat PRNG
    unsigned int m_i;    //!< index of first particle (should it be tag?).  For use in PRNG
    unsigned int m_j;    //!< index of second particle (should it be tag?). For use in PRNG
    uint64_t m_timestep; //!< timestep for use in PRNG
    Scalar m_T;          //!< Temperature for Themostat
    Scalar m_dot;        //!< Velocity difference dotted with displacement vector
    Scalar m_deltaT;     //!<  timestep size stored from constructor
    };

#undef DEVICE

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_DPDMORSE_H__
