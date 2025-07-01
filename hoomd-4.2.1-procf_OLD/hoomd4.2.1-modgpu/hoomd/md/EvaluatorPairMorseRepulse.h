//~ ########## Created by the Rheoinformatic research group ##########
////~ HOOMD-blue:
//// Copyright (c) 2009-2022 The Regents of the University of Michigan.
//// Part of HOOMD-blue, released under the BSD 3-Clause License.
////~
////~ This file:
/////~ Written by Rob Campbell (2024) based on EvaluatorPairMorse.h


// ########## Modified by Rheoinformatic //~ [RHEOINF] ##########

#ifndef __PAIR_EVALUATOR_MORSE_H__
#define __PAIR_EVALUATOR_MORSE_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairMorseRepulse.h
    \brief Defines the pair evaluator class for Morse potential
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
//! Class for evaluating the Morse pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ.

    <b>Morse specifics</b>

    EvaluatorPairMorseRepulse evaluates the sum of two functions, the Morse potential:
    \f[ V_{\mathrm{Morse}}(r) = D_0 \left[ \exp \left(-2\alpha \left(r - r_0\right) \right)
                                           -2\exp \left(-\alpha \left(r-r_0\right) \right)  \right]
    \f]

    and one of two different types of repulsion: electrostatic repulsion (a la DLVO), or Yukawa repulsion:
    \f[ V_{\mathrm{electrostatic}}(r) = Z \frac{ (a_i a_j)}{r_0} \exp \left( -kappa_e \left(r - r_0\right) \right)
    \f]

    \f[ V_{\mathrm{yukawa}}(r) = \varepsilon \frac{ \exp \left( -\kappa_y r \right) }{r} \f]    

    NOTE: The added repulsion will decrease the depth of the attractive potential well. If you want to achieve the 
          same attraction as in a pure Morse case with the addition of repulsion, then you must adjust the parameters
          to be sure the potential energy curve has the correct form
*/

class EvaluatorPairMorseRepulse //~ change name to MorseRepulse
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar D0;
        Scalar alpha;
        bool Erep; //~ add bool for Electrostatics on/off [RHEOINF]
        Scalar Z; //~ add Z param [RHEOINF]
        Scalar kappa_e; //~ add kappa_e param [RHEOINF]
        bool Yrep; //~ add bool for Yukawa repulsion on/off [RHEOINF]
        Scalar epsilon; //~ add epsilon param [RHEOINF]
        Scalar kappa_y; //~ add kappa_y param [RHEOINF]
        Scalar r0;
        Scalar f_contact; //~ add f_contact param [RHEOINF]
        bool scaled_D0; //~ add scaled_D0 param [RHEOINF]


        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        // CUDA memory hints
        void set_memory_hints() const { }
#endif

#ifndef __HIPCC__
        param_type() : D0(0), alpha(0), Erep(false), Z(0), kappa_e(0), Yrep(false), epsilon(0), kappa_y(0), r0(0), f_contact(0), scaled_D0(false) { } //~ add params [RHEOINF]

        param_type(pybind11::dict v, bool managed = false, bool Erep = false, bool Yrep = false)
            {
            D0 = v["D0"].cast<Scalar>();
            alpha = v["alpha"].cast<Scalar>();
            this->Erep = Erep; //~ add boolean check for Electrostatic repulsion [RHEOINF]
            Z = v["Z"].cast<Scalar>(); //~ add Z param [RHEOINF]
            kappa_e = v["kappa_e"].cast<Scalar>(); //~ add kappa_e param [RHEOINF]
            this->Yrep = Yrep; //~ add boolean check for Yukawa repulsion [RHEOINF]
            epsilon = v["epsilon"].cast<Scalar>(); //~ add epsilon param [RHEOINF]
            kappa_y = v["kappa_y"].cast<Scalar>(); //~ add kappa_y param [RHEOINF]
            r0 = v["r0"].cast<Scalar>();
            f_contact = v["f_contact"].cast<Scalar>(); //~ add f_contact param [RHEOINF]
            this->scaled_D0 = scaled_D0; //~ add scaled_D0 param [RHEOINF
            }

        param_type(Scalar d, Scalar a, bool Er, Scalar z, Scalar ke, bool Yr, Scalar e, Scalar ky, Scalar r, Scalar f, bool sD, bool managed = false) //~ add params [RHEOINF]

            {
            D0 = d;
            alpha = a;
            Erep = Er; //~ add boolean check for Electrostatic repulsion [RHEOINF]
            Z = z; //~ add Z param [RHEOINF]
            kappa_e = ke; //~ add kappa_e param [RHEOINF]
            Yrep = Yr; //~ add boolean check for Yukawa repulsion [RHEOINF]
            epsilon = e; //~ add epsilon param [RHEOINF]
            kappa_y = ky; //~ add kappa_y param [RHEOINF]
            r0 = r;
            f_contact = f; //~ add f_contact param [RHEOINF]
            scaled_D0 = sD; //~ add scaled_D0 param [RHEOINF]
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["D0"] = D0;
            v["alpha"] = alpha;
            v["Erep"] = Erep; //~ add boolean check for Electrostatic repulsion [RHEOINF]
            v["Z"] = Z; //~ add Z param [RHEOINF]
            v["kappa_e"] = kappa_e; //~ add kappa_e param [RHEOINF]
            v["Yrep"] = Yrep; //~ add boolean check for Yukawa repulsion [RHEOINF]
            v["epsilon"] = epsilon; //~ add epsilon param [RHEOINF]
            v["kappa_y"] = kappa_y; //~ add kappa_y param [RHEOINF]
            v["r0"] = r0;
            v["f_contact"] = f_contact; //~ add f_contact param [RHEOINF]
            v["scaled_D0"] = scaled_D0; //~ add scaled_D0 param [RHEOINF] 
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairMorseRepulse(Scalar _rsq, Scalar _radcontact, unsigned int _pair_typeids[2], Scalar _rcutsq, const param_type& _params) //~ add radcontact and pair typeids [RHEOINF]
        : rsq(_rsq), rcutsq(_rcutsq), D0(_params.D0), alpha(_params.alpha), Erep(_params.Erep), Z(_params.Z), kappa_e(_params.kappa_e), Yrep(_params.Yrep), epsilon(_params.epsilon), kappa_y(_params.kappa_y), r0(_params.r0), f_contact(_params.f_contact), scaled_D0(_params.scaled_D0) //~ add params [RHEOINF]
        {
        typei = _pair_typeids[0]; //~ add typei [RHEOINF]
        typej = _pair_typeids[1]; //~ add typej [RHEOINF]
        }

    //~ add diameter [RHEOINF] 
    DEVICE static bool needsDiameter()
        {
        return true;
        }
    //! Accept the optional diameter values
    /*! \param di Diameter of particle i
        \param dj Diameter of particle j
    */
    DEVICE void setDiameter(Scalar di, Scalar dj)
        {
        diameter_i = di;
        diameter_j = dj;
        }
    //~

    //! Morse doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }
    //! Accept the optional charge values.
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param pair_eng Output parameter to write the computed pair energy
        \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the
       cutoff \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are
       performed in PotentialPair.

        \return True if they are evaluated or false if they are not because we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        //~ Add radsum from passed diameters [RHEOINF] 
        Scalar radsum = 0.5 * (diameter_i + diameter_j); 
        //~ Scale attraction strength by particle size is scaled_D0 is true
        if (scaled_D0)
          {
          D0 = D0 * (0.5*radsum);
          }   
        //~ 

        // compute the force divided by r in force_divr
        if (rsq < rcutsq)
            {
            Scalar r = fast::sqrt(rsq);
            Scalar Exp_factor = fast::exp(-alpha * (r - radsum));

            //~ check if contact force is provided [RHEOINF]
            if (f_contact != 0.0)
                {
                //~ if particles overlap (r < radsum) apply contact force
                if(r < radsum)force_divr = f_contact * (Scalar(1.0) - (r-radsum)) * pow((Scalar(0.50)*radsum),3) / r;

                //~ if particles do not overlap
                else{
                    //~ calculate Morse force as normal
                    force_divr = Scalar(2.0) * D0 * alpha * Exp_factor * (Exp_factor - Scalar(1.0)) / r;

                    //~ but still include contact force within 0.001 dist of colloid-colloid contact 
                    Scalar Del_max = Scalar(0.001); //~ 0.001 or 0.01
                    if(r<(radsum+Del_max))force_divr += f_contact * pow((Scalar(1.0) - (r-radsum)/Del_max), 3) * pow((Scalar(0.50)*radsum),3) / r;
                    } 

                }

            //if f_contact=0.0
            else{
                //~ calculate force as normal (use Morse repulsion)
                force_divr = Scalar(2.0) * D0 * alpha * Exp_factor * (Exp_factor - Scalar(1.0)) / r;
                }

            //~ add repulsion
            if (Erep) {
                // add electrostatic repulsion to the force divided by r in force_divr
                Scalar rcutinv = fast::rsqrt(rcutsq);
                Scalar rcut = Scalar(1.0) / rcutinv;
                if (r < rcut && kappa_e != 0)
                    {
                    Scalar radprod = (Scalar(0.5)*diameter_i) * (Scalar(0.5)*diameter_j); 
                    Scalar rmds = r - radsum;
                    Scalar radsuminv = Scalar(1.0) / radsum;

                    Scalar exp_val_e = fast::exp(-kappa_e * rmds);

                    Scalar forcerep_divr = kappa_e * radprod * radsuminv * Z * exp_val_e / r;
                    force_divr += forcerep_divr;
                    pair_eng += r * forcerep_divr / kappa_e;
                    }
                }

            if (Yrep) {            
                Scalar rinv = fast::rsqrt(rsq);
                Scalar r2inv = Scalar(1.0) / rsq;
                Scalar exp_val_y = fast::exp(-kappa_y * r);
            
                force_divr += epsilon * exp_val_y * r2inv * (rinv + kappa_y);
                pair_eng += epsilon * exp_val_y * rinv;
                }


            pair_eng = D0 * Exp_factor * (Exp_factor - Scalar(2.0));
            //~ force_divr = Scalar(2.0) * D0 * alpha * Exp_factor * (Exp_factor - Scalar(1.0)) / r; //~ move this into overlap check [RHEOINF]

            //~ turn off energy shift [RHEOINF]
            //~ form must change because of new repulsion; but also shouldn't be needed because of contact force [RHEOINF]
            //if (energy_shift)
            //    {
            //    Scalar rcut = fast::sqrt(rcutsq);
            //    Scalar Exp_factor_cut = fast::exp(-alpha * (rcut - r0));
            //    pair_eng -= D0 * Exp_factor_cut * (Exp_factor_cut - Scalar(2.0));
            //    }
            //~
            return true;
            }
        else
            return false;
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
        return std::string("morse");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;    //!< Stored rsq from the constructor
    Scalar radcontact;//!< Stored contact-distance from the constructor [RHEOINF]
    unsigned int pair_typeids;//!< Stored pair typeIDs from the constructor [RHEOINF]
    unsigned int typei;//!<~ Stored typeID of particle i from the constructor [RHEOINF]
    unsigned int typej;//!<~ Stored typeID of particle j from the constructor [RHEOINF]
    Scalar rcutsq; //!< Stored rcutsq from the constructor
    Scalar diameter_i;//!<~ add diameter_i [RHEOINF]
    Scalar diameter_j;//!<~ add diameter_j [RHEOINF]
    Scalar D0;     //!< Depth of the Morse potential at its minimum
    Scalar alpha;  //!< Controls width of the potential well
    bool Erep;        //!<~ Electrostatic on/off bool [RHEOINF]
    Scalar Z;         //!<~ (scaled) surface charge [RHEOINF]
    Scalar kappa_e;   //!<~ Debye (screeneing) length [RHEOINF]
    bool Yrep;        //!<~ Yukawa on/off bool [RHEOINF]
    Scalar epsilon;   //!<~ Yukawa energy factor [RHEOINF]
    Scalar kappa_y;   //!<~ Yukawa scaling factor [RHEOINF]
    Scalar r0;     //!< Offset, i.e., position of the potential minimum //~ comment out [RHEOINF]
    Scalar f_contact; //!< Contact force magnitude, for resolving overlap [RHEOINF]
    bool scaled_D0;   //!<~ on/off bool for scaling D0 by particle size [RHEOINF]
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_MORSE_H__
