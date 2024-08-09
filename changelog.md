# Changelog with full list of changed files


If the box next to a filename is NOT checked, that means the modifications have not been added

File lists are formatted as: `folder/`; file

* [Core Mods](/changelog.md#core-modifications) : Contact Force, Lubrication Force, track virial components (Nabi and Deepak)
* [Shear Rate](/changelog.md#shear-rate) : Add shear rate for regular particles (does not include rigid bodies) (Deepak)
* [Polydispersity](/changelog.md#polydispersity) : Track and use particle radii for surface-surface distance (h_ij) calculations, option to scale D0 by particle size to mimic size-dependent depletion (Rob)
* [On/Off Contact Force](/changelog.md#on-off-contact-force) : Add the ability to remove contact force and replace it with Morse repulsion (Sasha)
* [Bond tracking](/changelog.md#bond-tracking) : Track bond formation and breaking (Nabi, Deepak, and Rob)
* [Walls](/changelog.md#walls) : Wall options: flat or converging diverging (Josh)
* [Pressure-driven flow](/changelog.md#pressure-driven-flow) : make sure charge is available for body force (Deepak)
* [Morse with Repulsion](/changelog.md#morse-with-repulsion) : Add two repulsive options to Morse, Electrostatic repulsion and Yukawa repulsion (Rob)
* [Asakura-Oosawa Potential](/changelog.md#asakura-oosawa-potential) : Add AO Potential (might be incorrect calc?) (Rob)
* [HPMC](/changelog.md#hpmc) : enable compilation with HPMC on (see important notes in description) (Rob)

## Core Modifications
Contact Force, Lubrication Force, track virial components (Nabi and Deepak)
- **DPDMorse**: Add a new method for evaluating the pair-forces between two particles called "DPDMorse." This method calculates the correct combination of forces for each pair of particles as described in the [background reading on DPD for Colloids](/background-reading/2-DPD-for-Colloids-18pg.pdf); i.e., the standard DPD forces plus the Morse potential, hydrodynamics (the squeezing force AKA lubrication force), and a contact force for resolving semi-hard colloid-colloid particle overlaps. (filename: `EvaluatorPairDPDThermoDPDMorse.h`)
- **contact force**: Add a contact force to Brownian simulations for resolving semi-hard colloid-colloid particle overlaps
- **virial_ind**: Add the ability to track the "independent virials" (AKA the virial_ind) for each particle pair. This is the contribution of each of the individual forces in the virial (conservative, dissipative, random, Morse, lubrication, and contact) in addition to the total virial component of the pressure tensor

* [x] `hoomd/`
	* [x] Communicator.cc : **virial_ind**
	* [x] Communicator.h : **virial_ind**
	* [x] ForceCompute.cc : **virial_ind**
	* [x] ForceCompute.h : **virial_ind** 
	* [x] Integrator.cc : **virial_ind**	
	* [x] `md/`
		* [x] CMakeLists.txt : **set new file (EvaluatorPairDPDThermoDPDMorse.h)**
		* [x] compute.py : **virial_ind**
		* [x] ComputeThermo.cc : **virial_ind**
		* [x] ComputeThermo.h : **virial_ind**
		* [x] ComputeThermoTypes.h : **virial_ind**
		* [x] EvaluatorPairDPDThermoDPD.h : **virial_ind (redefine forces to include all virial_ind terms)**
		* [x] **[ADD NEW FILE]** EvaluatorPairDPDThermoDPDMorse.h
		* [x] EvaluatorPairDPDThermoLJ.h : **virial_ind**
		* [x] EvaluatorPairMorse.h : **contact force**
		* [x] module-md.cc : **add void/export for DPDMorse()**	
		* [x] `pair`
			* [x] \_\_init\_\_.py **call DPDMorse**
			* [x] pair.py : **define DPDMorse Python interface, add Morse contact force param**
		* [x] PotentialPairDPDThermo.h : **virial_ind**
	* [x] ParticleData.cc : **virial_ind**
	* [x] ParticleData.h : **virial_ind**


## Shear Rate 
Adding shear rate for regular particles (does not include rigid bodies) (Deepak)<br>
NOTE: we do not remove the streamline velocity from the kinetic component of the stress tensor (ComputeForces.cc). This is generally okay because above a Peclet Number of 1 Brownian forces do not typically matter so much for the behavior of the system (and our results are realistic, which supports this choice), but in the future you may want to include Brownian force in the integrator for shear and remove streamline velocity from x-velocity used in the kinetic component
- **flow velocity** add the flow velocity for calculating shear rate
- **y-boundary velocity** update velocity when wrapping particles across the Y boundary (gradient direction)
- **shear rate** add the shear rate variable
- **iostream** import package for Variant class
- **Oscillatory** create variant class for t_ramp oscillatory shear (legacy)
- **Sinusoidal** create variant class for sinusoidal oscillation (legacy)
- **Cosinusoidal** create variant class for cosinusoidal oscillation (use this one)
- **BoxShear** and **box_shear.py** new classes for shearing
- **virialxyi_ind** track xyi component of virial separately in Brownian shear sims

* [x] `hoomd/`
	* [x] BoxResizeUpdater.h : **flow velocity (vinf, cur_vel)**
	* [x] BoxResizeUpdater.cc : **flow velocity (vinf, cur_vel), y-boundary velocity**
	* [x] **[ADD NEW FILE]** BoxShearUpdater.h
	* [x] **[ADD NEW FILE]** BoxShearUpdater.cc
	* [x] Communicator.cc : **shear rate (SR), y-boundary velocity**
	* [x] Communicator.h : **shear rate (SR)**
	* [x] ForceCompute.h : **shear rate (SR)**
	* [x] HOOMDMPI.h : **shear rate (SR)** (add uint4)
	* [x] CmakeList.txt : **add BoxShearUpdater.h, BoxShearUpdater.cc**
	* [x] Integrator.h : **flow velocity (vinf), shear rate (SR)**
	* [x] Integrator.cc : **flow velocity (vinf), shear rate (SR), Box_Dim**
	* [x] module.cc : **BoxShear**
	* [x] `md/`
		* [x] FIREEnergyMinimizer.cc : **flow velocity (vinf)**
		* [x] FIREEnergyMinimizer.h : **flow velocity (vinf)**
		* [x] IntegrationMethodTwoStep.h : **shear rate (SR)**
		* [x] IntegrationMethodTwoStep.cc : **shear rate (SR)**
		* [x] IntegratorTwoStep.cc : **flow velocity (vinf)**
		* [x] IntegratorTwoStep.h : **flow velocity (vinf), shear rate (SR)**
		* [x] integrate.py : **flow velocity (vinf) and default to 0**
		* [x] PotentialPair.h : **virialxyi_ind**
		* [x] PotentialPairDPDThermo.h **shear rate (SR), virialxyi_ind**
                * [x] `test/`
			* [x] test_communication.cc : **dummy flow velocity (vinf)**
			* [x] test_fire_energy_minimizer.cc : **dummy flow velocity (vinf)**
		* [x] TwoStepBD.cc : **flow velocity (vinf, cur_vel), y-boundary velocity**
		* [x] TwoStepConstantVolume.cc : **box dims, shear rate (SR)**
		* [x] TwoStepLangevin.cc : **flow velocity (vinf), shear rate (SR)**
	* [x] `mpcd/`
		* [x] Integrator.h : **flow velocity (vinf)**
		* [x] Integrator.cc : **flow velocity (vinf)**
	* [x] `test/`
		* [x] test_system.cc : **dummy flow velocity (vinf)**
	* [x] `update/`
		* [x] CmakeList.txt : **add box_shear.py**
		* [x] \_\_init\_\_.py : **add BoxShear**
		* [x] **[ADD NEW FILE]** box_shear.py
		* [x] box_resize.py : **flow velocity (vinf)**
	* [x] Variant.h : **iostream, Oscillatory, Sinusoidal, Cosinusoidal**
	* [x] Variant.cc : **Oscillatory, Sinusoidal, Cosinusoidal**
	* [x] variant.py : **Oscillatory, Sinusoidal, Cosinusoidal**


## Polydispersity
Track and use polydisperse particle radii for surface-surface distance (h_ij) calculations, and scale D0 by particle size to mimic size-dependent attraction in depletion gels (Rob)
- **scaled_D0**: scale D0 by particles size ((radius_i + radius_j)/2) for AO-style multimodal depletion potential; activated by an optional boolean (true/false) flag
- **radcontact**: passes the radsum from PotentialPairDPDThermo to DPDMorse; must be added to all Evaluator files but not used elsewhere (for some reason diameter did not work for the DPDMorse Evaluator)
- **diameter**: adds diameter back (removed by HOOMD-blue devs between hoomdv3 and hoomdv4)
- **typeIDs**: tracks particle typeID, used to reset solvent radius to zero in DPD force calcs
- **a1 and a2 deprecated**: default values for a1 and a2 are provided. This means you can run new scripts without a1 and a2, and you can still run old scripts that specified a1 and a2 as a DPDMorse parameter in Python; however, a1 and a2 values are NOT used in the code anymore. Radii are always read from the simulation GSD file.
- **alpha**: scale the BD drag/diffusion coefficient by particle size (this was standard behavior in HOOMD-blue v3 but was removed with the removal of diameter in HOOMD-blue v4) 

* [x] `hoomd/`
    * [x] `example_plugins/`
        * [x] `pair_plugin/`
            * [x] EvaluatorPairExample.h : **radcontact, diameter, typeIDs**
    * [x] `md/`
        * [x] EvaluatorPairBuckingham.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairDLVO.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairDPDThermoDPD.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairDPDThermoDPDMorse.h : **radcontact, diameter, typeIDs, scaled_D0**
        * [x] EvaluatorPairDPDThermoLJ.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairEwald.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairExpandedGaussian.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairExpandedLJ.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairExpandedMie.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairForceShiftedLJ.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairFourier.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairGauss.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairLJ.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairLJ0804.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairLJ1208.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairLJGauss.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairMie.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairMoliere.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairMorse.h : **radcontact, diameter, typeIDs, scaled_D0**
        * [x] EvaluatorPairOPP.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairReactionField.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairTWF.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairTable.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairYukawa.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorPairZBL.h : **radcontact, diameter, typeIDs**
        * [x] EvaluatorWalls.h : **radcontact, diameter, typeIDs**
	* [x] `methods/`
		* [x] methods.py : **alpha**
        * [x] `pair/`
            * [x] pair.py : **optional scaled_D0 [Morse, DPDMorse], a1 and a2 deprecated**
        * [x] PotentialPair.h : **radcontact, track diameter, typeIDs**
        * [x] PotentialPairAlchemical.h : **radcontact, typeIDs**
        * [x] PotentialPairDPDThermo.h : **radcontact, diameter, typeIDs**
	* [x] TwoStepBD.cc : **alpha**
	* [x] TwoStepLangevin.cc : **alpha**
	* [x] TwoStepLangevinBase.cc : **alpha**
	* [x] TwoStepLangevinBase.h : **alpha**


## On-Off Contact Force
Add the ability to remove contact force and replace it with Morse repulsion (Sasha)
- **f_contact=0**: when f_contact = 0, removes contact force (and uses built-in Morse repulsion, or DPD Conservative Force if D0=0)

* [x] `hoomd/`
	* [x] `md/`
        	* [x] EvaluatorPairDPDThermoDPDMorse.h : **f_contact=0**
        	* [x] EvaluatorPairMorse.h : **f_contact=0**


## Bond Tracking
Track bond formation and breaking (Nabi, Deepak, and Rob)
- **Lifetime**: Add the ability to track the formation and breaking of any bond between two particles, and calculate and record the bond lifetime (filename: `Lifetime.h`)
- **bond_calc**: Create a flag to turn the bond-lifetime calculation On or Off (True/False) *NOTE: lifetime calculation is slow and was not able to be sped up with MPI*

* [x] `hoomd/`
	* [x] `md/`
		* [x] CMakeLists.txt : **set new file (Lifetime.h)**
		* [x] **[ADD NEW FILE]** Lifetime.h		
		* [x] `pair`
			* [x] pair.py : **add bond_calc flag to Pair and DPDMorse**
		* [x] PotentialPairDPDThermo.h : **Lifetime, bond_calc, get particle diameter (for bond calc)**


## Pressure-driven flow
- **charge** make sure charge is available for body force
* [x] `hoomd/`
	* [x] `md/`
		* [x] PotentialPairDPDThermo.h : **charge**

## Walls
Wall options: flat or converging diverging (Josh)
- **walls**: Add walls in the Y direction that are either flat or sinusoidal in shape, including correct bounce-back conditions for solvents interacting with the wall. Add the ability for opposite sinusoidal walls to be aligned (in phase) or offset by a phase shift and for the "sides" to have flat walls.
* [ ] `hoomd/`
	* [ ] `md/`
		* [ ] `methods/`
			* [ ] methods.py **wall amp freq, default amp=0 freq=0**
		* [ ] TwoStepConstantVolume.cc : **shear rate (SR), y-boundary velocity, wall amp freq and bounceback**
		* [ ] TwoStepConstantVolume.h **wall amp and freq**


## Morse with Repulsion
Adding two repulsive options to Morse
- **MorseRepulse**: Add the option for using two different repulsive potentials in combination with the Brownian Morse potential: Electrostatic repulsion (a la DLVO) or Yukawa repulsion. 
 * [x] `hoomd/`
	* [x] `md/`
		* [x] CMakeLists.txt : **set new file (EvaluatorPairMorseRepulse.h)**
		* [x] **[ADD NEW FILE]** EvaluatorPairMorseRepulse.h (the Morse potential with Electrostatic or Yukawa repulsion)
		* [x] module-md.cc : **add void/export for DPDMorseRepulse()** 
		* [x] `pair`
			* [x] \_\_init\_\_.py **call MorseRepulse**
			* [x] pair.py : **call MorseRepulse**

		
## Asakura-Oosawa Potential 
Adding AO Potential (might be incorrect calc?) (Rob)
- **DPDAO**: Add the Asakura-Oosawa potential as an alternative to Morse potential. This is an laternate method for evaluating the pair-forces between two particles that calculates the correct combination of forces for each pair of particles as described in the [background reading on DPD for Colloids](/background-reading/2-DPD-for-Colloids-18pg.pdf), but replaces the Morse Potential with the Asakura-Oosawa (AO) potential; i.e., the standard DPD forces plus the AO potential, hydrodynamics (the squeezing force AKA lubrication force), and a contact force for resolving semi-hard colloid-colloid particle overlaps.
 * [ ] `hoomd/`
	* [ ] `md/`
		* [ ] CMakeLists.txt : **set new file (EvaluatorPairDPDThermoDPDAO.h)**
		* [ ] **[ADD NEW FILE]** EvaluatorPairDPDThermoDPDAO.h (the Asakura-Oosawa potential)
		* [ ] `pair`
			* [ ] \_\_init\_\_.py **call DPDAO**
			* [ ] pair.py : **call DPDAO**


## HPMC
Fix HPMC integrator to allow compilation when HPMC is enabled (for Hard Particle Monte Carlo)
NOTE: **[IMPORTANT]** Our modifications are not designed for use with Hard Particle Monte Carlo (HPMC) simulations. HPMC is NOT included in the installation by default. If you want to run HPMC simulations with any of our modifications, you should do additional work to integrate the mods first.<br>
Additionally, HPMC is an active area of development in HOOMD-blue. Because our software is based on HOOMD-blue v4.2.1 many important HPMC features are not included (because they had not yet been implemented/corrected/integrated in v4.2.1).
- **vinf**: Add vinf to the HPMC integrators so they can be constructed from the modified Integrator class. This does NOT enable shear flow in HPMC sims, it only allows the HPMC integrator to correctly compile.
 * [ ] `hoomd/`
	* [ ] `hpmd/`
		* [ ] IntegratorHPMC.cc : **vinf**
		* [ ] IntegratorHPMC.h : **vinf**
		* [ ] IntegratorHPMCMono.h : **vinf**
		* [ ] IntegratorHPMCMonoNEC.h : **vinf**
		* [ ] module_convex_polygon.cc : **vinf**
		* [ ] module_convex_polyhedron.cc : **vinf**
		* [ ] module_convex_spheropolyhedron.cc : **vinf**
		* [ ] module_ellipsoid.cc : **vinf**
		* [ ] module_faceted_ellipsoid.cc : **vinf**
		* [ ] module_polyhedron.cc : **vinf**
		* [ ] module_simple_polygon.cc : **vinf**
		* [ ] module_sphere.cc : **vinf**
		* [ ] module_spheropolygon.cc : **vinf**
		* [ ] module_sphinx.cc : **vinf**
		* [ ] module_union_convex_polyhedron.cc : **vinf**
		* [ ] module_union_faceted_ellipsoid.cc : **vinf**
		* [ ] module_union_sphere.cc : **vinf**



