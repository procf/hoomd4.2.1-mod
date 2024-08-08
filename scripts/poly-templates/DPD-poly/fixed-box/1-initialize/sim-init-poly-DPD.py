## serial simulation to create init.gsd
## for a DPD simulation with 1 solvent type and 1 colloid type
## NOTE: for variable colloid size (POLYDISPERSITY)
## NOTE: for a variable number of colloids and FIXED BOX SIZE
## (Rob Campbell)


######### MODULE LIBRARY
# Use HOOMD-blue
import hoomd
import hoomd.md # molecular dynamics
# Use GSD files
import gsd # provides Python API; MUST import sub packages explicitly (as below)
import gsd.hoomd # read and write HOOMD schema GSD files
# Maths
import numpy as np
import math
import random # psuedo-random number generator
# Other
import os # miscellaneous operating system interfaces

######### CUSTOM CLASSES
#N/A


#########  SIMULATION INPUTS
# Density parameters
phi = 0.20 # volume fraction
poly = 0.05 # standard deviation in size (AKA % polydispersity)
rho = 3 # number density (per unit volume)

# Simulation box size (fixed by L_X)
L_X = 30 
L_Y = L_X
L_Z = L_X 
V_total = L_X * L_Y * L_Z # total volume of simulation box (cube)

# Solvent particle details
m_S = 1 # solvent particle mass 
R_S = 0.5 # solvent particle radius 
V_Solvents = (1 - phi) * V_total # total volume of solvent
N_Solvents = math.floor(rho * V_Solvents) # total number of solvent particles

# Colloid particle details
R_C1 = 1  # 1st type colloid particle radius
V_C1 = (4./3.) * math.pi * R_C1 ** 3 # 1st type colloid particle volume (1 particle)
m_C1 = V_C1 * rho # 1st type colloid particle mass
V_Colloids_type1 = phi * V_total # total volume of type 1 colloids
N_C1 = round(V_Colloids_type1 / V_C1) # number of 1st type of colloid particles (INT)

# colloids totals NOTE: ASSUMES 1 colloid types
N_C = N_C1  # total number of colloids
V_Colloids = V_Colloids_type1 # total volume of all colloids

# Total number of particles in the simulation
N_total = int(N_Solvents + N_C)

# set random seed for repeatable random number generation
seed_value = 42
np.random.seed(seed_value)

######### SIMULATION
## Checks for existing files. If none are found, creates a new 
## random distribution of particles for use as initial simulation state. 

if os.path.exists('init-poly-DPD.gsd'):
  print("Initialization file already exists. No new files created.")
else:
  print("New initialization file is being created")
  ## Initialize a snapshot of the system
  snapshot = gsd.hoomd.Frame()
  snapshot.configuration.box = [L_X, L_Y, L_Z, 0, 0, 0] # create the sim box
  snapshot.particles.N=N_total # add all particles to the snapshot
  #print("Setting particle types")
  # set the particle types
  snapshot.particles.types = ['A','B']
  # assign particles to each type
  typeid = []
  typeid.extend([1]*N_C1)
  typeid.extend([0]*N_Solvents)
  snapshot.particles.typeid = typeid
  # set a mass and diameter for each particle type
  mass = []
  diameter = []
  # polydisperse options for colloid1
  if poly == 0:
    mass.extend([m_C1]*N_C1)
    diameter.extend([2.0*R_C1]*N_C1)
  elif poly != 0:  
    if N_C1 != 0:
      C1_radii = np.random.normal(loc=R_C1, scale=poly, size=N_C1)
      if round(np.mean(C1_radii)) != R_C1:
        print('WARNING: unexpected mean particle size: R_C1 = ' + str(R_C1) + 
              ' but polydispersity produces a mean size of ' + str(round(np.mean(C1_radii)))+' )')
      C1_vols = (4./3.) * math.pi * C1_radii ** 3
      C1_masses = C1_vols * rho
      mass.extend(C1_masses)
      diameter.extend(2.0*C1_radii)
    elif R_C1 == 0:
      mass.extend([m_C1]*N_C1)
      diameter.extend([2.0*R_C1]*N_C1)
  # solvent particle mass and diameters
  mass.extend([m_S]*N_Solvents)
  snapshot.particles.mass = mass
  diameter.extend([2.0*R_S]*N_Solvents)
  snapshot.particles.diameter = diameter

  # randomly distribute all the particles in 3D space
  pos_arr = np.zeros((N_total,3))
  pos_arr[:,0] = np.random.uniform(-0.5*L_X, 0.5*L_X, N_total)
  pos_arr[:,1] = np.random.uniform(-0.5*L_Y, 0.5*L_Y, N_total)	
  pos_arr[:,2] = np.random.uniform(-0.5*L_Y, 0.5*L_Y, N_total)

  snapshot.particles.position = pos_arr
  # save the snapshot of the initialized system
  with gsd.hoomd.open(name='init-poly-DPD.gsd', mode='w') as f:
    f.append(snapshot)

  print("New initialization file (init.gsd) created.\n")
  print("Seed for Random Number Generator: "+str(seed_value)) 
  print("Simulation volume: L_X = " + str(L_X) + ", L_Y = " + str(L_Y) + ", L_Z = " + str(L_Z))
  print("Volume fraction: " + str(phi) + "\n")

  if (poly == 0): 
    print("radius of colloid: " + str(R_C1) + "\n") 

  if poly != 0:
    print("Colloid polydispersity: " + str(poly))
    if N_C1 != 0:
      vol_C1 = (4/3)*math.pi*C1_radii**3
      print("Real colloid volume fraction: " + str(round(sum(vol_C1)/V_total,2)))
      print("expected mean colloid radius: " + str(R_C1))
      print("mean colloid radius: " + str(round(np.mean(C1_radii),2)) + "\n") 
    if N_C1 == 0:
      print("(no colloids, no polydisperse distribution generated for colloids) \n")

  print("Total number of particles: " + str(N_total))
  print("Total number of solvent particles: " + str(N_Solvents))
  print("Total number of colloid particles: " + str(N_C))
