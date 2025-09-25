# hoomd4.2.1-mod

Try out our DPD code! <br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/procf/hoomd4.2.1-mod/blob/main/RheoinformaticDPD.ipynb)

---

This repository contains the modified version of HOOMD-blue v4.2 used for colloid simulations in the Rheoinformatic group. It also includes: 
* [Installation instructions](/README.md#installation)
* [Background Reading](/background-reading) about how these simulations work
* [Example scripts](/scripts) for installing and running simulations on an HPC research cluster
* [Citation information](/citation-guide.md) for papers published using this simulation platform
* A [Changelog](/changelog.md) summarizing what was changed, and a full list of the files that were changed
* [Admin tasks](/admin/README.md) including pull requests, upgrading, adding changes, and compiling for Google Colab

Additional branches are available, tho they may be incomplete:
- branch "hoomd4.2_w_wall": modifications for flat and sinusoidal walls
- branch "no_shear": clean version of original DPDMorse and virial_ind tracking without shear, bond-tracking, or other mods.

[Last Updated: June 2024]

Contact: Rob Campbell (campbell.r@northeastern.edu)

-----------------
For any questions, or to help with modifications, contact Rob.

To-Do:
- [ ] update Lifetime.h @Rob
- [ ] add wall mods @Josh @Rob
- [ ] add bond rigidity @Paniz
-----------------

## Installation

To install hoomd4.2.1-mod on Explorer:

Login and move to the location you want to put hoomd4.2-mod
```bash
ssh your-username@login.explorer.northeastern.edu
```
```bash
cd /projects/props/your-name/software/
```
Make a new directory for hoomd4.2.1-mod
```bash
mkdir hoomd4.2.1-mod
```
Clone the repository
```bash
git clone git@github.com:procf/hoomd4.2.1-mod.git
```
Run the install-hoomd-mod-4.2.1 script with sbatch
```bash
cd /scripts/install-update/ && sbatch install-hoomd-mod-4.2.1
```
<br>
