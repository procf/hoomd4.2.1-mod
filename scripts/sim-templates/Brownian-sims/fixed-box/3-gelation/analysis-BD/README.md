# Analysis And Plotting for DPD Colloid Sims (HOOMD-blue v.3.1-mod)

To use these scripts: 

1. Copy this analysis folder into the directory where a sim has been run.
```bash
cd path-to-GSD-sim-data
cp -r path-to-this-folder analysis
cd analysis
``` 

2. Update the filepaths sourced to in `package-install`, `compile-module`, `analyze`, and `plotting/plotter` to match your installation of HOOMD-blue.

3. Compile the fortran modules:
```bash
sbatch compile-module
```
NOTE: you will get several warnings about using a deprecated version of NumPy API and incompatible pointers, unused modules, and unused variables associated with the poresize calculation; this is normal. If you are having trouble generating the `fortranmod.cpython-38-x86_64-linux-gnu.so` file, try running the commands individually in an srun session instead of sbatch 


4. Make sure the required Python pacakges (pandas, pyvoro, networkx) are installed by running the `package-install` script

4. Update the sim parameters in each file to match the parameters of the sim you are analyzing.

5. Select which analyses to run: 
- View the `sim-analysis-DPD.py` `sim-networkCSV-DPD.py` and `sim-voronoi-DPD.py` files to select which analyses to run (comment out any unnecessary analyses in the "RUN CHECKS ON A SIMULATION" section).
- View the `analyze` file and select which python analysis scripts to run (comment out any unnecessary scripts)

6. Once the `data` folder has been generated and all scripts are completed without errors (no content in the error file and all desired analysis stated as completed in the out file), cd into `plotting`

7. Load the python module in Explorer and plot each analysis individually (adjusting the plot-style as desired) OR batch-create png files for all desired analyses (select which results to plot by commenting out unnecessary analysis from the `plotter` script and then run `plotter`)
```bash
sbatch plotter
```  
