# Variational Umbrella Seeding
by Willem Gispen, Jorge R. Espinosa, Eduardo Sanz, Carlos Vega, and Marjolein Dijkstra

This code repository accompagnies our paper ["Variational Umbrella Seeding for Calculating Nucleation Barriers"](https://doi.org/10.1063/5.0204540) accepted for publication in the Journal of Chemical Physics.
This code is also available from [github.com/MarjoleinDijkstraGroupUU](github.com/MarjoleinDijkstraGroupUU).

![](/vus-overview.svg)

## Abstract
> In this work, we introduce Variational Umbrella Seeding, a novel technique for computing nucleation barriers. This new method, a refinement of the original seeding approach, is far less sensitive to the choice of order parameter for measuring the size of a nucleus. Consequently, it surpasses seeding in accuracy, and Umbrella Sampling in computational speed. We test the method extensively and demonstrate excellent accuracy for crystal nucleation of nearly hard spheres and of two distinct models of water: mW and TIP4P/ICE. This method can easily be extended to calculate nucleation barriers for homogeneous melting, condensation, and cavitation.

## How to use this software


1. Install LAMMPS and required python packages
    * Compile LAMMPS in 'shared' mode and `make install-python` to build the python package wheel i.e. `.whl`-file. See the [LAMMPS documentation](https://docs.lammps.org/Python_install.html) for help. This code was developed and tested with the 3 November 2022 release of LAMMPS.
    * If you want to do parallel simulations, make sure to compile a parallel build of LAMMPS. To parallelize simulations, simply replace `python` by `mpirun -np {np} python` in the examples below, where `{np}` is the number of processors you want to use. 
    * Use conda to install the required python packages, notably [freud](https://freud.readthedocs.io/en/latest/) [1]: `conda env create -f freud-lammps.yml`. We recommend using miniconda for this.
    * Activate the conda environment: `conda activate freud-lammps`
    * Install LAMMPS in the conda environment: `pip install` the lammps `.whl`-file
    * The analysis scripts are presented as [Jupyter Notebooks](https://jupyter.org/). To open them, type `jupyter notebook` in the terminal. Alternatively, if you prefer, you can use the notebooks in their python script form, i.e. the same filenames but then `.py` instead of `.ipynb`.
    * `unzip results.zip`

2. Test the hybrid Monte Carlo scheme on a bulk phase.

    Use the python script `python hmc_npt_bias.py`. This script loads a LAMMPS script that defines the interactions in your system. We have included our LAMMPS scripts, e.g. `scripts/simulation/mW.nve` for the mW model. Start by simulating a bulk phase, for example a bulk liquid phase, and measure some bulk properties such as the density and potential energy. Comparing these against a normal molecular dynamics simulations is a good test that you chose an appropriate timestep and trajectory length for the hybrid Monte Carlo scheme. For example, to test the hybrid Monte Carlo scheme on the bulk liquid for mW: use the following commands:
     * `python hmc_npt_bias.py -model mW -P 0.0 -T 215.1 -target_size 0 -integrator npt -n_steps 250 -dt 2.0 --rand_vel`
     * `python hmc_npt_bias.py -model mW -P 0.0 -T 215.1 -target_size 0 -integrator npt -n_steps 250 -dt 2.0 --no-rand_vel`
    
    Please see the python script `hmc_npt_bias.py` for the precise meaning of the command line arguments. Essentially, this performs two simulations of a bulk mW liquid phase at 215.1K, one where the velocities are resampled after every 250 timesteps, and one where they are not resampled. Generally, the command line arguments we used can be found in the `args.dat` files that are included with every simulation.

3. **Seeding**

    * Estimate the critical nucleus size

        This can be done via Seeding (code not included in this repository). Please see Ref. [2]  for an example of code to do Seeding simulations.

    * Prepare configurations of nuclei with sizes close to your target sizes

        This can also be done via Seeding. Good choices for the target sizes are $n_c/4$, $n_c/2$, $3 n_c/4$, and $n_c$, where $n_c$ is the critical nucleus size. Our seeded configurations can be found in the `results` directory.

4. **Umbrella Sampling**: Perform Umbrella Sampling simulations with hybrid Monte Carlo to obtain barrier segments
    
    * Use the Jupyter notebook `scripts/notebooks/idealized-barrier.ipynb` to estimate reasonable bias widths for the bias potentials. Note that this notebook estimates bias widths, which can be converted into spring constants as $k = 1/(bias ~width)^2$.

    * Use the python script `python hmc_npt_bias.py` to perform the Umbrella Sampling simulations with hybrid Monte Carlo. 
        
      For example:
      `python hmc_npt_bias.py -model mW -P 0.0 -T 215.1 -target_size 70 -bias_width 7.6 -integrator npt -n_steps 250 -dt 2.0`
      starts a hybrid Monte Carlo simulation of the mW model at zero pressure, 215.1K, a bias potential with a target size of 70 and a bias width of 7.6, using an NPT integrator for the molecular dynamics trajectories.

    * Use the same script to perform unbiased simulations of the bulk parent phase. 
    
        For example:
      `python hmc_npt_bias.py -model mW -P 0.0 -T 215.1 -target_size 0 -integrator npt -n_steps 250 -dt 2.0`
      starts an unbiased molecular dynamics simulation of the mW model under the same conditions using an NPT integrator.

5. **Variational aCNT fit**

    Use the Jupyter notebook `scripts/notebooks/fit-acnt.ipynb` to fit the barrier segments with adjusted classical nucleation theory (aCNT). This notebook plots the barrier and prints the height of the nucleation barrier.
    
    Essentially, all this notebook really needs is histograms of nucleus size distributions of a few barrier segments. We have included this data in the `results` directory, so that you can try this notebook without performing additional hybrid Monte Carlo simulations. For optimal results, especially for the estimation of confidence intervals, make sure to estimate the correlation time of a simulation and subsample to get independent samples before you construct the nucleus size histogram. We have included the nucleus size histograms after subsampling, named `nucleus_size_hist_processed.csv`, as well as the nucleus size histograms without subsampling, named `nucleus_size_hist_full.csv`. To obtain the initial part of the barrier from the unbiased simulations, see the `nucleus_size_hist_processed.csv` files in the `n0` directories for each system. Here, a nucleus size of 0 refers to all liquidlike particles. To get the factor (N_n / N) from this file, divide the 'count' column by the total number of particles of all snapshots combined, which can be obtained by summing the 'count' column weighted with the nucleus size. To be precise, divide by (M_0 + M_1 + 2* M_2 + 3*M_3 + ...), where M_n refers to the 'count' column. These unbiased simulations use 2000, 4000, 6000, or 8000 particles/molecules per snapshot, depending on the system (see the `args.dat` files). Please see the `fit_g0` function in `fit-acnt.ipynb` script as an example of how to compute the initial part of the barrier.



## License

The simulation code is based on the hybrid Monte Carlo code by Guo, Haji-Akbari, and Palmer [3] and uses the LAMMPS code for molecular dynamics [4] and the freud library for calculating bond order
parameters [1].

All source code is made available under the GNU general public license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE` for the full license text.

If you use or build on our work, please cite our paper.



[1] V. Ramasubramani, B. D. Dice, E. S. Harper, M. P. Spellings, J. A. Anderson, and S. C. Glotzer. freud: A Software Suite for High Throughput Analysis of Particle Simulation Data. Computer Physics Communications Volume 254, September 2020, 107275. [https://doi.org/10.1016/j.cpc.2020.107275](https://doi.org/10.1016/j.cpc.2020.107275).

[2] Willem Gispen and Marjolein Dijkstra, "Kinetic Phase Diagram for Nucleation and Growth of Competing Crystal Polymorphs in Charged Colloids", Phys. Rev. Lett. 129, 098002 (2022) [https://doi.org/10.1103/PhysRevLett.129.098002](https://doi.org/10.1103/PhysRevLett.129.098002)

[3] J. Guo, A. Haji-Akbari, and J. C. Palmer, "Hybrid Monte Carlo with LAMMPS", J. Theor. Comput. Chem. 17, 
1840002 (2018) [https://doi.org/10.1142/S0219633618400023](https://doi.org/10.1142/S0219633618400023)

[4] S. Plimpton, "Fast Parallel Algorithms for Short-Range Molecular Dynamics", J. Comput. Phys. 117, 1 (1995) [https://doi.org/10.1006/jcph.1995.1039](https://doi.org/10.1006/jcph.1995.1039)