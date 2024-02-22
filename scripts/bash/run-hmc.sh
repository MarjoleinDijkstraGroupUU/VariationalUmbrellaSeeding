#!/bin/bash
#$ -cwd
#$ -l h=!node92&!node91&!node90&!node89&!node20

source ~/.bashrc
conda activate freud-lammps

## WCA
mpirun -np 4 python hmc_npt_bias.py -model wca -P 12.0 -target_size $N -bias_width ${bias_width} -dt 0.001 -n_steps 500 -lnvol_max 0.01 -integrator ${integrator} -iter $k -threshold $t

## mW
#mpirun -np 4 python hmc_npt_bias.py -model mW -dt 2.0 -P 0.0 -T $T -target_size $N -bias_width ${bias_width} -lnvol_max 0.01 -integrator ${integrator} -iter $k -n_steps 250

## TIP4P
#mpirun -np 4 python hmc_npt_bias.py -model tip4pice -dt 2.0 -P 0.9869 -T $T -target_size $N -bias_width ${bias_width} -lnvol_max 0.01 -integrator ${integrator} -iter $k -n_steps 10000
