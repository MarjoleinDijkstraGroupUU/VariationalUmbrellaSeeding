# EXAMPLE USAGE: see run-hmc.sh or see one of the following examples:
# mpirun -np 4 python hmc_npt_bias.py -model wca -P 12.0 -target_size 139 \ 
#   -bias_width 12.41 -dt 0.001 -n_steps 500 -lnvol_max 0.01 -integrator npt -iter 5 -threshold 5.7
# mpirun -np 4 python hmc_npt_bias.py -model mW -dt 5.0 -P 0.0 -T 215.1 -target_size 70 \
#   -bias_width 7.6 -lnvol_max 0.01 -integrator npt -iter 5 -n_steps 100
# mpirun -np 4 python hmc_npt_bias.py -model tip4pice -dt 2.0 -P 0.9869 -T 230.0 -target_size 240 \
#   -bias_width 19.40 -lnvol_max 0.01 -integrator npt -iter 5 -n_steps 10000

#-------------------- Import Packages --------------------#

import argparse
from ctypes import c_double
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd

import freud
from lammps import lammps

# parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
freud.parallel.set_num_threads(1)

#-------------------- Command Line Arguments --------------------#

parser = argparse.ArgumentParser(description='Hybrid Monte Carlo')

# Arguments: Interparticle interactions and thermodynamic statepoint
parser.add_argument('-model', metavar='model', type=str, required=True,
                    help='Model for interparticle interactions')
parser.add_argument('-P', metavar='P', type=float, required=True,
                    help='Pressure')
parser.add_argument('-T', metavar='T', type=float, required=False, default=1.0,
                    help='Temperature')

# Arguments: Bias potential
parser.add_argument('-target_size', metavar='target_size', type=int, required=False, default=0,
                    help='Target nucleus size of bias potential')
parser.add_argument('-bias_width', metavar='bias_width', type=float, required=False, default=np.inf,
                    help='Bias width of bias potential. \
                      The default of np.inf means that no bias potential is used.')
parser.add_argument('-threshold', metavar='threshold', type=float, required=False, default=0.5,
                    help='Threshold for solid-liquid order parameter')
parser.add_argument('-iter', metavar='iter', type=int, required=False, default=0,
                    help='Iteration in case you want to do multiple parallel simulations')

# Arguments: Molecular dynamics and hybrid Monte Carlo
parser.add_argument('-n_steps', metavar='n_steps', type=int, required=False, default=100,
                    help='Number of MD Steps per HMC trial step')
parser.add_argument('-dt', metavar='dt', type=float, required=True,
                    help='MD timestep')
parser.add_argument('-lnvol_max', metavar='lnvol_max', type=float, required=False,
                    help='Maximum multiplier of box size in volume move')
parser.add_argument('-integrator', metavar='integrator', type=str, required=False, default='nve',
                    help='Integrator to use for MD trajectory')
parser.add_argument('--metropolis', action=argparse.BooleanOptionalAction, required=False, default=True,
                    help='Whether to perform a Metropolis accept/reject step on *internal* energy. \
                    False means differences in internal energy are set to zero, \
                      but the bias potential is still taken into account. \
                        If the bias potential is also set to zero (with bias_width=np.inf),\
                          every trajectory is automatically accepted')
parser.add_argument('--rand_vel', action=argparse.BooleanOptionalAction, required=False, default=False,
                    help='Whether to re-sample velocities at the start of an MD trajectory.\
                        Defaults to False for target_size=0, and True otherwise (see below)')

# Argument: Initial configuration
parser.add_argument('-restart', metavar='restart', type=str, required=False,
                    help='Path to LAMMPS data file containing an initial configuration. \
                      If not specified, we attempt to find a LAMMPS dump or LAMMPS data file \
                          named seed.dump or seed.dat')

args = parser.parse_args()


#-------------------- Run Parameters --------------------#

# Thermodynamic state -  configured for LAMMPS' "real" or "lj" units
if args.model == 'mW':
  units = 'real'
  T = args.T # Temperature [K]
  P = args.P # Pressure [atm] # 0.986923
  t_equil = 1e3 # typical equilibration time
  dir_seed = f'../../results/hmc/{args.model}/T{T}' # base directory to store data
elif args.model == 'wca':
  units = 'lj'
  T = 1.0
  P = T * args.P
  t_equil = 1.0 # typical equilibration time
  dir_seed = f'../../results/hmc/{args.model}/p{args.P}_{round(args.threshold)}-bonds' # base directory to store data
elif args.model == 'tip4pice':
  units = 'real'
  T = args.T # Temperature [K]
  P = args.P # Pressure [atm] # 0.986923
  t_equil = 1e3 # typical equilibration time
  dir_seed = f'../../results/hmc/{args.model}/T{T}' # base directory to store data
else:
  sys.exit()

# HMC parameters
n_steps = args.n_steps   # Number of MD Steps per HMC trial step
dt = args.dt             # time step [fs] or [sqrt(m sigma^2 / eps)]
dT = n_steps * dt        # time per MD trajectory
p_hmc = 0.5 if args.lnvol_max is not None else 1.0  # Probability of selecting HMC move (vs Volume Move); >= 1.0 for NVT ensemble
if args.integrator == 'npt':
    p_hmc = 1.0 # don't perform volume moves
if args.model == 'tip4pice':
    n_sweeps = round(5e4 * t_equil / dT / p_hmc)     # Number of MC Sweeps (1 Sweep = 1 HMC or Volume Move)
else:
    n_sweeps = round(2e4 * t_equil / dT / p_hmc)     # Number of MC Sweeps (1 Sweep = 1 HMC or Volume Move)
lnvol_max = args.lnvol_max    # Maximum log volume displacement
rseed = args.iter             # Random number seed

if args.integrator != 'nve':
  # do not take differences in *internal* energies into account.
  # see the help for the metropolis argument above
  args.metropolis = False

if args.target_size == 0:
  # don't apply a bias potential for the bulk parent phase.
  # i.e. for the parent phase, this is a normal MD simulation.
  args.bias_width = np.inf
else:
  # always randomize velocities for target_size > 0.
  args.rand_vel = True

# File output frequency
freq_thermo = max(1, round(t_equil / dT))            # Thermodynamic output
freq_traj = max(1, round(100*t_equil / dT))          # XYZ trajectory
freq_restart = max(1, round(100*t_equil / dT))       # Restart file (LAMMPS write_data)
freq_hist = min(10000, max(1, round(100*t_equil / dT))) # Nucleus size histogram

# Output directory
dir_out = f'{dir_seed}/n{args.target_size}/{args.integrator}-dt{args.dt}-nsteps{args.n_steps}-rvel-{args.rand_vel}/{args.iter}'
os.makedirs(dir_out, exist_ok=True)


#-------------------- Physical Constants and Conversion Factors --------------------#
# Note: These should changes if LAMMPS' units change

# Constants.
kB = 1.380648520000e-23  # Boltzmann's constant [m^2 kg s^-2 K^-1]
Nav = 6.02214090000e23   # Avogadro's number [molecules mol^-1]
R = kB * Nav / 1000.0    # Gas constant [kJ mol^-1 K^-1]

# Thermal energy
kTs = R*T                # SI units: [kJ mol^-1]
kTL = kTs/4.184          # LAMMPS units [real units, kcal mol^-1]

# Velocity prefactor
vf = 1e-4*R*T            # LAMMPS units [real units, g A^2 * fs^-2 mol^-1]

# Pressure prefactor for volume change move
Pb = P*1.01325           # Pressure [bar]
Pc = kB*1.0e30*1.0e-5    # Conversion factor [bar A^3 K^-1]
Pf = Pb/(Pc*T)           # Prefactor for Metropolis criterion [A^-3]

if units == 'lj':
  kB = 1.0
  kTL = kB * T
  Pf = P / kTL
  vf = kTL

#-------------------- Support Functions --------------------#

# Velocity initialization
## -Draw initial velocities from Maxwell-Boltzmann distribution
## -Send velocities to LAMMPS
## -"run 0" to set velocities internally
def init_vel(_rseed=0):
  if args.model == 'tip4pice':
    command = f"""
      velocity all create {T} {1+_rseed} dist gaussian
      velocity all zero linear
      timestep {dt/100}
      run 1
      velocity all scale {T}
      timestep {dt}
    """
    lmp.commands_string(command)
    v = lmp.gather_atoms("v",1,3)
  else:
    # WARNING: THIS ROUTINE ONLY WORKS FOR POINT PARTICLES.
    v=(3*natoms*c_double)()
    for i in range(natoms):
      indx = 3*i
      sigma = math.sqrt(vf/mass[atype[i]])
      v[indx] =  random.gauss(0.0,sigma) 
      v[indx+1] = random.gauss(0.0,sigma)
      v[indx+2] = random.gauss(0.0,sigma)

  lmp.scatter_atoms("v",1,3,v)
  lmp.command("run 0")


# Compute MC move acceptance ratio
def acc_ratio(acc, trys):
  if trys == 0.0:
    ratio = 0.0
  else:
    ratio = acc/trys
  return ratio


# Compute the change in bias potential
# units: kT
def get_bias(_nucleus_size, _target_size):
  bias = 0.5 * (_nucleus_size - _target_size)**2 / args.bias_width**2
  return bias


# Compute the nucleus size of the current snapshot
def compute_nucleus_size(_box, points, threshold=args.threshold):
  # construct box
  _box = freud.box.Box.from_box(_box)
  points= np.array(points).reshape((-1, 3))
  if args.model == 'tip4pice':
    points = points[np.array(atype)==1]

  # get neighbor list
  nq = freud.locality.AABBQuery.from_system((_box, points))
  if args.model == 'wca':
    query_args = dict(mode='ball', r_max=1.5, exclude_ii=True)
  elif args.model in ['mW', 'tip4pice']:
    query_args = dict(r_max=3.51, exclude_ii=True)
  nlist = nq.from_system((_box, points)).query(points, query_args).toNeighborList()

  if args.model == 'wca':
    # compute the number of solidlike bonds per particle
    solid_liquid = freud.order.SolidLiquid(normalize_q=True, l=6, q_threshold=0.7, solid_threshold=threshold)
    solid_liquid.compute((_box, points), neighbors=nlist)
    op = solid_liquid.num_connections
  elif args.model in ['mW', 'tip4pice']:
    # compute the averaged Steinhardt bond order parameter q6
    ql = freud.order.Steinhardt(l=6, average=True)
    ql.compute((_box, points), neighbors=nlist)
    op = ql.particle_order # cluster
    if args.model == 'mW':
      # from Espinosa et al. 2016 "Seeding approach to crystal nucleation"
      threshold = 0.3593 + (0.3806-0.3593)*(274.6-T)/40
    else:
      # from Espinosa et al. 2016 "On the time required"
      threshold = 0.372 + (0.361-0.372)*(T-238.75)/(255.0-238.75)
  
  # determine which particles are solidlike
  _solidlike = (op > threshold)

  # only keep bonds between solidlike particles in neighborlist
  nlist.filter(op[nlist.query_point_indices] > threshold)
  nlist.filter(op[nlist.point_indices] > threshold)

  # find largest cluster of solidlike particles
  _cluster = freud.cluster.Cluster()
  _cluster.compute((_box, points), neighbors=nlist)
  nucleus = (_cluster.cluster_idx == 0)
  _nucleus_size = nucleus.sum()

  return _nucleus_size, _cluster, _solidlike


def get_nucleus_size_distribution(_cluster, _solidlike):
   # obtain the cluster id of each solidlike particle
  solidlike_cluster_idx = _cluster.cluster_idx[_solidlike]
  # get a list containing the size of each cluster
  _, cluster_sizes = np.unique(solidlike_cluster_idx, return_counts=True)
  # represent fluidlike particles as clusters of size 0
  cluster_sizes = np.append(cluster_sizes, np.repeat([0], (1-_solidlike).sum()))
  return cluster_sizes


#-------------------- LAMMPS script --------------------#

# Seed random number generator
random.seed(rseed)

# Initialize LAMMPS
lmp = lammps(name="",cmdargs=["-log","none","-screen","none"])
# lmp = lammps(name="", cmdargs=["-log","none",])
# lmp = lammps(name="", cmdargs=["-screen","none",])

# Specify interparticle interactions with LAMMPS script
if args.model != 'tip4pice':
  lmp.file(f"{args.model}.nve")
else:
  # TIP4P/ICE needs some commands before and some after loading the configuration.
  command = """
    units real
    atom_style full
    bond_style harmonic
    angle_style harmonic
    pair_style      lj/cut/tip4p/long 1 2 1 1 0.1577 9.0
    atom_modify     map array sort 0 0.0
  """
  lmp.commands_string(command)

# Read initial configuration
if args.restart is not None:
  # read configuration specified from command line
  lmp.command(f'read_data {args.restart}')
elif args.target_size == 0:
  # if simulation of fluid, read fluid configuration
  if os.path.exists(f'{dir_seed}/fluid.dump'):
    lmp.command(f'read_dump {dir_seed}/fluid.dump 0 x y z add yes')
  if os.path.exists(f'{dir_seed}/fluid.dat'):
    lmp.command(f'read_data {dir_seed}/fluid.dat')
else:
  # read a seeded nucleus configuration.
  # first try to find a seeded configuration that is specific to the target size.
  # if not available, try a generic seeded configuration.
  if os.path.exists(f'{dir_seed}/n{args.target_size}/seed.dat'):
    lmp.command(f'read_data {dir_seed}/n{args.target_size}/seed.dat')
  elif os.path.exists(f'{dir_seed}/n{args.target_size}/seed.dump'):
    lmp.command(f'read_dump {dir_seed}/n{args.target_size}/seed.dump 0 x y z add yes')
  elif os.path.exists(f'{dir_seed}/seed.dat'):
    lmp.command(f'read_data {dir_seed}/seed.dat')
  elif os.path.exists(f'{dir_seed}/seed.dump'):
    lmp.command(f'read_dump {dir_seed}/seed.dump 0 x y z add yes')

# TIP4P/ICE needs some commands after loading the configuration.
if args.model == 'tip4pice':
  lmp.file(f"{args.model}.nve")
lmp.command('run 0')

# Set the integration time step
lmp.command(f"timestep {dt}")

# MD integrator
## relaxation times of thermostat and barostat
relaxT, relaxP = (100*dt, 500*dt)
if args.model == 'tip4pice':
    relaxT, relaxP = (2e3, 2e3)
if args.model == 'mW':
    relaxT, relaxP = (50*dt, 100*dt)
## define integrator
if args.integrator == 'nvt':
  integrator = f"fix nvt all nvt temp {T} {T} {relaxT}"
elif args.integrator == 'npt':
  integrator = f"fix npt all npt temp {T} {T} {relaxT} iso {P} {P} {relaxP}"
## set integrator in lammps
if args.integrator != 'nve':
  lmp.command("unfix nve")
  lmp.commands_string(integrator)

# Define compute for kinetic energy 
lmp.command("compute thermo_ke all ke")


#-------------------- Initialization --------------------#

# Get initial system properties
natoms = lmp.extract_global("natoms",0) # number of particles
mass = lmp.extract_atom("mass",2) # particle mass
atomid = lmp.gather_atoms("id",0,1) # particle id
atype = lmp.gather_atoms("type",0,1) # particle type

# Allocate coordinate and velocity arrays 
x=(3*natoms*c_double)() # xyz coordinates
x_new = (3*natoms*c_double)() # proposed new xyz coordinates
v=(3*natoms*c_double)() # velocities

# Initialize properties
pe = 0.0 # potential energy
ke = 0.0 # kinetic energy
etot = 0.0 # total energy
box = 0.0 # box size
vol = 0.0 # box volume
dH = 0.0 # difference in total energy
exp_dH = 0.0 # exponential of total energy difference

# Initialize counters
n_acc_hmc = 0.0 # number of accepted HMC moves
n_try_hmc = 0.0 # number of attempted HMC moves
n_acc_vol = 0.0 # number of accepted volume moves
n_try_vol = 0.0 # number of attempted volume moves

# Initialize histograms of nucleus sizes
if args.bias_width == np.inf:
  nucleus_sizes = np.arange(100)
else:
  nmin = round(max(0, args.target_size-20*args.bias_width))
  nmax = round(args.target_size+20*args.bias_width)
  nucleus_sizes = np.arange(nmin, nmax)
counts = np.zeros_like(nucleus_sizes)
hist = pd.DataFrame({'nucleus_size': nucleus_sizes, 'count': counts},)

# Get initial coordinates and velocities
x = lmp.gather_atoms("x",1,3)
v = lmp.gather_atoms("v",1,3)

# Compute initial PE [dimensionless]
pe = lmp.extract_compute("thermo_pe",0,0)/kTL

# Compute box edge lengths and volume
boxlo,boxhi,_,_,_,_,_ = lmp.extract_box()
box = np.array(boxhi) - np.array(boxlo)
vol = np.prod(box)

# Compute the initial nucleus size
t0 = time.time()
if rank == 0:
  # compute on only one core
  nucleus_size, cluster, solidlike = compute_nucleus_size(box, x)
  # print the time it takes to compute the nucleus size
  print(f"Initial cluster size = {nucleus_size} in time {time.time()-t0}")
else:
  nucleus_size, cluster, solidlike = None, None, None
# 'broadcast' to make sure every core has access to the nucleus size
nucleus_size = comm.bcast(nucleus_size, root=0)

# Open files for writing
if rank == 0:
  # Write thermodynamic quantities, nucleus sizes, and HMC acceptance statistics
  with open(f'{dir_out}/thermo.dat', 'w') as thermo:
      thermo.write("step time pot_eng kin_eng press density hmc_acc vol_acc dH exp_dH nucleus_size nucleus_size_new\n")

  # Write all command line arguments
  with open(f'{dir_out}/args.dat', 'w') as argsfile:
      kwargs = vars(args)
      for key in kwargs:
        argsfile.write(f"{key} {kwargs[key]}\n")


#-------------------- Main Simulation --------------------#

for isweep in range(n_sweeps):
  if random.random() <= p_hmc:
    #-------------------- HMC move --------------------#

    # Update number of trials 
    n_try_hmc += 1.0

    # Set coordinates and box size,
    # i.e. this changes the coordinates and box size if the previous move was rejected.
    lmp.scatter_atoms("x",1,3,x)
    if args.integrator == 'npt':
      lmp.command(f"change_box all x final 0.0 {box[0]} y final 0.0 {box[1]} z final 0.0 {box[2]} remap units box")

    # Generate initial velocities; compute KE [dimensionless]
    if args.rand_vel or isweep == 0:
      init_vel(_rseed=rseed+isweep)
    ke = lmp.extract_compute("thermo_ke",0,0)/kTL
    etot = pe + ke
   
    # Simulate a short MD trajectory of n_steps timestpes
    lmp.command(f"run {n_steps}")
   
    # Compute new PE, KE, total energy,[dimensionless]
    pe_new = lmp.extract_compute("thermo_pe",0,0)/kTL
    ke_new = lmp.extract_compute("thermo_ke",0,0)/kTL
    etot_new = pe_new + ke_new

    # Compute total energy difference dH, the argument for the acceptance criterion [dimensionless]
    dH = (etot_new - etot)
    if not args.metropolis:
      # ignore the change in total potential+kinetic energy
      dH = 0

    # Get new coordinates and new box size
    x_new = lmp.gather_atoms("x",1,3)
    if args.integrator == 'npt':
      boxlo,boxhi,_,_,_,_,_ = lmp.extract_box()
      box_new = np.array(boxhi) - np.array(boxlo)
    else:
      box_new = box

    # Compute the nucleus size
    if rank == 0:
      nucleus_size_new, cluster_new, solidlike_new = compute_nucleus_size(box_new, x_new)
    else:
      nucleus_size_new, cluster_new, solidlike_new = None, None, None
    nucleus_size_new = comm.bcast(nucleus_size_new, root=0)

    # Compute the change in bias potential and add it to total energy difference
    target_size = args.target_size
    dH += (get_bias(nucleus_size_new, target_size) - get_bias(nucleus_size, target_size))
    if not np.isfinite(args.bias_width) and nucleus_size_new > 200:
        print('Spontaneous nucleation')

    # Track exp_dH for consistency check
    if np.abs(dH) > 500:
      # for numerical stability
      dH = np.sign(dH)*500
    exp_dH += math.exp(-dH)

    # Perform Metropolis acceptance-rejection test
    if random.random() <= math.exp(-dH) or isweep == 0: # accept
      n_acc_hmc += 1
      pe = pe_new

      # accept new box size
      if args.integrator == 'npt':
        box = box_new

      # accept new nucleus size
      if rank == 0:
        nucleus_size, cluster, solidlike = nucleus_size_new, cluster_new, solidlike_new
      else:
        nucleus_size = nucleus_size_new

      # accept new coordinates
      for i in range(3*natoms):
        x[i] = x_new[i]
  else:
    #-------------------- Log-volume MC Move --------------------#

    # Update number of trials
    n_try_vol += 1.0

    # Scatter coordinates
    lmp.scatter_atoms("x",1,3,x)

    # Generate random displacement in ln(volume)
    lnvol = math.log(vol) + (random.random() - 0.5)*lnvol_max

    # Calculate new box volume, size and scale factor
    vol_new = math.exp(lnvol)
    scalef = (vol_new / vol)**(1/3)
    box_new = scalef * box
    if args.model == 'tip4pice':
      sys.exit('Volume moves not supported')
    lmp.command(f"change_box all x final 0.0 {box_new[0]} y final 0.0 {box_new[1]} z final 0.0 {box_new[2]} remap units box")
    
    # Scale the coordinates and send to LAMMPS
    for i in range(3*natoms):   
      x_new[i] = scalef*x[i]
    lmp.scatter_atoms("x",1,3,x_new)
    lmp.command("run 0")

    # Compute the new PE [dimensionless]
    pe_new = lmp.extract_compute("thermo_pe",0,0)/kTL

    # Calculate argument for the acceptance criterion [dimensionless]
    arg = (pe_new-pe) + Pf*(vol_new-vol) - (float(natoms) + 1.0)*math.log(vol_new/vol)

    # Compute the nucleus size
    if rank == 0:
      nucleus_size_new, cluster_new, solidlike_new = compute_nucleus_size(box_new, x_new)
    else:
      nucleus_size_new, cluster_new, solidlike_new = None, None, None
    nucleus_size_new = comm.bcast(nucleus_size_new, root=0)

    # Compute the change in bias potential 
    target_size = args.target_size
    arg += (get_bias(nucleus_size_new, target_size) - get_bias(nucleus_size, target_size))

    # Perform Metropolis acceptance-rejection test
    if random.random() <= math.exp(-arg): # Accept
      n_acc_vol += 1.0
      pe = pe_new
      vol = vol_new
      box = box_new
      nucleus_size, cluster, solidlike = nucleus_size_new, cluster_new, solidlike_new
      for i in range(3*natoms):
        x[i] = x_new[i]
    else: # Reject, restore the old box size
      lmp.command(f"change_box all x final 0.0 {box[0]} y final 0.0 {box[1]} z final 0.0 {box[2]} remap units box")     


  #-------------------- Output --------------------#

  # Update histogram
  if rank == 0 and ((isweep + 1) % freq_thermo == 0):
    if args.bias_width == np.inf:
      # If performing an unbiased simulation, count all nuclei in the system
      nucleus_size_distribution = get_nucleus_size_distribution(cluster, solidlike)
      nucleus_sizes, counts = np.unique(nucleus_size_distribution, return_counts=True)
      # Add to histogram
      for i, nucleus_size in enumerate(nucleus_sizes):
          row = (hist['nucleus_size'] == nucleus_size)
          hist.loc[row, ['count']] += counts[i]
    else:
      # Count only the largest nucleus size
      row = (hist['nucleus_size'] == nucleus_size)
      hist.loc[row, ['count']] += 1

    # Write histogram to file
    if ((isweep + 1) % freq_hist == 0):
      hist.to_csv(f'{dir_out}/nucleus_size_hist_step_{isweep+1}.csv', index=False)
    
  # Write log file
  if (isweep + 1) % freq_thermo == 0:  # Write thermodynamic data and HMC statistics
    # Get statistics
    hmc_acc = acc_ratio(n_acc_hmc, n_try_hmc)
    vol_acc = acc_ratio(n_acc_vol, n_try_vol)
    exp_dH_ = acc_ratio(exp_dH, n_try_hmc)
    virial = lmp.extract_compute("thermo_press",0,0)  # Get virial pressure 
    N = lmp.get_thermo("atoms")
    density = lmp.get_thermo("density")

    # Write and print log file
    if rank == 0:
      with open(f'{dir_out}/thermo.dat', 'a') as thermo:
          thermo.write(f'{isweep + 1} {(n_try_hmc)*dT:.5f} {kTL*pe/N:.5f} {kTL*ke/N:.5f} {virial/kTL:.5f} {density:.5f} {hmc_acc:.5f} {vol_acc:.5f} {dH:.5f} {exp_dH_} {nucleus_size} {nucleus_size_new} \n')
      print(f'{isweep + 1} {(n_try_hmc)*dT:.5f} {kTL*pe/N:.5f} {kTL*ke/N:.5f} {virial/kTL:.5f} {density:.5f} {hmc_acc:.5f} {vol_acc:.5f} {dH:.5f} {exp_dH_} {nucleus_size} {nucleus_size_new} \n')

  # Write configurations and restarts
  if (isweep + 1) % freq_restart == freq_restart/2:
    lmp.command(f"write_data {dir_out}/restart_a.dat") # Alternate restart files for redundancy
  if (isweep + 1) % freq_restart == 0:
    lmp.command(f"write_data {dir_out}/restart_b.dat")
  if (isweep + 1) % freq_restart == 0:
    lmp.command(f"write_data {dir_out}/restart_{isweep+1}.dat") # Save all configurations

MPI.Finalize()
