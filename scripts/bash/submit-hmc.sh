# -pe openmpi_mod 4
# WCA: 
  #  "5.7 0 npt 0" "5.7 46 npt 5.94" "5.7 46 nve 5.94" "5.7 92 npt 9.43" "5.7 92 nve 9.43" "5.7 139 npt 12.41" "5.7 139 nve 12.41" "5.7 185 npt 15.02" "5.7 185 nve 15.02" \
  #  "6.7 0 npt 0" "6.7 39 npt 5.48" "6.7 39 nve 5.48" "6.7 78 npt 8.70" "6.7 78 nve 8.70" "6.7 116 npt 11.33" "6.7 116 nve 11.33" "6.7 155 npt 13.75" "6.7 155 nve 13.75" \
  #  "7.7 0 npt 0" "7.7 32 npt 4.94" "7.7 32 nve 4.94" "7.7 65 npt 7.93" "7.7 65 nve 7.93" "7.7 97 npt 10.36" "7.7 97 nve 10.36" "7.7 130 npt 12.59" "7.7 130 nve 12.59"
# mW
  # "215.1 0 npt 0" "215.1 17 npt 2.96" "215.1 17 nve 2.96" "215.1 35 npt 4.79" "215.1 35 nve 4.79" "215.1 52 npt 6.23" "215.1 52 nve 6.23" "215.1 70 npt 7.60" "215.1 70 nve 7.60" \
  # "225.0 0 npt 0" "225.0 37 npt 4.90" "225.0 37 nve 4.90" "225.0 75 npt 7.85" "225.0 75 nve 7.85" "225.0 112 npt 10.25" "225.0 112 nve 10.25" "225.0 150 npt 12.46" "225.0 150 nve 12.46" \
  # "235.0 0 npt 0" "235.0 80 npt 8.26" "235.0 80 nve 8.26" "235.0 160 npt 13.11" "235.0 160 nve 13.11" "235.0 240 npt 17.18" "235.0 240 nve 17.18" "235.0 320 npt 20.82" "235.0 320 nve 20.82"
# tip4pice
  # "230.0 0 npt 0" "230.0 60 npt 7.70" "230.0 120 npt 12.22" "230.0 180 npt 16.01" "230.0 240 npt 19.40"

for N_integrator_bias in "5.7 0 npt 0" "5.7 46 npt 5.94"
do
  set -- ${N_integrator_bias}
  for k in 6 #{1..4}
  do
    qsub -e out -o out -pe openmpi_mod 4 \
      -v t=$1 -v T=$1 \
      -v N=$2 -v integrator=$3 -v k=$k -v bias_width=$4 \
      ~/VariationalUmbrellaSeeding/scripts/bash/run-hmc.sh
  done
done

