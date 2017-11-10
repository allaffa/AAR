#!/usr/bin/env bash
# This script was created by gmakegentest.py



# PATH for DLLs on windows
PATH="$PATH":"/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/lib"

exec='../ex12'
testname='runex12_fas_newton_0_p4est'
label='snes_tutorials-ex12_fas_newton_0_p4est'
runfiles=''
wPETSC_DIR='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_dir='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_arch='arch-linux2-c-debug'
# Must be consistent with gmakefile
testlogfile=${petsc_dir}/${petsc_arch}/tests/examples-${petsc_arch}.log
DATAFILESPATH=${DATAFILESPATH:-""}
args='-run_type full -variable_coefficient nonlinear -interpolate 1 -petscspace_order 1 -snes_type fas -snes_fas_levels 2 -pc_type svd -ksp_rtol 1.0e-10 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -cells 2,2'
timeoutfactor=1

mpiexec=${PETSCMPIEXEC:-"mpiexec"}
diffexec=${PETSCDIFF:-"${petsc_dir}/bin/petscdiff"}

. "${petsc_dir}/config/petsc_harness.sh"

# The diff flags come from script arguments
diff_exe="${diffexec} ${diff_flags}"
mpiexec="${mpiexec} ${mpiexec_flags}"
nsize=${nsize:-1}



if ! $force; then
    printf "ok ${label} # SKIP PETSC_HAVE_HDF5 requirement not met, PETSC_HAVE_TRIANGLE requirement not met, PETSC_HAVE_HDF5 requirement not met, PETSC_HAVE_P4EST requirement not met\n"
    total=1; skip=1
    petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
    exit
fi


petsc_testrun "${mpiexec} -n ${nsize} ${exec} ${args} " ex12_fas_newton_0_p4est.tmp ${testname}.err "${label}" 
res=$?

if test $res = 0; then
   petsc_testrun "${diff_exe} /sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/src/snes/examples/tutorials/output/ex12_fas_newton_0_p4est.out ex12_fas_newton_0_p4est.tmp" diff-${testname}.out diff-${testname}.out diff-${label} ""
else
   printf "ok ${label} # SKIP Command failed so no diff\n"
fi

petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
