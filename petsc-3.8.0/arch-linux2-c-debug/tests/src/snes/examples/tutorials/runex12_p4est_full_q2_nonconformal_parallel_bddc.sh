#!/usr/bin/env bash
# This script was created by gmakegentest.py



# PATH for DLLs on windows
PATH="$PATH":"/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/lib"

exec='../ex12'
testname='runex12_p4est_full_q2_nonconformal_parallel_bddc'
label='snes_tutorials-ex12_p4est_full_q2_nonconformal_parallel_bddc'
runfiles=''
wPETSC_DIR='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_dir='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_arch='arch-linux2-c-debug'
# Must be consistent with gmakefile
testlogfile=${petsc_dir}/${petsc_arch}/tests/examples-${petsc_arch}.log
DATAFILESPATH=${DATAFILESPATH:-""}
args='-run_type full -interpolate 1 -petscspace_order 2 -snes_max_it 20 -snes_type newtonls -dm_mat_type is -pc_type bddc -ksp_type cg -snes_monitor_short -snes_linesearch_type basic -snes_converged_reason -snes_view -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type simple -cells 2,2'
timeoutfactor=1

mpiexec=${PETSCMPIEXEC:-"mpiexec"}
diffexec=${PETSCDIFF:-"${petsc_dir}/bin/petscdiff"}

. "${petsc_dir}/config/petsc_harness.sh"

# The diff flags come from script arguments
diff_exe="${diffexec} ${diff_flags}"
mpiexec="${mpiexec} ${mpiexec_flags}"
nsize=${nsize:-7}



if ! $force; then
    printf "ok ${label} # SKIP PETSC_HAVE_HDF5 requirement not met, PETSC_HAVE_TRIANGLE requirement not met, PETSC_HAVE_HDF5 requirement not met, PETSC_HAVE_P4EST requirement not met\n"
    total=1; skip=1
    petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
    exit
fi


petsc_testrun "${mpiexec} -n ${nsize} ${exec} ${args} " ex12_p4est_full_q2_nonconformal_parallel_bddc.tmp ${testname}.err "${label}" 'grep -v "CG or CGNE: variant" | sed -e "s/BDDC: Graph max count: 9223372036854775807/BDDC: Graph max count: 2147483647/g"'
res=$?

if test $res = 0; then
   petsc_testrun "${diff_exe} /sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/src/snes/examples/tutorials/output/ex12_p4est_full_q2_nonconformal_parallel_bddc.out ex12_p4est_full_q2_nonconformal_parallel_bddc.tmp" diff-${testname}.out diff-${testname}.out diff-${label} ""
else
   printf "ok ${label} # SKIP Command failed so no diff\n"
fi

petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
