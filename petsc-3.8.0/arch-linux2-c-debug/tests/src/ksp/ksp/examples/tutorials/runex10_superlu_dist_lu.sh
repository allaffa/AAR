#!/usr/bin/env bash
# This script was created by gmakegentest.py



# PATH for DLLs on windows
PATH="$PATH":"/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/lib"

exec='../ex10'
testname='runex10_superlu_dist_lu'
label='ksp_ksp_tutorials-ex10_superlu_dist_lu'
runfiles=''
wPETSC_DIR='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_dir='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_arch='arch-linux2-c-debug'
# Must be consistent with gmakefile
testlogfile=${petsc_dir}/${petsc_arch}/tests/examples-${petsc_arch}.log
DATAFILESPATH=${DATAFILESPATH:-""}
args='-f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type lu -pc_factor_mat_solver_package superlu_dist -num_numfac 2 -num_rhs 2'
timeoutfactor=1

mpiexec=${PETSCMPIEXEC:-"mpiexec"}
diffexec=${PETSCDIFF:-"${petsc_dir}/bin/petscdiff"}

. "${petsc_dir}/config/petsc_harness.sh"

# The diff flags come from script arguments
diff_exe="${diffexec} ${diff_flags}"
mpiexec="${mpiexec} ${mpiexec_flags}"
nsize=${nsize:-${nsize}}



if ! $force; then
    printf "ok ${label} # SKIP Requires DATAFILESPATH, PETSC_HAVE_SUPERLU_DIST requirement not met\n"
    total=1; skip=1
    petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
    exit
fi


for nsize in 1 2; do

   petsc_testrun "${mpiexec} -n ${nsize} ${exec} ${args} " ex10_superlu_dist_lu.tmp ${testname}.err "${label}_nsize-${nsize}" 
   res=$?

   if test $res = 0; then
      petsc_testrun "${diff_exe} /sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/src/ksp/ksp/examples/tutorials/output/ex10_mumps.out ex10_superlu_dist_lu.tmp" diff-${testname}.out diff-${testname}.out diff-${label}_nsize-${nsize} ""
   else
      printf "ok ${label} # SKIP Command failed so no diff\n"
   fi

done

petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
