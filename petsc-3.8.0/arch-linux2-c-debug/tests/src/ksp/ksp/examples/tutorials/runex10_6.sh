#!/usr/bin/env bash
# This script was created by gmakegentest.py



# PATH for DLLs on windows
PATH="$PATH":"/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/lib"

exec='../ex10'
testname='runex10_6'
label='ksp_ksp_tutorials-ex10_6'
runfiles=''
wPETSC_DIR='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_dir='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_arch='arch-linux2-c-debug'
# Must be consistent with gmakefile
testlogfile=${petsc_dir}/${petsc_arch}/tests/examples-${petsc_arch}.log
DATAFILESPATH=${DATAFILESPATH:-""}
args='-f0 ${DATAFILESPATH}/matrices/fem1 -pc_factor_levels 2 -pc_factor_fill 1.73 -ksp_gmres_cgs_refinement_type refine_always'
timeoutfactor=1

mpiexec=${PETSCMPIEXEC:-"mpiexec"}
diffexec=${PETSCDIFF:-"${petsc_dir}/bin/petscdiff"}

. "${petsc_dir}/config/petsc_harness.sh"

# The diff flags come from script arguments
diff_exe="${diffexec} ${diff_flags}"
mpiexec="${mpiexec} ${mpiexec_flags}"
nsize=${nsize:-1}



if ! $force; then
    if test -z "${DATAFILESPATH}"; then
        printf "ok ${label} # SKIP Requires DATAFILESPATH\n"
        total=1; skip=1
        petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
        exit
    fi
fi


petsc_testrun "${mpiexec} -n ${nsize} ${exec} ${args} " ex10_6.tmp ${testname}.err "${label}" 
res=$?

if test $res = 0; then
   petsc_testrun "${diff_exe} /sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/src/ksp/ksp/examples/tutorials/output/ex10_6.out ex10_6.tmp" diff-${testname}.out diff-${testname}.out diff-${label} ""
else
   printf "ok ${label} # SKIP Command failed so no diff\n"
fi

petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
