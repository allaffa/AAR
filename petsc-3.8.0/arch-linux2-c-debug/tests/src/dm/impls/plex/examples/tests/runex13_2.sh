#!/usr/bin/env bash
# This script was created by gmakegentest.py



# PATH for DLLs on windows
PATH="$PATH":"/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/lib"

exec='../ex13'
testname='runex13_2'
label='dm_impls_plex_tests-ex13_2'
runfiles=''
wPETSC_DIR='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_dir='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_arch='arch-linux2-c-debug'
# Must be consistent with gmakefile
testlogfile=${petsc_dir}/${petsc_arch}/tests/examples-${petsc_arch}.log
DATAFILESPATH=${DATAFILESPATH:-""}
args='-test_num 1 -dm_view ascii::ascii_info_detail -oriented_dm_view ascii::ascii_info_detail -orientation_view'
timeoutfactor=1

mpiexec=${PETSCMPIEXEC:-"mpiexec"}
diffexec=${PETSCDIFF:-"${petsc_dir}/bin/petscdiff"}

. "${petsc_dir}/config/petsc_harness.sh"

# The diff flags come from script arguments
diff_exe="${diffexec} ${diff_flags}"
mpiexec="${mpiexec} ${mpiexec_flags}"
nsize=${nsize:-2}



if ! $force; then
    printf "ok ${label} # SKIP PETSC_HAVE_TRIANGLE requirement not met\n"
    total=1; skip=1
    petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
    exit
fi


petsc_testrun "${mpiexec} -n ${nsize} ${exec} ${args} " ex13_2.tmp ${testname}.err "${label}" 
res=$?

if test $res = 0; then
   petsc_testrun "${diff_exe} /sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/src/dm/impls/plex/examples/tests/output/ex13_2.out ex13_2.tmp" diff-${testname}.out diff-${testname}.out diff-${label} ""
else
   printf "ok ${label} # SKIP Command failed so no diff\n"
fi

petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
