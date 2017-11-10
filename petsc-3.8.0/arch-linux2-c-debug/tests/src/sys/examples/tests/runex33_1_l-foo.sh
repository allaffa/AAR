#!/usr/bin/env bash
# This script was created by gmakegentest.py



# PATH for DLLs on windows
PATH="$PATH":"/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/lib"

exec='../ex33'
testname='runex33_1_l-foo'
label='sys_tests-ex33_1_l-foo'
runfiles=''
wPETSC_DIR='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_dir='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_arch='arch-linux2-c-debug'
# Must be consistent with gmakefile
testlogfile=${petsc_dir}/${petsc_arch}/tests/examples-${petsc_arch}.log
DATAFILESPATH=${DATAFILESPATH:-""}
args='-a b -l foo'
timeoutfactor=1

mpiexec=${PETSCMPIEXEC:-"mpiexec"}
diffexec=${PETSCDIFF:-"${petsc_dir}/bin/petscdiff"}

. "${petsc_dir}/config/petsc_harness.sh"

# The diff flags come from script arguments
diff_exe="${diffexec} ${diff_flags}"
mpiexec="${mpiexec} ${mpiexec_flags}"
nsize=${nsize:-${nsize}}





for nsize in 1,2; do
   for c in d "e,f" g; do
      for h in "i,j" "k"; do

         petsc_testrun "echo ${args}" ex33_1_l-foo.tmp ${testname}.err cmd-${label}_nsize-${nsize}_c-${c}_h-${h} 
         res=$?

         if test $res = 0; then
            petsc_testrun "${diff_exe} /sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/src/sys/examples/tests/output/ex33_1_l-foo.out ex33_1_l-foo.tmp" diff-${testname}.out diff-${testname}.out diff-${label}_nsize-${nsize}_c-${c}_h-${h} ""
         else
            printf "ok ${label} # SKIP Command failed so no diff\n"
         fi

      done
   done
done

petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
