#!/usr/bin/env bash
# This script was created by gmakegentest.py



# PATH for DLLs on windows
PATH="$PATH":"/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/lib"

exec='../ex62'
testname='runex62_pc_simplec'
label='snes_tutorials-ex62_pc_simplec'
runfiles=''
wPETSC_DIR='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_dir='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_arch='arch-linux2-c-debug'
# Must be consistent with gmakefile
testlogfile=${petsc_dir}/${petsc_arch}/tests/examples-${petsc_arch}.log
DATAFILESPATH=${DATAFILESPATH:-""}
args='-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -ksp_type fgmres -ksp_max_it 5 -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -fieldsplit_pressure_inner_ksp_type preonly -fieldsplit_pressure_inner_pc_type jacobi -fieldsplit_pressure_inner_pc_jacobi_type rowsum -fieldsplit_pressure_upper_ksp_type preonly -fieldsplit_pressure_upper_pc_type jacobi -fieldsplit_pressure_upper_pc_jacobi_type rowsum -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0'
timeoutfactor=1

mpiexec=${PETSCMPIEXEC:-"mpiexec"}
diffexec=${PETSCDIFF:-"${petsc_dir}/bin/petscdiff"}

. "${petsc_dir}/config/petsc_harness.sh"

# The diff flags come from script arguments
diff_exe="${diffexec} ${diff_flags}"
mpiexec="${mpiexec} ${mpiexec_flags}"
nsize=${nsize:-1}



if ! $force; then
    printf "ok ${label} # SKIP PETSC_HAVE_TRIANGLE requirement not met\n"
    total=1; skip=1
    petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
    exit
fi


petsc_testrun "${mpiexec} -n ${nsize} ${exec} ${args} " ex62_pc_simplec.tmp ${testname}.err "${label}" 
res=$?

if test $res = 0; then
   petsc_testrun "${diff_exe} /sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/src/snes/examples/tutorials/output/ex62_pc_simplec.out ex62_pc_simplec.tmp" diff-${testname}.out diff-${testname}.out diff-${label} ""
else
   printf "ok ${label} # SKIP Command failed so no diff\n"
fi

petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
