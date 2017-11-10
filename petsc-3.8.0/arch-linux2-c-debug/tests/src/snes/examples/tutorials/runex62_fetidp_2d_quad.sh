#!/usr/bin/env bash
# This script was created by gmakegentest.py



# PATH for DLLs on windows
PATH="$PATH":"/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/lib"

exec='../ex62'
testname='runex62_fetidp_2d_quad'
label='snes_tutorials-ex62_fetidp_2d_quad'
runfiles=''
wPETSC_DIR='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_dir='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_arch='arch-linux2-c-debug'
# Must be consistent with gmakefile
testlogfile=${petsc_dir}/${petsc_arch}/tests/examples-${petsc_arch}.log
DATAFILESPATH=${DATAFILESPATH:-""}
args='-run_type full -dm_refine 2 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -snes_view -snes_error_if_not_converged -show_solution 0 -dm_mat_type is -ksp_type fetidp -ksp_rtol 1.0e-8 -ksp_fetidp_saddlepoint -fetidp_ksp_type cg -fetidp_fieldsplit_p_ksp_max_it 1 -fetidp_fieldsplit_p_ksp_type richardson -fetidp_fieldsplit_p_ksp_richardson_scale 200 -fetidp_fieldsplit_p_pc_type none -ksp_fetidp_saddlepoint_flip 1 -fetidp_bddc_pc_bddc_vertex_size 2 -fetidp_bddc_pc_bddc_dirichlet_pc_factor_mat_solver_package umfpack -fetidp_bddc_pc_bddc_neumann_pc_factor_mat_solver_package umfpack -simplex 0 -petscpartitioner_type simple -fetidp_fieldsplit_lag_ksp_type preonly'
timeoutfactor=1

mpiexec=${PETSCMPIEXEC:-"mpiexec"}
diffexec=${PETSCDIFF:-"${petsc_dir}/bin/petscdiff"}

. "${petsc_dir}/config/petsc_harness.sh"

# The diff flags come from script arguments
diff_exe="${diffexec} ${diff_flags}"
mpiexec="${mpiexec} ${mpiexec_flags}"
nsize=${nsize:-5}



if ! $force; then
    printf "ok ${label} # SKIP PETSC_HAVE_SUITESPARSE requirement not met\n"
    total=1; skip=1
    petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
    exit
fi


petsc_testrun "${mpiexec} -n ${nsize} ${exec} ${args} " ex62_fetidp_2d_quad.tmp ${testname}.err "${label}" 'grep -v "CG or CGNE: variant" | sed -e "s/BDDC: Graph max count: 9223372036854775807/BDDC: Graph max count: 2147483647/g"'
res=$?

if test $res = 0; then
   petsc_testrun "${diff_exe} /sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/src/snes/examples/tutorials/output/ex62_fetidp_2d_quad.out ex62_fetidp_2d_quad.tmp" diff-${testname}.out diff-${testname}.out diff-${label} ""
else
   printf "ok ${label} # SKIP Command failed so no diff\n"
fi

petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
