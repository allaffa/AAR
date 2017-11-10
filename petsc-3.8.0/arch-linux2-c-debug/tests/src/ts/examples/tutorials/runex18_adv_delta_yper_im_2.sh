#!/usr/bin/env bash
# This script was created by gmakegentest.py



# PATH for DLLs on windows
PATH="$PATH":"/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/lib"

exec='../ex18'
testname='runex18_adv_delta_yper_im_2'
label='ts_tutorials-ex18_adv_delta_yper_im_2'
runfiles=''
wPETSC_DIR='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_dir='/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0'
petsc_arch='arch-linux2-c-debug'
# Must be consistent with gmakefile
testlogfile=${petsc_dir}/${petsc_arch}/tests/examples-${petsc_arch}.log
DATAFILESPATH=${DATAFILESPATH:-""}
args='-x_bd_type none -use_fv -use_implicit -velocity_petscspace_order 0 -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_mimex_version 0 -ts_final_time 5.0 -ts_max_steps 80 -ts_dt 0.083333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -dm_refine 2 -source_loc 0.458333,0.458333'
timeoutfactor=1

mpiexec=${PETSCMPIEXEC:-"mpiexec"}
diffexec=${PETSCDIFF:-"${petsc_dir}/bin/petscdiff"}

. "${petsc_dir}/config/petsc_harness.sh"

# The diff flags come from script arguments
diff_exe="${diffexec} ${diff_flags}"
mpiexec="${mpiexec} ${mpiexec_flags}"
nsize=${nsize:-1}

if ! $force; then
    printf "ok ${label} # TODO broken\n"
    total=1; todo=1
    petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
    exit
fi




petsc_testrun "${mpiexec} -n ${nsize} ${exec} ${args} " ex18_adv_delta_yper_im_2.tmp ${testname}.err "${label}" 
res=$?

if test $res = 0; then
   petsc_testrun "${diff_exe} /sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/src/ts/examples/tutorials/output/ex18_adv_delta_yper_im_2.out ex18_adv_delta_yper_im_2.tmp" diff-${testname}.out diff-${testname}.out diff-${label} ""
else
   printf "ok ${label} # SKIP Command failed so no diff\n"
fi

petsc_testend "/sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0/arch-linux2-c-debug/tests" 
