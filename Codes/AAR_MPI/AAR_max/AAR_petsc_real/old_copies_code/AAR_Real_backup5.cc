/*=============================================================================================
  | Alternating Anderson Richardson (AAR) code
  | Copyright (C) 2017 Material Physics & Mechanics Group at Georgia Tech.
  | 
  | Authors: Phanisri Pradeep Pratapa, Phanish Suryanarayana
  |
  | Last Modified: 16 Aug 2017   
  |-------------------------------------------------------------------------------------------*/

static char help[] = "Alternating Anderson Richardson (AAR) code\n\
options:\n\
-name name of file\n";

#include "petsc.h"
#include "petscksp.h"
#include "assert.h"
#include "petscdmda.h"
#include <petsctime.h>
#include <mpi.h>
#include "petscsys.h"
#include "math.h"
#include "mkl_lapacke.h"
#include "mkl.h"
#include <iostream>
#include "petscmat.h" 


typedef struct
{

  PetscInt m_aar;
  PetscInt p_aar;
  PetscScalar omega;
  PetscReal solver_tol;
  PetscInt maxit;
  PetscInt gmres_compute;
  char prec_type[20];

}AAR_PARAMS;

typedef struct
{

  PetscInt M, N, lprank, m_aar, p_aar, num_iterations;
  PetscScalar omega, tol;
  Mat *A;
  Vec x_guess, x_old, x_new, x_final, res_old, res_new, Ax, res_old_rescaled, res_new_rescaled, prec_res_old, prec_res_new, gamma;

  const PetscScalar *df_col;
  PetscInt *row_inds;
  PetscInt *row_inds_local;
  PetscInt *col_inds;
  PetscScalar* gamma_array;

  Vec *DX, *DF;
  Vec dp, dp1, dp2;

  Mat DF_matrix;

  KSP LS_solver;

}AAR_LSQR_SOLVER;

typedef struct
{

  PetscInt M, N, lprank, m_aar, p_aar, num_iterations;
  PetscScalar omega, tol;
  Mat *A;
  Vec x_guess, x_old, x_new, x_final, res_old, res_new, Ax, res_old_rescaled, res_new_rescaled, prec_res_old, prec_res_new;

  const PetscScalar *df_col;
  const PetscScalar *df_col2;
  const PetscScalar *dres;
  PetscScalar *FtF_temp, *FtF, *Ftf, *svec;

  Vec *DX, *DF;
  Vec dp, dp1, dp2;

}AAR_NE_SOLVER;

typedef struct
{

  PetscInt iterations;
  PetscScalar residual_norm;

}AAR_OUTPUT;

void ReadInputFile(AAR_PARAMS*, char*);

void BuildPreconditioner(PC*, Mat*, char*);

void AAR_LSQR_SetUp(AAR_LSQR_SOLVER*, Mat*, Vec*, PC*, PetscInt, PetscInt, PetscReal, PetscInt, PetscReal);

void AAR_LSQR_Solver_Destroy(AAR_LSQR_SOLVER*);

void AAR_NE_SetUp(AAR_NE_SOLVER*, Mat*, Vec*, PC*, PetscInt, PetscInt, PetscReal, PetscInt, PetscReal);

void AAR_NE_Solver_Destroy(AAR_NE_SOLVER*);

AAR_OUTPUT AAR_LSQR_Solve(AAR_LSQR_SOLVER*, Vec*, PC*);

AAR_OUTPUT AAR_NE_Solve(AAR_NE_SOLVER*, Vec*, PC*);

static PetscErrorCode PCSetFromOptions_HYPRE_Pilut();
static PetscErrorCode PCSetFromOptions_HYPRE_Parasails();

int main( int argc, char **argv )
{
  int ierr; 
  PetscInt number_runs = 5;
  PetscInt restarted_gmres1_iterations, restarted_gmres2_iterations;
  PetscReal t_aar_init = 0.0, t_aar_final = 0.0, t_gmres_init = 0.0, t_gmres_final = 0.0, t_aar_lsqr_TOT = 0.0, t_aar_ne_TOT = 0.0, t_gmres1_TOT = 0.0, t_gmres2_TOT = 0.0;
  PetscInt gmres1_iter_TOT = 0.0, gmres2_iter_TOT = 0.0, aar_lsqr_iter_TOT = 0.0, aar_ne_iter_TOT = 0.0;
  PetscReal norm_infA;
  PetscReal rel_res_gmres1, rel_res_gmres2;
  PetscReal rel_res_aar_lsqr, rel_res_aar_ne;
  AAR_PARAMS aar_parameters;
  AAR_OUTPUT aar_lsqr_output, aar_ne_output;
  char matrix_file[30]="";

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsSetValue("-pc_hypre_pilut_maxiter","1"); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-pc_hypre_pilut_tol","0.05"); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-pc_hypre_parasails_nlevels","1"); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-pc_hypre_parasails_thresh","0.05"); CHKERRQ(ierr);
  PetscOptionsInsert(&argc,&argv,PETSC_NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  ReadInputFile(&aar_parameters, matrix_file);

  PetscViewer    view_out,view_in, view_matrix_out;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,matrix_file,FILE_MODE_READ,&view_in);
  Mat A;
  MatCreate(PETSC_COMM_WORLD,&A);
  MatSetType(A,MATAIJ);
  MatLoad(A,view_in);
  PetscViewerDestroy(&view_in);
  MatNorm(A,NORM_INFINITY,&norm_infA);
 
  if(strcmp(aar_parameters.prec_type, "none")==0)
    aar_parameters.omega = 2/norm_infA;
  else
    aar_parameters.omega = 0.2;

  PetscInt M, N;
  MatGetSize(A,&M,&N);

  Vec x,b,x_guess,x_final,x_gmres;
  VecCreate(PETSC_COMM_WORLD, &x);
  VecCreate(PETSC_COMM_WORLD, &x_guess);
  VecCreate(PETSC_COMM_WORLD, &x_final);
  VecCreate(PETSC_COMM_WORLD, &x_gmres);
  VecCreate(PETSC_COMM_WORLD, &b);
  VecSetSizes(x,PETSC_DECIDE,N);
  VecSetSizes(x_guess,PETSC_DECIDE,N);
  VecSetSizes(x_final,PETSC_DECIDE,N);
  VecSetSizes(x_gmres,PETSC_DECIDE,N);
  VecSetSizes(b,PETSC_DECIDE,M);
  VecSetFromOptions(x);
  VecSetFromOptions(x_guess);
  VecSetFromOptions(x_final);
  VecSetFromOptions(x_gmres);
  VecSetFromOptions(b);

  PetscScalar t_prec_construction_init = 0.0, t_prec_construction_final = 0.0;

  //Preconditioner construction
  PC prec;
  t_prec_construction_init = MPI_Wtime();
  BuildPreconditioner(&prec, &A, aar_parameters.prec_type);
  t_prec_construction_final = MPI_Wtime();

  //Computation of preconditioned right-hand side
  Vec prec_b;
  VecCreate(PETSC_COMM_WORLD, &prec_b);
  VecSetSizes(prec_b,PETSC_DECIDE,M);
  VecSetFromOptions(prec_b);

  //Restarted GMRES construction
  KSP restarted_gmres1;
  KSPCreate(PETSC_COMM_WORLD, &restarted_gmres1);
  KSPGMRESSetRestart(restarted_gmres1,10);
  KSPSetTolerances(restarted_gmres1,aar_parameters.solver_tol,PETSC_DEFAULT,PETSC_DEFAULT,aar_parameters.maxit);
  KSPSetOperators(restarted_gmres1,A,A);
  KSPSetPC(restarted_gmres1,prec);
  KSPSetFromOptions(restarted_gmres1);

  KSP restarted_gmres2;
  KSPCreate(PETSC_COMM_WORLD, &restarted_gmres2);
  KSPGMRESSetRestart(restarted_gmres2,30);
  KSPSetTolerances(restarted_gmres2,aar_parameters.solver_tol,PETSC_DEFAULT,PETSC_DEFAULT,aar_parameters.maxit);
  KSPSetOperators(restarted_gmres2,A,A);
  KSPSetPC(restarted_gmres2,prec);
  KSPSetFromOptions(restarted_gmres2);

  //Set initial guess equal to zero vector
  VecSet(x_guess, 0.0);

  //Set up the AAR_LSQR solver
  AAR_LSQR_SOLVER aar_lsqr_solver;
  AAR_LSQR_SetUp(&aar_lsqr_solver, &A, &x_guess, &prec, 6, 12, aar_parameters.omega, aar_parameters.maxit, aar_parameters.solver_tol);

  //Set up the AAR_NE solver
  AAR_NE_SOLVER aar_ne_solver;
  AAR_NE_SetUp(&aar_ne_solver, &A, &x_guess, &prec, 6, 12, aar_parameters.omega, aar_parameters.maxit, aar_parameters.solver_tol);

  PetscRandom rnd;
  unsigned int long seed;
  PetscRandomCreate(PETSC_COMM_WORLD, &rnd);
  PetscRandomSetFromOptions(rnd);
  seed = world_rank;
  PetscRandomSetSeed(rnd,seed);
  PetscRandomSeed(rnd);

  for (PetscInt n_run = 0; n_run < number_runs; ++n_run)
  {
    PetscPrintf(PETSC_COMM_WORLD, " ------------------------- \n Run Index: %d \n ---------------------------------------", n_run);
    //PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n CONSTRUCTION OF THE PRECONDITIONED RHS \n --------------------------------------------");
    VecSetRandom(x,rnd);
    //VecSetRandom(x_guess,rnd);
    MatMult(A,x,b);
    PetscReal b_norm;
    VecNorm(b,NORM_2,&b_norm);
    PCApply(prec,b,prec_b);
    PetscReal prec_b_norm;
    VecNorm(prec_b,NORM_2,&prec_b_norm);

    PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n RESTARTED GMRES(10) \n --------------------------------------------");

    t_gmres_init=MPI_Wtime();
    if(aar_parameters.gmres_compute == 1)
      KSPSolve(restarted_gmres1,b,x_gmres);
    t_gmres_final=MPI_Wtime();
    KSPGetIterationNumber(restarted_gmres1,&restarted_gmres1_iterations);
    KSPGetResidualNorm(restarted_gmres1,&rel_res_gmres1);
    rel_res_gmres1 = rel_res_gmres1 / prec_b_norm;
    t_gmres1_TOT += (t_gmres_final - t_gmres_init);
    gmres1_iter_TOT += restarted_gmres1_iterations;


    PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n RESTARTED GMRES(30) \n --------------------------------------------");

    t_gmres_init=MPI_Wtime();
    if(aar_parameters.gmres_compute == 1)
      KSPSolve(restarted_gmres2,b,x_gmres);
    t_gmres_final=MPI_Wtime();
    KSPGetIterationNumber(restarted_gmres2,&restarted_gmres2_iterations);
    KSPGetResidualNorm(restarted_gmres2,&rel_res_gmres2);
    rel_res_gmres2 = rel_res_gmres2 / prec_b_norm;
    t_gmres2_TOT += (t_gmres_final - t_gmres_init);
    gmres2_iter_TOT += restarted_gmres2_iterations;


    PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n AAR_LSQR \n --------------------------------------------");
    t_aar_init=MPI_Wtime();
    aar_lsqr_output = AAR_LSQR_Solve (&aar_lsqr_solver, &b, &prec);
    t_aar_final=MPI_Wtime();
    rel_res_aar_lsqr = aar_lsqr_output.residual_norm / prec_b_norm;
    t_aar_lsqr_TOT += (t_aar_final - t_aar_init);
    aar_lsqr_iter_TOT += aar_lsqr_output.iterations;

    PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n AAR_NE \n --------------------------------------------");
    t_aar_init=MPI_Wtime();
    aar_ne_output = AAR_NE_Solve (&aar_ne_solver, &b, &prec);
    t_aar_final=MPI_Wtime();
    rel_res_aar_ne = aar_ne_output.residual_norm / prec_b_norm;
    t_aar_ne_TOT += (t_aar_final - t_aar_init);
    aar_ne_iter_TOT += aar_ne_output.iterations;
  }

  PetscPrintf(PETSC_COMM_WORLD, "Construction of the preconditioner: Time elapsed: %f (s)\n", t_prec_construction_final-t_prec_construction_init);
  PetscPrintf(PETSC_COMM_WORLD, "AAR_LSQR - Number of iterations: %d\nFinal relative residual norm: %.9f\n Time elapsed: %f (s)\n", aar_lsqr_iter_TOT/number_runs, rel_res_aar_lsqr, t_aar_lsqr_TOT/number_runs);
  PetscPrintf(PETSC_COMM_WORLD, "AAR_NE - Number of iterations: %d\nFinal relative residual norm: %.9f\n Time elapsed: %f (s)\n", aar_ne_iter_TOT/number_runs, rel_res_aar_ne, t_aar_ne_TOT/number_runs);
  PetscPrintf(PETSC_COMM_WORLD, "Restarted GMRES(10) - Number of iterations: %d\nFinal relative residual norm: %.9f\n Time elapsed: %f (s)\n", gmres1_iter_TOT/number_runs, rel_res_gmres1, t_gmres1_TOT/number_runs);
  PetscPrintf(PETSC_COMM_WORLD, "Restarted GMRES(30) - Number of iterations: %d\nFinal relative residual norm: %.9f\n Time elapsed: %f (s)\n", gmres2_iter_TOT/number_runs, rel_res_gmres2, t_gmres2_TOT/number_runs);

  VecDestroy(&x);
  VecDestroy(&x_guess);
  VecDestroy(&x_final);
  VecDestroy(&b);
  VecDestroy(&prec_b);
  MatDestroy(&A);
  PCDestroy(&prec);
  KSPDestroy(&restarted_gmres1);
  KSPDestroy(&restarted_gmres2);
  AAR_LSQR_Solver_Destroy(&aar_lsqr_solver);
  AAR_NE_Solver_Destroy(&aar_ne_solver);
  PetscRandomDestroy(&rnd);

  ierr = PetscFinalize();CHKERRQ(ierr);
 
  return 0;
}
 
void ReadInputFile(AAR_PARAMS* params, char *matrix_file)
{
  PetscBool flg;
  FILE* fConfFile;
  char ConfigFile[20];
  char matrix_name[20];
  strcat(matrix_file, "./matrix_collection/");

  PetscOptionsGetString(PETSC_NULL,"-name",ConfigFile,sizeof(ConfigFile),&flg);
  strcat(ConfigFile,".in");

  if((fConfFile = fopen((const char*)ConfigFile, "rb"))==NULL)
  {
    PetscPrintf(PETSC_COMM_WORLD,"Could not open input file... Exiting...\n");
    exit(1);
  }

  char str[60];

  do
    {
      fscanf(fConfFile, "%s", str);
      if(strcmp(str, "matrix_file:")==0)
      {
        fscanf(fConfFile,"%s",matrix_name);    
        strcat(matrix_file, matrix_name);
      }else if(strcmp(str, "prec_type:")==0)
      {
        fscanf(fConfFile,"%s",&params->prec_type);
      }else if(strcmp(str, "m_aar:")==0)
      {
        fscanf(fConfFile,"%d",&params->m_aar);    
      }else if(strcmp(str, "p_aar:")==0)
      {
        fscanf(fConfFile,"%d",&params->p_aar);    
      }else if(strcmp(str, "solver_tol:")==0)
      {
        fscanf(fConfFile,"%lf",&params->solver_tol);    
      }else if(strcmp(str, "maxit:")==0)
      {
        fscanf(fConfFile,"%d",&params->maxit);    
      }
      else if(strcmp(str, "gmres:")==0)
      {
        fscanf(fConfFile,"%d",&params->gmres_compute);    
      }
    }while(!feof(fConfFile));

  strcat(matrix_file, ".dat");

}

void BuildPreconditioner(PC* prec, Mat* A, char* prec_type)
{

  PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n CONSTRUCTION OF THE PRECONDITIONER STARTED \n --------------------------------------------");

  PetscInt M, N;
  MatGetSize(*A,&M,&N);

  PCCreate(PETSC_COMM_WORLD, prec);
  if(strcmp(prec_type, "block_jacobi")==0)
  {
    PetscPrintf(PETSC_COMM_WORLD, " -------------------------------------- \n BLOCK JACOBI preconditioner \n --------------------------------------------");
    PCSetType(*prec, PCPBJACOBI);
    PCBJacobiSetTotalBlocks(*prec, 1024, NULL);
    PCSetOperators(*prec,*A,*A);
    PCSetFromOptions(*prec);
    PCSetUp(*prec);
  }
  else if(strcmp(prec_type, "amg")==0)
  {
    PetscBool flag = PETSC_TRUE;
    PetscPrintf(PETSC_COMM_WORLD, " ----------------------------------------- \n AMG preconditioner \n --------------------------------------------");
    PCSetType(*prec, PCGAMG);
    PCGAMGSetNSmooths(*prec,0); //Unsmoothed aggregation
    PCGAMGSetNlevels(*prec,5);
    ////PCGAMGSetRepartition(*prec, flag);
    ////PCGAMGSetUseParallelCoarseGridSolve(*prec, flag);
    PCSetOperators(*prec,*A,*A);
    PCSetFromOptions(*prec);
    PCSetUp(*prec);
  }
  else if(strcmp(prec_type, "asm")==0)
  {
    PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n ASM preconditioner \n --------------------------------------------");
    PCSetType(*prec, PCASM);
    PCASMSetTotalSubdomains(*prec, 1024, NULL, NULL);
    ////PCASMSetType(*prec, PC_ASM_RESTRICT); // By default ASM is already restricted
    PCASMSetOverlap(*prec,1);
    PCSetOperators(*prec,*A,*A);
    PCSetFromOptions(*prec);
    PCSetUp(*prec);
  }
  else if(strcmp(prec_type, "ilut")==0)
  { 
    PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n ILUT preconditioner \n --------------------------------------------");
    PCSetType(*prec, PCHYPRE);
    PCHYPRESetType(*prec,"pilut");
    PCSetOperators(*prec,*A,*A);
    PCSetFromOptions(*prec);
    //PCSetFromOptions_HYPRE_Pilut();
    PCSetUp(*prec);
  }
  else if(strcmp(prec_type, "spai")==0)
  { 
    PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n SPAI preconditioner \n --------------------------------------------");
    PCSetType(*prec, PCHYPRE);
    PCHYPRESetType(*prec,"parasails");
    PCSetOperators(*prec,*A,*A);
    PCSetFromOptions(*prec);
    //PCSetFromOptions_HYPRE_Parasails();
    PCSetUp(*prec);
  }
  else
  { 
    PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n NO preconditioner \n --------------------------------------------");
    PCSetType(*prec,PCNONE);
    PCSetOperators(*prec,*A,*A);
    PCSetUp(*prec);
  }

  PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n CONSTRUCTION OF THE PRECONDITIONER FINISHED \n --------------------------------------------");

}

static PetscErrorCode PCSetFromOptions_HYPRE_Pilut()
{
   PetscBool      flag;
   PetscInt       maxiter = 1;
   PetscReal      drop_tol = 0.9;
   PetscErrorCode ierr;

   //PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the ILUT preconditioner", "");
   ierr = PetscOptionsSetValue("-pc_hypre_pilut_maxiter","1"); CHKERRQ(ierr);
   ierr = PetscOptionsSetValue("-pc_hypre_pilut_tol","0.9"); CHKERRQ(ierr);
   //PetscOptionsEnd();

   return(0);
}


static PetscErrorCode PCSetFromOptions_HYPRE_Parasails()
{
   PetscBool      flag;
   PetscInt       nlevels = 5;
   PetscReal      drop_tol = 0.01;
   PetscErrorCode ierr;

   //PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the SPAI preconditioner", "");
   ierr = PetscOptionsSetValue("-pc_hypre_parasails_nlevels","5"); CHKERRQ(ierr);
   ierr = PetscOptionsSetValue("-pc_hypre_parasails_thresh","0.01"); CHKERRQ(ierr);
   //PetscOptionsEnd();

   return(0);
}

void AAR_LSQR_SetUp(AAR_LSQR_SOLVER* solver, Mat* A, Vec* x_guess, PC* prec, PetscInt m_aar, PetscInt p_aar, PetscReal omega, PetscInt num_iterations, PetscReal solver_tol)
{

  solver->A = A;
  solver->m_aar = m_aar;
  solver->p_aar = p_aar;
  solver->omega = omega;
  solver->num_iterations = num_iterations;
  solver->tol = solver_tol;

  MatGetSize(*A,&solver->M,&solver->N);

  PetscInt local_nrows, low, high;
  VecGetLocalSize(*x_guess, &local_nrows);
  VecGetOwnershipRange(*x_guess, &low, &high);

  solver->row_inds = new PetscInt [local_nrows]();
  solver->row_inds_local = new PetscInt [local_nrows]();
  solver->col_inds = new PetscInt [p_aar]();
  solver->gamma_array = new PetscScalar[p_aar]();

  for(PetscInt index = 0; index<p_aar; ++index)
    solver->col_inds[index]=index;
  for(PetscInt index = 0; index<local_nrows; ++index)
  {
    solver->row_inds[index]=low+index;        
    solver->row_inds_local[index]=index;
  }

  MatCreate(PETSC_COMM_WORLD, &solver->DF_matrix);
  MatSetSizes(solver->DF_matrix,PETSC_DECIDE,p_aar,solver->M,p_aar);
  MatSetType(solver->DF_matrix,MATAIJ);
  MatMPIDenseSetPreallocation(solver->DF_matrix,PETSC_NULL);
  MatSetUp(solver->DF_matrix);
  PetscInt row_low, row_high, col_low, col_high;
  MatGetOwnershipRange(solver->DF_matrix,&row_low,&row_high);
  MatGetOwnershipRangeColumn(solver->DF_matrix,&col_low,&col_high);
  ISLocalToGlobalMapping row_map;
  ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,row_high-row_low,solver->row_inds,PETSC_COPY_VALUES,&row_map);
  ISLocalToGlobalMapping col_map;
  ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,col_high-col_low,solver->col_inds,PETSC_COPY_VALUES,&col_map);
  MatSetLocalToGlobalMapping(solver->DF_matrix,row_map,col_map);

  KSPCreate(PETSC_COMM_WORLD, &solver->LS_solver);
  KSPSetType(solver->LS_solver,KSPLSQR);
  PC prec_inner_solver;
  KSPSetTolerances(solver->LS_solver,1e-16,PETSC_DEFAULT,PETSC_DEFAULT,p_aar);
  KSPGetPC(solver->LS_solver,&prec_inner_solver);
  PCSetType(prec_inner_solver,PCNONE);

  VecCreate(PETSC_COMM_WORLD, &solver->x_guess);
  VecCreate(PETSC_COMM_WORLD, &solver->x_old);
  VecCreate(PETSC_COMM_WORLD, &solver->x_new);
  VecCreate(PETSC_COMM_WORLD, &solver->x_final);
  VecCreate(PETSC_COMM_WORLD, &solver->res_old);
  VecCreate(PETSC_COMM_WORLD, &solver->res_new);
  VecCreate(PETSC_COMM_WORLD, &solver->prec_res_old);
  VecCreate(PETSC_COMM_WORLD, &solver->prec_res_new);
  VecCreate(PETSC_COMM_WORLD, &solver->Ax);
  VecCreate(PETSC_COMM_WORLD, &solver->res_old_rescaled);
  VecCreate(PETSC_COMM_WORLD, &solver->res_new_rescaled);
  VecSetSizes(solver->x_guess,PETSC_DECIDE,solver->N);
  VecSetSizes(solver->x_old,PETSC_DECIDE,solver->N);
  VecSetSizes(solver->x_new,PETSC_DECIDE,solver->N);
  VecSetSizes(solver->x_final,PETSC_DECIDE,solver->N);
  VecSetSizes(solver->res_old,PETSC_DECIDE,solver->M);
  VecSetSizes(solver->res_new,PETSC_DECIDE,solver->M);
  VecSetSizes(solver->prec_res_old,PETSC_DECIDE,solver->M);
  VecSetSizes(solver->prec_res_new,PETSC_DECIDE,solver->M);
  VecSetSizes(solver->Ax,PETSC_DECIDE,solver->M);
  VecSetSizes(solver->res_old_rescaled,PETSC_DECIDE,solver->M);
  VecSetSizes(solver->res_new_rescaled,PETSC_DECIDE,solver->M);
  VecSetFromOptions(solver->x_guess);
  VecSetFromOptions(solver->x_old);
  VecSetFromOptions(solver->x_new);
  VecSetFromOptions(solver->x_final);
  VecSetFromOptions(solver->res_old);
  VecSetFromOptions(solver->res_new);
  VecSetFromOptions(solver->prec_res_old);
  VecSetFromOptions(solver->prec_res_new);
  VecSetFromOptions(solver->Ax);
  VecSetFromOptions(solver->res_old_rescaled);
  VecSetFromOptions(solver->res_new_rescaled);
  VecSet(solver->x_old, 0.0);

  VecCreate(PETSC_COMM_WORLD,&solver->gamma);
  VecSetSizes(solver->gamma,p_aar,p_aar);
  VecSetFromOptions(solver->gamma);

  solver->DX = new Vec[p_aar]();
  solver->DF = new Vec[p_aar]();
 
  for(PetscInt vec_count=0; vec_count<p_aar; ++vec_count)
  {
    VecCreate(PETSC_COMM_WORLD, &solver->DX[vec_count]);
    VecCreate(PETSC_COMM_WORLD, &solver->DF[vec_count]);
    VecSetSizes(solver->DX[vec_count],PETSC_DECIDE,solver->N);
    VecSetSizes(solver->DF[vec_count],PETSC_DECIDE,solver->M);
    VecSetFromOptions(solver->DX[vec_count]);
    VecSetFromOptions(solver->DF[vec_count]);
    VecSet(solver->DX[vec_count], 0.0);
    VecSet(solver->DF[vec_count], 0.0);
  }
  for( PetscInt col_index=0; col_index<solver->p_aar; ++col_index )
  {
    VecGetArrayRead(solver->DF[col_index],&solver->df_col);   
    MatSetValuesLocal(solver->DF_matrix,local_nrows,solver->row_inds_local,1,&col_index,solver->df_col,INSERT_VALUES);
    VecRestoreArrayRead(solver->DF[col_index],&solver->df_col);   
  }
  MatAssemblyBegin(solver->DF_matrix,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(solver->DF_matrix,MAT_FINAL_ASSEMBLY);

  VecCreate(PETSC_COMM_WORLD, &solver->dp);
  VecSetSizes(solver->dp, PETSC_DECIDE, solver->M);
  VecSetFromOptions(solver->dp);
  VecSet(solver->dp,0.0);
  VecCreate(PETSC_COMM_WORLD, &solver->dp1);
  VecSetSizes(solver->dp1, PETSC_DECIDE, solver->M);
  VecSetFromOptions(solver->dp1);
  VecSet(solver->dp1,0.0);
  VecCreate(PETSC_COMM_WORLD, &solver->dp2);
  VecSetSizes(solver->dp2, PETSC_DECIDE, solver->M);
  VecSetFromOptions(solver->dp2);
  VecSet(solver->dp2,0.0);

  //Set the initial guess
  VecCopy(solver->x_guess, *x_guess);

}

void AAR_NE_SetUp(AAR_NE_SOLVER* solver, Mat* A, Vec* x_guess, PC* prec, PetscInt m_aar, PetscInt p_aar, PetscReal omega, PetscInt num_iterations, PetscReal solver_tol)
{

  solver->A = A;
  solver->m_aar = m_aar;
  solver->p_aar = p_aar;
  solver->omega = omega;
  solver->num_iterations = num_iterations;
  solver->tol = solver_tol;

  MatGetSize(*A,&solver->M,&solver->N);

  PetscInt local_nrows, low, high;
  VecGetLocalSize(*x_guess, &local_nrows);
  VecGetOwnershipRange(*x_guess, &low, &high);

  solver->FtF = new PetscScalar [p_aar*p_aar]();
  solver->Ftf = new PetscScalar [p_aar]();
  solver->FtF_temp = new PetscScalar [p_aar*p_aar + p_aar]();
  solver->svec = new PetscScalar [p_aar]();

  VecCreate(PETSC_COMM_WORLD, &solver->x_guess);
  VecCreate(PETSC_COMM_WORLD, &solver->x_old);
  VecCreate(PETSC_COMM_WORLD, &solver->x_new);
  VecCreate(PETSC_COMM_WORLD, &solver->x_final);
  VecCreate(PETSC_COMM_WORLD, &solver->res_old);
  VecCreate(PETSC_COMM_WORLD, &solver->res_new);
  VecCreate(PETSC_COMM_WORLD, &solver->prec_res_old);
  VecCreate(PETSC_COMM_WORLD, &solver->prec_res_new);
  VecCreate(PETSC_COMM_WORLD, &solver->Ax);
  VecCreate(PETSC_COMM_WORLD, &solver->res_old_rescaled);
  VecCreate(PETSC_COMM_WORLD, &solver->res_new_rescaled);
  VecSetSizes(solver->x_guess,PETSC_DECIDE,solver->N);
  VecSetSizes(solver->x_old,PETSC_DECIDE,solver->N);
  VecSetSizes(solver->x_new,PETSC_DECIDE,solver->N);
  VecSetSizes(solver->x_final,PETSC_DECIDE,solver->N);
  VecSetSizes(solver->res_old,PETSC_DECIDE,solver->M);
  VecSetSizes(solver->res_new,PETSC_DECIDE,solver->M);
  VecSetSizes(solver->prec_res_old,PETSC_DECIDE,solver->M);
  VecSetSizes(solver->prec_res_new,PETSC_DECIDE,solver->M);
  VecSetSizes(solver->Ax,PETSC_DECIDE,solver->M);
  VecSetSizes(solver->res_old_rescaled,PETSC_DECIDE,solver->M);
  VecSetSizes(solver->res_new_rescaled,PETSC_DECIDE,solver->M);
  VecSetFromOptions(solver->x_guess);
  VecSetFromOptions(solver->x_old);
  VecSetFromOptions(solver->x_new);
  VecSetFromOptions(solver->x_final);
  VecSetFromOptions(solver->res_old);
  VecSetFromOptions(solver->res_new);
  VecSetFromOptions(solver->prec_res_old);
  VecSetFromOptions(solver->prec_res_new);
  VecSetFromOptions(solver->Ax);
  VecSetFromOptions(solver->res_old_rescaled);
  VecSetFromOptions(solver->res_new_rescaled);
  VecSet(solver->x_old, 0.0);

  solver->DX = new Vec[p_aar]();
  solver->DF = new Vec[p_aar]();
 
  for(PetscInt vec_count=0; vec_count<p_aar; ++vec_count)
  {
    VecCreate(PETSC_COMM_WORLD, &solver->DX[vec_count]);
    VecCreate(PETSC_COMM_WORLD, &solver->DF[vec_count]);
    VecSetSizes(solver->DX[vec_count],PETSC_DECIDE,solver->N);
    VecSetSizes(solver->DF[vec_count],PETSC_DECIDE,solver->M);
    VecSetFromOptions(solver->DX[vec_count]);
    VecSetFromOptions(solver->DF[vec_count]);
  }

  VecCreate(PETSC_COMM_WORLD, &solver->dp);
  VecSetSizes(solver->dp, PETSC_DECIDE, solver->M);
  VecSetFromOptions(solver->dp);
  VecSet(solver->dp,0.0);
  VecCreate(PETSC_COMM_WORLD, &solver->dp1);
  VecSetSizes(solver->dp1, PETSC_DECIDE, solver->M);
  VecSetFromOptions(solver->dp1);
  VecSet(solver->dp1,0.0);
  VecCreate(PETSC_COMM_WORLD, &solver->dp2);
  VecSetSizes(solver->dp2, PETSC_DECIDE, solver->M);
  VecSetFromOptions(solver->dp2);
  VecSet(solver->dp2,0.0);

  //Set the initial guess
  VecCopy(solver->x_guess, *x_guess);

}


AAR_OUTPUT AAR_LSQR_Solve (AAR_LSQR_SOLVER* solver, Vec* b, PC* prec)
{

  PetscReal t_ls_init, t_ls_final, t_ls_tot=0.0;
  PetscReal t_rich_init, t_rich_final, t_rich_tot=0.0;
  PetscReal t_res_init, t_res_final, t_res_tot=0.0;

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  AAR_OUTPUT output;

  Vec prec_b;
  VecCreate(PETSC_COMM_WORLD, &prec_b);
  VecSetSizes(prec_b,PETSC_DECIDE,solver->M);
  VecSetFromOptions(prec_b);
  PCApply(*prec,*b,prec_b);
  PetscReal prec_b_norm;
  VecNorm(prec_b,NORM_2,&prec_b_norm);

  PetscInt res_vec_low, res_vec_high;
  PetscInt local_nrows;
  VecGetLocalSize(solver->res_new_rescaled, &local_nrows);
  VecGetOwnershipRange(solver->res_new_rescaled,&res_vec_low,&res_vec_high);

  // Now we proceed with the Richardson sweeps
  PetscInt iter_count, update_count;
  PetscBool anderson_mixing_computed = PETSC_FALSE;

  PetscReal res_norm = 1;

  iter_count = 1;
  update_count = 1;

  VecCopy(solver->x_guess, solver->x_old);

  MatMult(*(solver->A),solver->x_old,solver->Ax);
  VecWAXPY(solver->res_old,-1.0,solver->Ax,*b);
  PCApply(*prec,solver->res_old,solver->prec_res_old);
  VecWAXPY(solver->x_new,solver->omega,solver->prec_res_old,solver->x_old);
  MatMult(*(solver->A),solver->x_new,solver->Ax);
  VecWAXPY(solver->res_new,-1.0,solver->Ax,*b);
  PCApply(*prec,solver->res_new,solver->prec_res_new);
  VecNorm(solver->prec_res_new,NORM_2,&res_norm);

  //If the AAR object is used multiple times, everything must be set to zero at the first iteration
  for(PetscInt vec_count=0; vec_count<solver->p_aar; ++vec_count)
  {
    VecSet(solver->DX[vec_count], 0.0);
    VecSet(solver->DF[vec_count], 0.0);
  }
  VecSet(solver->gamma, 0.0);

  PetscReal t_aar_init, t_aar_final;

  t_aar_init = MPI_Wtime();

  while( res_norm/prec_b_norm > solver->tol && iter_count <= solver->num_iterations )
  {
    
    //PetscPrintf(PETSC_COMM_WORLD, "AAR iter - %d - rel_res_norm: %0.13f\n", iter_count, res_norm/prec_b_norm);

    if((iter_count)%solver->m_aar!=0 || anderson_mixing_computed)
    {
      t_rich_init = MPI_Wtime();      

      //Computation of the residual
      VecSet(solver->res_old_rescaled, 0.0);
      VecSet(solver->res_new_rescaled, 0.0);
      VecAXPY(solver->res_old_rescaled,solver->omega,solver->prec_res_old);
      VecAXPY(solver->res_new_rescaled,solver->omega,solver->prec_res_new);
      VecWAXPY(solver->DX[(update_count-1)%solver->p_aar],-1.0,solver->x_old,solver->x_new);
      VecWAXPY(solver->DF[(update_count-1)%solver->p_aar],-1.0,solver->res_old_rescaled,solver->res_new_rescaled);

      VecCopy(solver->x_new, solver->x_old);
      VecCopy(solver->res_new, solver->res_old);
      VecCopy(solver->prec_res_new, solver->prec_res_old);
      VecWAXPY(solver->x_new,solver->omega,solver->prec_res_old,solver->x_old);

      if(anderson_mixing_computed)
         anderson_mixing_computed = PETSC_FALSE;

      t_rich_final = MPI_Wtime();
      t_rich_tot += t_rich_final - t_rich_init;

    }
    else if(iter_count%solver->m_aar==0)
    {

      t_ls_init = MPI_Wtime();

      for( PetscInt col_index=0; col_index<solver->p_aar; ++col_index )
      {
        VecGetArrayRead(solver->DF[col_index],&solver->df_col);   
        MatSetValuesLocal(solver->DF_matrix,local_nrows,solver->row_inds_local,1,&col_index,solver->df_col,INSERT_VALUES);
        VecRestoreArrayRead(solver->DF[col_index],&solver->df_col);   
      }

      MatAssemblyBegin(solver->DF_matrix,MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(solver->DF_matrix,MAT_FINAL_ASSEMBLY);

      KSPSetOperators(solver->LS_solver,solver->DF_matrix,solver->DF_matrix);
      KSPSolve(solver->LS_solver,solver->res_new_rescaled,solver->gamma);

      if(world_rank == 0 )
          VecGetValues(solver->gamma,solver->p_aar,solver->col_inds,solver->gamma_array);

      MPI_Bcast( solver->gamma_array, solver->p_aar, MPI_DOUBLE, 0, PETSC_COMM_WORLD); 

      VecSet(solver->dp1, 0.0);
      VecSet(solver->dp2, 0.0);
      VecMAXPY(solver->dp1,solver->p_aar,solver->gamma_array,solver->DX);
      VecMAXPY(solver->dp2,solver->p_aar,solver->gamma_array,solver->DF);
      VecWAXPY(solver->dp,1.0,solver->dp1,solver->dp2);
      VecAXPY(solver->x_new,-1.0,solver->dp);

      if(iter_count == solver->m_aar)
        anderson_mixing_computed = PETSC_TRUE;

      t_ls_final = MPI_Wtime();
      t_ls_tot += (t_ls_final-t_ls_init);

    }

    t_res_init = MPI_Wtime();
    //Computation of the residual
    MatMult(*solver->A,solver->x_new,solver->Ax);
    VecWAXPY(solver->res_new,-1.0,solver->Ax,*b);
    PCApply(*prec,solver->res_new,solver->prec_res_new);
    t_res_final = MPI_Wtime();
    t_res_tot += t_res_final - t_res_init;

    if(iter_count%solver->m_aar == 0 )
      VecNorm(solver->prec_res_new,NORM_2,&res_norm);

    if(anderson_mixing_computed)
      update_count--;

    iter_count ++;
    update_count ++;

  }
 
  t_aar_final = MPI_Wtime();

  VecCopy(solver->x_new, solver->x_final);

  output.residual_norm = res_norm;
  output.iterations = iter_count - 1;

  return output;
}

AAR_OUTPUT AAR_NE_Solve (AAR_NE_SOLVER* solver, Vec* b, PC* prec)
{

  PetscReal t_ls_init, t_ls_final, t_ls_tot=0.0;
  PetscReal t_rich_init, t_rich_final, t_rich_tot=0.0;
  PetscReal t_res_init, t_res_final, t_res_tot=0.0;

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  AAR_OUTPUT output;

  PetscInt local_nrows, low, high;
  VecGetLocalSize(*b,&local_nrows);
  VecGetOwnershipRange(*b, &low, &high);

  Vec prec_b;
  VecCreate(PETSC_COMM_WORLD, &prec_b);
  VecSetSizes(prec_b,PETSC_DECIDE,solver->M);
  VecSetFromOptions(prec_b);
  PCApply(*prec,*b,prec_b);
  PetscReal prec_b_norm;
  VecNorm(prec_b,NORM_2,&prec_b_norm);

  PetscInt res_vec_low, res_vec_high;
  VecGetOwnershipRange(solver->res_new_rescaled,&res_vec_low,&res_vec_high);

  // Now we proceed with the Richardson sweeps
  PetscInt iter_count, update_count;
  PetscBool anderson_mixing_computed = PETSC_FALSE;

  PetscReal res_norm = 1;

  iter_count = 1;
  update_count = 1;

  VecCopy(solver->x_guess, solver->x_old);

  MatMult(*(solver->A),solver->x_old,solver->Ax);
  VecWAXPY(solver->res_old,-1.0,solver->Ax,*b);
  PCApply(*prec,solver->res_old,solver->prec_res_old);
  VecWAXPY(solver->x_new,solver->omega,solver->prec_res_old,solver->x_old);
  MatMult(*(solver->A),solver->x_new,solver->Ax);
  VecWAXPY(solver->res_new,-1.0,solver->Ax,*b);
  PCApply(*prec,solver->res_new,solver->prec_res_new);
  VecNorm(solver->prec_res_new,NORM_2,&res_norm);

  //If the AAR object is used multiple times, everything must be set to zero at the first iteration
  for(PetscInt vec_count=0; vec_count<solver->p_aar; ++vec_count)
  {
    VecSet(solver->DX[vec_count], 0.0);
    VecSet(solver->DF[vec_count], 0.0);
  }
  PetscInt col_index_update;

  PetscReal t_aar_init, t_aar_final;

  t_aar_init = MPI_Wtime();

  while( res_norm/prec_b_norm > solver->tol && iter_count <= solver->num_iterations )
  {
    
    //PetscPrintf(PETSC_COMM_WORLD, "AAR iter - %d - rel_res_norm: %0.13f\n", iter_count, res_norm/prec_b_norm);

    if((iter_count)%solver->m_aar!=0 || anderson_mixing_computed)
    {
      t_rich_init = MPI_Wtime();      

      //Computation of the residual
      VecSet(solver->res_old_rescaled, 0.0);
      VecSet(solver->res_new_rescaled, 0.0);
      VecAXPY(solver->res_old_rescaled,solver->omega,solver->prec_res_old);
      VecAXPY(solver->res_new_rescaled,solver->omega,solver->prec_res_new);
      VecWAXPY(solver->DX[(update_count-1)%solver->p_aar],-1.0,solver->x_old,solver->x_new);
      VecWAXPY(solver->DF[(update_count-1)%solver->p_aar],-1.0,solver->res_old_rescaled,solver->res_new_rescaled);

      VecGetArrayRead(solver->DF[(update_count-1)%solver->p_aar],&solver->df_col);   
      col_index_update = (update_count-1)%solver->p_aar;

      VecCopy(solver->x_new, solver->x_old);
      VecCopy(solver->res_new, solver->res_old);
      VecCopy(solver->prec_res_new, solver->prec_res_old);
      VecWAXPY(solver->x_new,solver->omega,solver->prec_res_old,solver->x_old);

      if(anderson_mixing_computed)
         anderson_mixing_computed = PETSC_FALSE;

      t_rich_final = MPI_Wtime();
      t_rich_tot += t_rich_final - t_rich_init;

    }
    else if(iter_count % solver->m_aar==0 && iter_count<solver->p_aar)
    {

      t_ls_init = MPI_Wtime();
      PetscScalar *FtF_temp1, *FtF1, *Ftf1, *svec1;

      VecGetArrayRead(solver->res_new_rescaled,&solver->dres);   

      FtF1 = new PetscScalar [(iter_count-1)*(iter_count-1)]();
      Ftf1 = new PetscScalar [iter_count-1]();
      FtF_temp1 = new PetscScalar [(iter_count-1)*(iter_count-1) + iter_count-1]();
      svec1 = new PetscScalar [iter_count-1]();

      for( PetscInt col_index=0; col_index< iter_count-1; ++col_index )
      {
        PetscInt vec_low, vec_high;
        VecGetOwnershipRange(solver->DF[col_index],&vec_low,&vec_high);
        VecGetArrayRead(solver->DF[col_index],&solver->df_col);   
        for( PetscInt col2_index=0; col2_index< iter_count-1; ++col2_index )
        { 
          VecGetArrayRead(solver->DF[col2_index],&solver->df_col2);   
          
          PetscScalar matmat_value = 0.0;

          for( PetscInt k = 0; k<vec_high-vec_low; ++k )
            matmat_value += solver->df_col[k]*solver->df_col2[k];

          VecRestoreArrayRead(solver->DF[col2_index],&solver->df_col2);   

          FtF_temp1[ col_index + col2_index*(iter_count-1) ] = matmat_value;
        }

        PetscScalar matvec_value = 0.0;
        for( PetscInt k = 0; k<vec_high-vec_low; ++k )
          matvec_value += solver->df_col[k]*solver->dres[k];

        FtF_temp1[(iter_count-1)*(iter_count-1)+col_index]=matvec_value;

        VecRestoreArrayRead(solver->DF[col_index],&solver->df_col);   
      }
      VecRestoreArrayRead(solver->res_new_rescaled,&solver->dres);   

      MPI_Allreduce(MPI_IN_PLACE, FtF_temp1, (iter_count-1)*(iter_count), MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

      PetscInt ctr = 0;
      PetscInt ctr_temp1 = 0;

      for(PetscInt j=0; j<iter_count-1; ++j)
      {
        for(PetscInt i=0; i<iter_count-1; ++i)
        {
          FtF1[ctr]=FtF_temp1[ctr_temp1];
          ctr+=1;
          ctr_temp1+=1;
        }
      }
      ctr=0;
      for(PetscInt j=0; j<iter_count-1; ++j)
      {
        Ftf1[ctr]=FtF_temp1[ctr_temp1];
        ctr+=1;
        ctr_temp1+=1;
      }

      LAPACKE_dgelsd(LAPACK_COL_MAJOR,iter_count-1,iter_count-1,1,FtF1,iter_count-1,Ftf1,iter_count-1,svec1,-1.0,&solver->lprank);

      VecMAXPY(solver->dp1,iter_count-1,Ftf1,solver->DX);
      VecMAXPY(solver->dp2,iter_count-1,Ftf1,solver->DF);
      VecWAXPY(solver->dp,1.0,solver->dp1,solver->dp2);
      VecAXPY(solver->x_new,-1.0,solver->dp);

      delete [] FtF_temp1;
      delete [] FtF1;
      delete [] Ftf1;
      delete [] svec1;

      if(iter_count == solver->m_aar)
        anderson_mixing_computed = PETSC_TRUE;

      t_ls_final = MPI_Wtime();
      t_ls_tot += (t_ls_final-t_ls_init);
 
    }
    else if(iter_count%solver->m_aar==0 && iter_count>=solver->p_aar)
    {

      t_ls_init = MPI_Wtime();

      VecGetArrayRead(solver->res_new_rescaled,&solver->dres);   

      for( PetscInt col_index=0; col_index<solver->p_aar; ++col_index )
      {
        PetscInt vec_low, vec_high;
        VecGetOwnershipRange(solver->DF[col_index],&vec_low,&vec_high);
        VecGetArrayRead(solver->DF[col_index],&solver->df_col);   
        for( PetscInt col2_index=0; col2_index<solver->p_aar; ++col2_index )
        { 
          VecGetArrayRead(solver->DF[col2_index],&solver->df_col2);   
          
          PetscScalar matmat_value = 0.0;

          for( PetscInt k = 0; k<vec_high-vec_low; ++k )
            matmat_value += solver->df_col[k]*solver->df_col2[k];

          VecRestoreArrayRead(solver->DF[col2_index],&solver->df_col2);   

          solver->FtF_temp[ col_index + col2_index*(solver->p_aar) ] = matmat_value;
        }

        PetscScalar matvec_value = 0.0;
        for( PetscInt k = 0; k<vec_high-vec_low; ++k )
          matvec_value += solver->df_col[k]*solver->dres[k];

        solver->FtF_temp[solver->p_aar*solver->p_aar+col_index]=matvec_value;

        VecRestoreArrayRead(solver->DF[col_index],&solver->df_col);   
      }
      VecRestoreArrayRead(solver->res_new_rescaled,&solver->dres);   

      MPI_Allreduce(MPI_IN_PLACE, solver->FtF_temp, solver->p_aar*(solver->p_aar+1), MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

      PetscInt ctr = 0;
      PetscInt ctr_temp = 0;

      for(PetscInt j=0; j<solver->p_aar; ++j)
      {
        for(PetscInt i=0; i<solver->p_aar; ++i)
        {
          solver->FtF[ctr]=solver->FtF_temp[ctr_temp];
          ctr+=1;
          ctr_temp+=1;
        }
      }
      ctr=0;
      for(PetscInt j=0; j<solver->p_aar; ++j)
      {
        solver->Ftf[ctr]=solver->FtF_temp[ctr_temp];
        ctr+=1;
        ctr_temp+=1;
      }

      LAPACKE_dgelsd(LAPACK_COL_MAJOR,solver->p_aar,solver->p_aar,1,solver->FtF,solver->p_aar,solver->Ftf,solver->p_aar,solver->svec,-1.0,&solver->lprank);

      VecSet(solver->dp1, 0.0);
      VecSet(solver->dp2, 0.0);
      VecMAXPY(solver->dp1,solver->p_aar,solver->Ftf,solver->DX);
      VecMAXPY(solver->dp2,solver->p_aar,solver->Ftf,solver->DF);
      VecWAXPY(solver->dp,1.0,solver->dp1,solver->dp2);
      VecAXPY(solver->x_new,-1.0,solver->dp);

      t_ls_final = MPI_Wtime();

      t_ls_tot += (t_ls_final-t_ls_init);

    }

    t_res_init = MPI_Wtime();
    //Computation of the residual
    MatMult(*solver->A,solver->x_new,solver->Ax);
    VecWAXPY(solver->res_new,-1.0,solver->Ax,*b);
    PCApply(*prec,solver->res_new,solver->prec_res_new);
    t_res_final = MPI_Wtime();
    t_res_tot += t_res_final - t_res_init;

    if(iter_count%solver->m_aar == 0 )
      VecNorm(solver->prec_res_new,NORM_2,&res_norm);

    if(anderson_mixing_computed)
      update_count--;

    iter_count ++;
    update_count ++;

  }
 
  t_aar_final = MPI_Wtime();

  VecCopy(solver->x_new, solver->x_final);

  output.residual_norm = res_norm;
  output.iterations = iter_count - 1;

  return output;
}


void AAR_LSQR_Solver_Destroy(AAR_LSQR_SOLVER* solver)
{
  delete [] solver->df_col;
  delete [] solver->row_inds;
  delete [] solver->row_inds_local;
  delete [] solver->col_inds;

  VecDestroy(&solver->dp); 
  VecDestroy(&solver->dp1);
  VecDestroy(&solver->dp2);

  delete [] solver->gamma_array;

  for( PetscInt counter=0 ; counter<solver->p_aar; ++counter )
  {
    VecDestroy(&solver->DX[counter]);
    VecDestroy(&solver->DF[counter]);
  }


  VecDestroy(&solver->x_old);
  VecDestroy(&solver->x_new);
  VecDestroy(&solver->res_old);
  VecDestroy(&solver->res_new);
  VecDestroy(&solver->res_old_rescaled);
  VecDestroy(&solver->res_new_rescaled);
  VecDestroy(&solver->prec_res_old);
  VecDestroy(&solver->prec_res_new);
  VecDestroy(&solver->Ax);
  VecDestroy(&solver->gamma);

  MatDestroy(&solver->DF_matrix);
  KSPDestroy(&solver->LS_solver);
}


void AAR_NE_Solver_Destroy(AAR_NE_SOLVER* solver)
{
  delete [] solver->df_col;
  delete [] solver->df_col2;
  delete [] solver->dres;

  VecDestroy(&solver->dp); 
  VecDestroy(&solver->dp1);
  VecDestroy(&solver->dp2);

  delete [] solver->FtF_temp;
  delete [] solver->FtF;
  delete [] solver->Ftf;
  delete [] solver->svec;

  for( PetscInt counter=0 ; counter<solver->p_aar; ++counter )
  {
    VecDestroy(&solver->DX[counter]);
    VecDestroy(&solver->DF[counter]);
  }


  VecDestroy(&solver->x_old);
  VecDestroy(&solver->x_new);
  VecDestroy(&solver->res_old);
  VecDestroy(&solver->res_new);
  VecDestroy(&solver->res_old_rescaled);
  VecDestroy(&solver->res_new_rescaled);
  VecDestroy(&solver->prec_res_old);
  VecDestroy(&solver->prec_res_new);
  VecDestroy(&solver->Ax);
}
