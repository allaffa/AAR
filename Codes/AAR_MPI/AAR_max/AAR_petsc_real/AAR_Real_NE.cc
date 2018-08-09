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
  Vec x_guess, x_old, x_new, x_final, res_old, res_new, Ax, res_old_rescaled, res_new_rescaled, prec_res_old, prec_res_new;

  const PetscScalar *df_col;
  const PetscScalar *df_col2;
  const PetscScalar *dres;
  PetscScalar *FtF_temp, *FtF, *Ftf, *svec;

  Vec *DX, *DF;
  Vec dp, dp1, dp2;

}AAR_SOLVER;


typedef struct
{

  PetscInt iterations;
  PetscScalar residual_norm;

}AAR_OUTPUT;

void ReadInputFile(AAR_PARAMS*, char*);

void BuildPreconditioner(PC*, Mat*, char*);

void AARSetUp(AAR_SOLVER*, Mat*, Vec*, PC*, PetscInt, PetscInt, PetscReal, PetscInt, PetscReal);

void AAR_Solver_Destroy(AAR_SOLVER*);

AAR_OUTPUT AAR_Solve(AAR_SOLVER*, Vec*, PC*);

static PetscErrorCode PCSetFromOptions_HYPRE_Pilut();
static PetscErrorCode PCSetFromOptions_HYPRE_Parasails();

int main( int argc, char **argv )
{
  int ierr; 
  PetscInt number_runs = 1;
  PetscInt restarted_gmres_iterations;
  PetscReal t_aar_init = 0.0, t_aar_final = 0.0, t_gmres_init = 0.0, t_gmres_final = 0.0, t_aar_TOT = 0.0, t_gmres_TOT = 0.0;
  PetscInt gmres_iter_TOT = 0.0, aar_iter_TOT = 0.0;
  PetscReal norm_infA;
  PetscReal rel_res_gmres;
  PetscReal rel_res_aar;
  AAR_PARAMS aar_parameters;
  AAR_OUTPUT aar_output;
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
    aar_parameters.omega = 0.0001;

  PetscInt M, N;
  MatGetSize(A,&M,&N);
  /*PetscViewerASCIIOpen(PETSC_COMM_WORLD,"matrix_output",&view_matrix_out);
  PetscViewerPushFormat(view_matrix_out, PETSC_VIEWER_ASCII_MATLAB);
  //MatView(A,view_matrix_out); THIS IS VERY SLOW!!!
  PetscViewerDestroy(&view_matrix_out);*/

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

  PetscScalar t_prec_construct_init = 0.0, t_prec_construction_final = 0.0;

  //Preconditioner construction
  PC prec;
  BuildPreconditioner(&prec, &A, aar_parameters.prec_type);

  //Computation of preconditioned right-hand side
  Vec prec_b;
  VecCreate(PETSC_COMM_WORLD, &prec_b);
  VecSetSizes(prec_b,PETSC_DECIDE,M);
  VecSetFromOptions(prec_b);

  //PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n RESTARTED GMRES \n --------------------------------------------");
  //Restarted GMRES construction
  KSP restarted_gmres;
  KSPCreate(PETSC_COMM_WORLD, &restarted_gmres);
  KSPGMRESSetRestart(restarted_gmres,aar_parameters.p_aar);
  KSPSetTolerances(restarted_gmres,aar_parameters.solver_tol,PETSC_DEFAULT,PETSC_DEFAULT,aar_parameters.maxit);
  KSPSetOperators(restarted_gmres,A,A);
  KSPSetPC(restarted_gmres,prec);
  KSPSetFromOptions(restarted_gmres);

  AAR_SOLVER aar_solver;
  AARSetUp(&aar_solver, &A, &x_guess, &prec, aar_parameters.m_aar, aar_parameters.p_aar, aar_parameters.omega, aar_parameters.maxit, aar_parameters.solver_tol);

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
    VecSetRandom(x_guess,rnd);
    MatMult(A,x,b);
    PetscReal b_norm;
    VecNorm(b,NORM_2,&b_norm);
    PCApply(prec,b,prec_b);
    PetscReal prec_b_norm;
    VecNorm(prec_b,NORM_2,&prec_b_norm);

    PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n RESTARTED GMRES \n --------------------------------------------");
    t_gmres_init=MPI_Wtime();
    if(aar_parameters.gmres_compute == 1)
      KSPSolve(restarted_gmres,b,x_gmres);
    t_gmres_final=MPI_Wtime();
    KSPGetIterationNumber(restarted_gmres,&restarted_gmres_iterations);
    KSPGetResidualNorm(restarted_gmres,&rel_res_gmres);
    rel_res_gmres = rel_res_gmres / prec_b_norm;

    t_gmres_TOT += (t_gmres_final - t_gmres_init);
    gmres_iter_TOT += restarted_gmres_iterations;

    PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n AAR \n --------------------------------------------");
    t_aar_init=MPI_Wtime();
    aar_output = AAR_Solve (&aar_solver, &b, &prec);
    t_aar_final=MPI_Wtime();
    rel_res_aar = aar_output.residual_norm / prec_b_norm;

    t_aar_TOT += (t_aar_final - t_aar_init);
    aar_iter_TOT += aar_output.iterations;

  }

  PetscPrintf(PETSC_COMM_WORLD, "AAR - Number of iterations: %d\nFinal relative residual norm: %.9f\n Time elapsed: %f (s)\n", aar_iter_TOT/number_runs, rel_res_aar, t_aar_TOT/number_runs);
  PetscPrintf(PETSC_COMM_WORLD, "Restarted GMRES - Number of iterations: %d\nFinal relative residual norm: %.9f\n Time elapsed: %f (s)\n", gmres_iter_TOT/number_runs, rel_res_gmres, t_gmres_TOT/number_runs);

  VecDestroy(&x);
  VecDestroy(&x_guess);
  VecDestroy(&x_final);
  VecDestroy(&b);
  VecDestroy(&prec_b);
  MatDestroy(&A);
  PCDestroy(&prec);
  KSPDestroy(&restarted_gmres);
  AAR_Solver_Destroy(&aar_solver);
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

void AARSetUp(AAR_SOLVER* solver, Mat* A, Vec* x_guess, PC* prec, PetscInt m_aar, PetscInt p_aar, PetscReal omega, PetscInt num_iterations, PetscReal solver_tol)
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

  VecCopy(solver->x_guess, *x_guess);
}


AAR_OUTPUT AAR_Solve (AAR_SOLVER* solver, Vec* b, PC* prec)
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

  /*if(world_rank == 0)
  {
    PetscPrintf(PETSC_COMM_WORLD, "\n");
    //PetscPrintf(PETSC_COMM_WORLD, "Total time: %f\n", t_aar_final - t_aar_init);
    PetscPrintf(PETSC_COMM_WORLD, "LS time: %f\n", t_ls_tot);
    PetscPrintf(PETSC_COMM_WORLD, "Richardson time: %f\n", t_rich_tot);
    PetscPrintf(PETSC_COMM_WORLD, "Residual time: %f\n", t_res_tot);
  }*/

  VecCopy(solver->x_new, solver->x_final);

  output.residual_norm = res_norm;
  output.iterations = iter_count - 1;

  return output;
}


void AAR_Solver_Destroy(AAR_SOLVER* solver)
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
