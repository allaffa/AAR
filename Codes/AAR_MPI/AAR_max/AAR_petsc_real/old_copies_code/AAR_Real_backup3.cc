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
  char prec_type[20];

}AAR_PARAMS;


typedef struct
{

  

}AAR_SOLVER;

typedef struct
{

  PetscInt iterations;
  PetscScalar residual_norm;

}AAR_OUTPUT;

void ReadInputFile(AAR_PARAMS*, char*);

void BuildPreconditioner(PC*, Mat*, char*);

AAR_OUTPUT AAR (Mat*, Vec*, Vec*, PC*, PetscInt, PetscInt, PetscReal, PetscInt, PetscReal, Vec*);

static PetscErrorCode PCSetFromOptions_HYPRE_Pilut();
static PetscErrorCode PCSetFromOptions_HYPRE_Parasails();

int main( int argc, char **argv )
{
  int ierr; 
  PetscInt number_runs = 5;
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
  ierr = PetscOptionsSetValue("-pc_hypre_pilut_tol","0.1"); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-pc_hypre_parasails_nlevels","1"); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-pc_hypre_parasails_thresh","0.01"); CHKERRQ(ierr);
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

  PetscRandom rnd;
  unsigned int long seed;
  PetscRandomCreate(PETSC_COMM_WORLD, &rnd);
  PetscRandomSetFromOptions(rnd);
  seed = world_rank;
  PetscRandomSetSeed(rnd,seed);
  PetscRandomSeed(rnd);

  for (PetscInt n_run = 0; n_run < number_runs; ++n_run)
  {
    //PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n CONSTRUCTION OF THE PRECONDITIONED RHS \n --------------------------------------------");
    VecSetRandom(x,rnd);
    VecSetRandom(x_guess,rnd);
    MatMult(A,x,b);
    PetscReal b_norm;
    VecNorm(b,NORM_2,&b_norm);
    PCApply(prec,b,prec_b);
    PetscReal prec_b_norm;
    VecNorm(prec_b,NORM_2,&prec_b_norm);

    t_gmres_init=MPI_Wtime();
    KSPSolve(restarted_gmres,b,x_gmres);
    t_gmres_final=MPI_Wtime();
    KSPGetIterationNumber(restarted_gmres,&restarted_gmres_iterations);
    KSPGetResidualNorm(restarted_gmres,&rel_res_gmres);
    rel_res_gmres = rel_res_gmres / prec_b_norm;

    t_gmres_TOT += (t_gmres_final - t_gmres_init);
    gmres_iter_TOT += restarted_gmres_iterations;

    //PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n AAR \n --------------------------------------------");
    t_aar_init=MPI_Wtime();
    aar_output = AAR (&A, &b, &x_guess, &prec, aar_parameters.m_aar, aar_parameters.p_aar, aar_parameters.omega, aar_parameters.maxit, aar_parameters.solver_tol, &x_final);
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
    PCBJacobiSetTotalBlocks(*prec, (PetscInt)(M/1024) + 1, NULL);
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
    //PCGAMGSetRepartition(*prec, flag);
    //PCGAMGSetUseParallelCoarseGridSolve(*prec, flag);
    PCSetOperators(*prec,*A,*A);
    PCSetFromOptions(*prec);
    PCSetUp(*prec);
  }
  else if(strcmp(prec_type, "asm")==0)
  {
    PetscPrintf(PETSC_COMM_WORLD, " ------------------------------------------ \n ASM preconditioner \n --------------------------------------------");
    PCSetType(*prec, PCASM);
    PCASMSetTotalSubdomains(*prec, (PetscInt)(M/1024) + 1, NULL, NULL);
    PCASMSetType(*prec, PC_ASM_RESTRICT);
    PCASMSetOverlap(*prec,1);
    PCSetFromOptions(*prec);
    PCSetOperators(*prec,*A,*A);
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

AAR_OUTPUT AAR (Mat* A, Vec* b, Vec* x_guess, PC* prec, PetscInt m_aar, PetscInt p_aar, PetscReal omega, PetscInt num_iterations, PetscReal solver_tol, Vec* x_final)
{

  PetscReal t_ls_init, t_ls_final, t_ls_tot=0.0;
  PetscReal t_rich_init, t_rich_final, t_rich_tot=0.0;
  PetscReal t_res_init, t_res_final, t_res_tot=0.0;

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  AAR_OUTPUT output;

  PetscInt M, N, lprank;
  MatGetSize(*A,&M,&N);

  PetscInt local_nrows, low, high;
  VecGetLocalSize(*b,&local_nrows);
  VecGetOwnershipRange(*b, &low, &high);

  const PetscScalar *df_col;
  const PetscScalar *df_col2;
  const PetscScalar *dres;

  Vec prec_b;
  VecCreate(PETSC_COMM_WORLD, &prec_b);
  VecSetSizes(prec_b,PETSC_DECIDE,M);
  VecSetFromOptions(prec_b);
  PCApply(*prec,*b,prec_b);
  PetscReal prec_b_norm;
  VecNorm(prec_b,NORM_2,&prec_b_norm);

  PetscScalar *FtF_temp, *FtF, *Ftf, *svec;

  FtF = new PetscScalar [p_aar*p_aar]();
  Ftf = new PetscScalar [p_aar]();
  FtF_temp = new PetscScalar [p_aar*p_aar + p_aar]();
  svec = new PetscScalar [p_aar]();

  Vec x_old, x_new, res_old, res_new, Ax, res_old_rescaled, res_new_rescaled, prec_res_old, prec_res_new;
  VecCreate(PETSC_COMM_WORLD, &x_old);
  VecCreate(PETSC_COMM_WORLD, &x_new);
  VecCreate(PETSC_COMM_WORLD, &res_old);
  VecCreate(PETSC_COMM_WORLD, &res_new);
  VecCreate(PETSC_COMM_WORLD, &prec_res_old);
  VecCreate(PETSC_COMM_WORLD, &prec_res_new);
  VecCreate(PETSC_COMM_WORLD, &Ax);
  VecCreate(PETSC_COMM_WORLD, &res_old_rescaled);
  VecCreate(PETSC_COMM_WORLD, &res_new_rescaled);
  VecSetSizes(x_old,PETSC_DECIDE,N);
  VecSetSizes(x_new,PETSC_DECIDE,N);
  VecSetSizes(res_old,PETSC_DECIDE,M);
  VecSetSizes(res_new,PETSC_DECIDE,M);
  VecSetSizes(prec_res_old,PETSC_DECIDE,M);
  VecSetSizes(prec_res_new,PETSC_DECIDE,M);
  VecSetSizes(Ax,PETSC_DECIDE,M);
  VecSetSizes(res_old_rescaled,PETSC_DECIDE,M);
  VecSetSizes(res_new_rescaled,PETSC_DECIDE,M);
  VecSetFromOptions(x_old);
  VecSetFromOptions(x_new);
  VecSetFromOptions(res_old);
  VecSetFromOptions(res_new);
  VecSetFromOptions(prec_res_old);
  VecSetFromOptions(prec_res_new);
  VecSetFromOptions(Ax);
  VecSetFromOptions(res_old_rescaled);
  VecSetFromOptions(res_new_rescaled);
  VecSet(x_old, 0.0);

  PetscInt res_vec_low, res_vec_high;
  VecGetOwnershipRange(res_new_rescaled,&res_vec_low,&res_vec_high);

  Vec dp;
  VecCreate(PETSC_COMM_WORLD, &dp);
  VecSetSizes(dp, PETSC_DECIDE, M);
  VecSetFromOptions(dp);
  VecSet(dp,0.0);
  Vec dp1;
  VecCreate(PETSC_COMM_WORLD, &dp1);
  VecSetSizes(dp1, PETSC_DECIDE, M);
  VecSetFromOptions(dp1);
  VecSet(dp1,0.0);
  Vec dp2;
  VecCreate(PETSC_COMM_WORLD, &dp2);
  VecSetSizes(dp2, PETSC_DECIDE, M);
  VecSetFromOptions(dp2);
  VecSet(dp2,0.0);

  // Now we proceed with the Richardson sweeps
  PetscInt iter_count, update_count;
  PetscBool anderson_mixing_computed = PETSC_FALSE;

  PetscReal res_norm = 1;

  Vec DX[p_aar], DF[p_aar];
  for(PetscInt vec_count=0; vec_count<p_aar; ++vec_count)
  {
    VecCreate(PETSC_COMM_WORLD, &DX[vec_count]);
    VecCreate(PETSC_COMM_WORLD, &DF[vec_count]);
    VecSetSizes(DX[vec_count],PETSC_DECIDE,N);
    VecSetSizes(DF[vec_count],PETSC_DECIDE,M);
    VecSetFromOptions(DX[vec_count]);
    VecSetFromOptions(DF[vec_count]);
  }

  iter_count = 1;
  update_count = 1;

  VecCopy(*x_guess, x_old);

  MatMult(*A,x_old,Ax);
  VecWAXPY(res_old,-1.0,Ax,*b);
  PCApply(*prec,res_old,prec_res_old);
  VecWAXPY(x_new,omega,prec_res_old,x_old);
  MatMult(*A,x_new,Ax);
  VecWAXPY(res_new,-1.0,Ax,*b);
  PCApply(*prec,res_new,prec_res_new);
  VecNorm(prec_res_new,NORM_2,&res_norm);

  PetscInt col_index_update;

  PetscReal t_aar_init, t_aar_final;

  MPI_Barrier(PETSC_COMM_WORLD);

  t_aar_init = MPI_Wtime();

  while( res_norm/prec_b_norm > solver_tol && iter_count <= num_iterations )
  {
    
    //PetscPrintf(PETSC_COMM_WORLD, "AAR - rel_res_norm: %0.13f\n", res_norm/prec_b_norm);

    if((iter_count)%m_aar!=0 || anderson_mixing_computed)
    {
      t_rich_init = MPI_Wtime();      

      //Computation of the residual
      VecSet(res_old_rescaled, 0.0);
      VecSet(res_new_rescaled, 0.0);
      VecAXPY(res_old_rescaled,omega,prec_res_old);
      VecAXPY(res_new_rescaled,omega,prec_res_new);
      VecWAXPY(DX[(update_count-1)%p_aar],-1.0,x_old,x_new);
      VecWAXPY(DF[(update_count-1)%p_aar],-1.0,res_old_rescaled,res_new_rescaled);

      VecGetArrayRead(DF[(update_count-1)%p_aar],&df_col);   
      col_index_update = (update_count-1)%p_aar;

      VecCopy(x_new, x_old);
      VecCopy(res_new, res_old);
      VecCopy(prec_res_new, prec_res_old);
      VecWAXPY(x_new,omega,prec_res_old,x_old);

      if(anderson_mixing_computed)
         anderson_mixing_computed = PETSC_FALSE;

      t_rich_final = MPI_Wtime();
      t_rich_tot += t_rich_final - t_rich_init;

    }
    else if(iter_count % m_aar==0 && iter_count<p_aar)
    {

      t_ls_init = MPI_Wtime();
      PetscScalar *FtF_temp1, *FtF1, *Ftf1, *svec1;

      VecGetArrayRead(res_new_rescaled,&dres);   

      FtF1 = new PetscScalar [(iter_count-1)*(iter_count-1)]();
      Ftf1 = new PetscScalar [iter_count-1]();
      FtF_temp1 = new PetscScalar [(iter_count-1)*(iter_count-1) + iter_count-1]();
      svec1 = new PetscScalar [iter_count-1]();

      for( PetscInt col_index=0; col_index< iter_count-1; ++col_index )
      {
        PetscInt vec_low, vec_high;
        VecGetOwnershipRange(DF[col_index],&vec_low,&vec_high);
        VecGetArrayRead(DF[col_index],&df_col);   
        for( PetscInt col2_index=0; col2_index< iter_count-1; ++col2_index )
        { 
          VecGetArrayRead(DF[col2_index],&df_col2);   
          
          PetscScalar matmat_value = 0.0;

          for( PetscInt k = 0; k<vec_high-vec_low; ++k )
            matmat_value += df_col[k]*df_col2[k];

          VecRestoreArrayRead(DF[col2_index],&df_col2);   

          FtF_temp1[ col_index + col2_index*(iter_count-1) ] = matmat_value;
        }

        PetscScalar matvec_value = 0.0;
        for( PetscInt k = 0; k<vec_high-vec_low; ++k )
          matvec_value += df_col[k]*dres[k];

        FtF_temp1[(iter_count-1)*(iter_count-1)+col_index]=matvec_value;

        VecRestoreArrayRead(DF[col_index],&df_col);   
      }
      VecRestoreArrayRead(res_new_rescaled,&dres);   

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

      LAPACKE_dgelsd(LAPACK_COL_MAJOR,iter_count-1,iter_count-1,1,FtF1,iter_count-1,Ftf1,iter_count-1,svec1,-1.0,&lprank);

      VecMAXPY(dp1,iter_count-1,Ftf1,DX);
      VecMAXPY(dp2,iter_count-1,Ftf1,DF);
      VecWAXPY(dp,1.0,dp1,dp2);
      VecAXPY(x_new,-1.0,dp);

      delete [] FtF_temp1;
      delete [] FtF1;
      delete [] Ftf1;
      delete [] svec1;

      if(iter_count == m_aar)
        anderson_mixing_computed = PETSC_TRUE;

      t_ls_final = MPI_Wtime();
      t_ls_tot += (t_ls_final-t_ls_init);
 
    }
    else if(iter_count%m_aar==0 && iter_count>=p_aar)
    {

      t_ls_init = MPI_Wtime();

      VecGetArrayRead(res_new_rescaled,&dres);   

      for( PetscInt col_index=0; col_index<p_aar; ++col_index )
      {
        PetscInt vec_low, vec_high;
        VecGetOwnershipRange(DF[col_index],&vec_low,&vec_high);
        VecGetArrayRead(DF[col_index],&df_col);   
        for( PetscInt col2_index=0; col2_index< p_aar; ++col2_index )
        { 
          VecGetArrayRead(DF[col2_index],&df_col2);   
          
          PetscScalar matmat_value = 0.0;

          for( PetscInt k = 0; k<vec_high-vec_low; ++k )
            matmat_value += df_col[k]*df_col2[k];

          VecRestoreArrayRead(DF[col2_index],&df_col2);   

          FtF_temp[ col_index + col2_index*(p_aar) ] = matmat_value;
        }

        PetscScalar matvec_value = 0.0;
        for( PetscInt k = 0; k<vec_high-vec_low; ++k )
          matvec_value += df_col[k]*dres[k];

        FtF_temp[p_aar*p_aar+col_index]=matvec_value;

        VecRestoreArrayRead(DF[col_index],&df_col);   
      }
      VecRestoreArrayRead(res_new_rescaled,&dres);   

      MPI_Allreduce(MPI_IN_PLACE, FtF_temp, p_aar*(p_aar+1), MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

      PetscInt ctr = 0;
      PetscInt ctr_temp = 0;

      for(PetscInt j=0; j<p_aar; ++j)
      {
        for(PetscInt i=0; i<p_aar; ++i)
        {
          FtF[ctr]=FtF_temp[ctr_temp];
          ctr+=1;
          ctr_temp+=1;
        }
      }
      ctr=0;
      for(PetscInt j=0; j<p_aar; ++j)
      {
        Ftf[ctr]=FtF_temp[ctr_temp];
        ctr+=1;
        ctr_temp+=1;
      }

      LAPACKE_dgelsd(LAPACK_COL_MAJOR,p_aar,p_aar,1,FtF,p_aar,Ftf,p_aar,svec,-1.0,&lprank);

      VecSet(dp1, 0.0);
      VecSet(dp2, 0.0);
      VecMAXPY(dp1,p_aar,Ftf,DX);
      VecMAXPY(dp2,p_aar,Ftf,DF);
      VecWAXPY(dp,1.0,dp1,dp2);
      VecAXPY(x_new,-1.0,dp);

      t_ls_final = MPI_Wtime();

      t_ls_tot += (t_ls_final-t_ls_init);

    }

    t_res_init = MPI_Wtime();
    //Computation of the residual
    MatMult(*A,x_new,Ax);
    VecWAXPY(res_new,-1.0,Ax,*b);
    PCApply(*prec,res_new,prec_res_new);
    t_res_final = MPI_Wtime();
    t_res_tot += t_res_final - t_res_init;

    if(iter_count%m_aar == 0 )
      VecNorm(prec_res_new,NORM_2,&res_norm);

    if(anderson_mixing_computed)
      update_count--;

    iter_count ++;
    update_count ++;

  }

  MPI_Barrier(PETSC_COMM_WORLD);
 
  t_aar_final = MPI_Wtime();

  /*if(world_rank == 0)
  {
    PetscPrintf(PETSC_COMM_WORLD, "\n");
    //PetscPrintf(PETSC_COMM_WORLD, "Total time: %f\n", t_aar_final - t_aar_init);
    PetscPrintf(PETSC_COMM_WORLD, "LS time: %f\n", t_ls_tot);
    PetscPrintf(PETSC_COMM_WORLD, "Richardson time: %f\n", t_rich_tot);
    PetscPrintf(PETSC_COMM_WORLD, "Residual time: %f\n", t_res_tot);
  }*/

  delete [] df_col;
  delete [] df_col2;
  delete [] dres;

  VecDestroy(&dp); 
  VecDestroy(&dp1);
  VecDestroy(&dp2);

  delete [] FtF_temp;
  delete [] FtF;
  delete [] Ftf;
  delete [] svec;

  for( PetscInt counter=0 ; counter<p_aar; ++counter )
  {
    VecDestroy(&DX[counter]);
    VecDestroy(&DF[counter]);
  }

  VecCopy(x_new, *x_final);

  VecDestroy(&x_old);
  VecDestroy(&x_new);
  VecDestroy(&res_old);
  VecDestroy(&res_new);
  VecDestroy(&res_old_rescaled);
  VecDestroy(&res_new_rescaled);
  VecDestroy(&prec_res_old);
  VecDestroy(&prec_res_new);
  VecDestroy(&Ax);

  output.residual_norm = res_norm;
  output.iterations = iter_count - 1;

  return output;
}
