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

  PetscInt iterations;
  PetscScalar residual_norm;

}AAR_OUTPUT;

void ReadInputFile(AAR_PARAMS*, char*);

AAR_OUTPUT AAR (Mat*, Vec*, Vec*, PC*, PetscInt, PetscInt, PetscReal, PetscInt, PetscReal, Vec*);

int main( int argc, char **argv )
{
  int ierr; 
  PetscInt restarted_gmres_iterations;
  PetscReal t_aar_init,t_aar_final, t_gmres_init, t_gmres_final;
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
  PetscScalar    one = 1.0;
  VecSet(x,one);
  VecSet(x_guess,0.0);
  MatMult(A,x,b);
  PetscReal b_norm;
  VecNorm(b,NORM_2,&b_norm);

  //Preconditioner construction
  PC prec;
  PCCreate(PETSC_COMM_WORLD, &prec);
  if(strcmp(aar_parameters.prec_type, "asm")==0)
  {
    PCSetType(prec, PCASM);
    PCASMSetType(prec, PC_ASM_RESTRICT);
    PCASMSetOverlap(prec,1);
    PCSetOperators(prec,A,A);
  }
  else if(strcmp(aar_parameters.prec_type, "spai")==0)
  {
    PCSetType(prec, PCHYPRE);
    PCHYPRESetType(prec,"parasails");
    PCSetOperators(prec,A,A);
    PCSetFromOptions(prec);
    //PCSetFromOptions_HYPRE_Parasails();
    PCSetUp(prec);
  }
  else
  { 
    PCSetType(prec,PCNONE);
    PCSetOperators(prec,A,A);
  }

  //Restarted GMRES construction
  KSP restarted_gmres;
  KSPCreate(PETSC_COMM_WORLD, &restarted_gmres);
  KSPGMRESSetRestart(restarted_gmres,10);
  KSPSetTolerances(restarted_gmres,aar_parameters.solver_tol,PETSC_DEFAULT,PETSC_DEFAULT,aar_parameters.maxit);
  KSPSetOperators(restarted_gmres,A,A);
  KSPSetPC(restarted_gmres,prec);
  KSPSetFromOptions(restarted_gmres);

  //Computation of preconditioned right-hand side
  Vec prec_b;
  VecCreate(PETSC_COMM_WORLD, &prec_b);
  VecSetSizes(prec_b,PETSC_DECIDE,M);
  VecSetFromOptions(prec_b);
  PCApply(prec,b,prec_b);
  PetscReal prec_b_norm;
  VecNorm(prec_b,NORM_2,&prec_b_norm);

  /*PetscViewerASCIIOpen(PETSC_COMM_WORLD,"vector_output",&view_out);
  PetscViewerPushFormat(view_out, PETSC_VIEWER_ASCII_MATLAB);
  VecView(b,view_out);
  PetscViewerDestroy(&view_out);  */

  t_aar_init=MPI_Wtime();
  aar_output = AAR (&A, &b, &x_guess, &prec, aar_parameters.m_aar, aar_parameters.p_aar, aar_parameters.omega, aar_parameters.maxit, aar_parameters.solver_tol, &x_final);
  t_aar_final=MPI_Wtime();
  rel_res_aar = aar_output.residual_norm / prec_b_norm;

  t_gmres_init=MPI_Wtime();
  KSPSolve(restarted_gmres,b,x_gmres);
  t_gmres_final=MPI_Wtime();

  KSPGetIterationNumber(restarted_gmres,&restarted_gmres_iterations);
  KSPGetResidualNorm(restarted_gmres,&rel_res_gmres);
  rel_res_gmres = rel_res_gmres / prec_b_norm;

  PetscPrintf(PETSC_COMM_WORLD, "AAR - Number of iterations: %d\nFinal relative residual norm: %.9f\n Time elapsed: %f (s)\n", aar_output.iterations, rel_res_aar, t_aar_final - t_aar_init);
  PetscPrintf(PETSC_COMM_WORLD, "Restarted GMRES - Number of iterations: %d\nFinal relative residual norm: %.9f\n Time elapsed: %f (s)\n", restarted_gmres_iterations, rel_res_gmres, t_gmres_final - t_gmres_init);


  VecDestroy(&x);
  VecDestroy(&x_guess);
  VecDestroy(&x_final);
  VecDestroy(&b);
  MatDestroy(&A);

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
        strcat(matrix_file, ".dat");
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

}

AAR_OUTPUT AAR (Mat* A, Vec* b, Vec* x_guess, PC* prec, PetscInt m_aar, PetscInt p_aar, PetscReal omega, PetscInt num_iterations, PetscReal solver_tol, Vec* x_final)
{

  PetscReal t_ls_init, t_ls_final, t_ls_tot=0.0;
  PetscReal t_rich_init, t_rich_final, t_rich_tot=0.0;
  PetscReal t_res_init, t_res_final, t_res_tot=0.0;

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  AAR_OUTPUT output;

  PetscInt M, N;
  MatGetSize(*A,&M,&N);

  PetscInt local_nrows, low, high;
  VecGetLocalSize(*b,&local_nrows);
  VecGetOwnershipRange(*b, &low, &high);

  const PetscScalar *df_col;

  PetscInt *row_inds = new PetscInt [local_nrows]();
  PetscInt *row_inds_local = new PetscInt [local_nrows]();
  PetscInt *col_inds = new PetscInt [p_aar]();

  for(PetscInt index = 0; index<p_aar; ++index)
    col_inds[index]=index;
  for(PetscInt index = 0; index<local_nrows; ++index)
  {
    row_inds[index]=low+index;        
    row_inds_local[index]=index;
  }

  Vec prec_b;
  VecCreate(PETSC_COMM_WORLD, &prec_b);
  VecSetSizes(prec_b,PETSC_DECIDE,M);
  VecSetFromOptions(prec_b);
  PCApply(*prec,*b,prec_b);
  PetscReal prec_b_norm;
  VecNorm(prec_b,NORM_2,&prec_b_norm);

  Mat DF_matrix;
  MatCreate(PETSC_COMM_WORLD, &DF_matrix);
  MatSetSizes(DF_matrix,PETSC_DECIDE,p_aar,M,p_aar);
  MatSetType(DF_matrix,MATAIJ);
  //MatSetType(DF_matrix,MATDENSE);
  MatMPIDenseSetPreallocation(DF_matrix,PETSC_NULL);
  MatSetUp(DF_matrix);
  PetscInt row_low, row_high, col_low, col_high;
  MatGetOwnershipRange(DF_matrix,&row_low,&row_high);
  MatGetOwnershipRangeColumn(DF_matrix,&col_low,&col_high);
  ISLocalToGlobalMapping row_map;
  ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,row_high-row_low,row_inds,PETSC_COPY_VALUES,&row_map);
  ISLocalToGlobalMapping col_map;
  ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,col_high-col_low,col_inds,PETSC_COPY_VALUES,&col_map);
  MatSetLocalToGlobalMapping(DF_matrix,row_map,col_map);

  KSP solver;
  KSPCreate(PETSC_COMM_WORLD, &solver);
  KSPSetType(solver,KSPLSQR);
  PC prec_inner_solver;
  KSPSetTolerances(solver,1e-16,PETSC_DEFAULT,PETSC_DEFAULT,p_aar);
  KSPGetPC(solver,&prec_inner_solver);
  PCSetType(prec_inner_solver,PCNONE);

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

  Vec gamma;
  VecCreate(PETSC_COMM_WORLD,&gamma);
  VecSetSizes(gamma,p_aar,p_aar);
  VecSetFromOptions(gamma);
  PetscScalar* gamma_array = new PetscScalar[p_aar]();

  // Now we proceed with the Richardson sweeps
  PetscInt iter_count, update_count;
  PetscBool anderson_mixing_computed = PETSC_FALSE;
  /*PetscReal b_norm;
  VecNorm(*b,NORM_2,&b_norm);*/

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
  t_aar_init = MPI_Wtime();

  while( res_norm/prec_b_norm > solver_tol && iter_count <= num_iterations )
  {

    PetscPrintf(PETSC_COMM_WORLD, "\nrel res norm: %f\n", res_norm/prec_b_norm);

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
      MatSetValuesLocal(DF_matrix,local_nrows,row_inds_local,1,&col_index_update,df_col,INSERT_VALUES);
      VecRestoreArrayRead(DF[col_index_update],&df_col);   

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
      PetscInt *j = new PetscInt [iter_count-1]();

      for(PetscInt index = 0; index< iter_count-1; ++index)
        j[index]=index;

      Mat DF_matrix1;
      MatCreate(PETSC_COMM_WORLD, &DF_matrix1);
      MatSetSizes(DF_matrix1,PETSC_DECIDE,iter_count-1,M,iter_count-1);
      MatSetType(DF_matrix1,MATAIJ);
      //MatSetType(DF_matrix1,MATDENSE);
      MatMPIDenseSetPreallocation(DF_matrix1,PETSC_NULL);
      MatSetUp(DF_matrix1);
      PetscInt row_low, row_high, col_low, col_high;
      MatGetOwnershipRange(DF_matrix1,&row_low,&row_high);
      MatGetOwnershipRangeColumn(DF_matrix1,&col_low,&col_high);
      ISLocalToGlobalMapping col_map1;
      ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,col_high-col_low,j,PETSC_COPY_VALUES,&col_map1);
      MatSetLocalToGlobalMapping(DF_matrix1,row_map,col_map1);


      for( PetscInt col_index=0; col_index< iter_count-1; ++col_index )
      {
        PetscInt vec_low, vec_high;
        VecGetOwnershipRange(DF[col_index],&vec_low,&vec_high);
        VecGetArrayRead(DF[col_index],&df_col);   
        MatSetValuesLocal(DF_matrix1,local_nrows,row_inds_local,1,&col_index,df_col,INSERT_VALUES);
        VecRestoreArrayRead(DF[col_index],&df_col);   
      }

      MatAssemblyBegin(DF_matrix1,MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(DF_matrix1,MAT_FINAL_ASSEMBLY);

      Vec gamma1;
      VecCreate(PETSC_COMM_WORLD,&gamma1);
      VecSetSizes(gamma1,iter_count-1,iter_count-1);
      VecSetFromOptions(gamma1);
      VecSet(gamma1, 0.0);

      KSP solver1;
      KSPCreate(PETSC_COMM_WORLD, &solver1);
      KSPSetType(solver1,KSPLSQR);
      PC prec1;
      KSPSetTolerances(solver1,1e-16,PETSC_DEFAULT,PETSC_DEFAULT,p_aar);
      KSPGetPC(solver1,&prec1);
      PCSetType(prec1,PCNONE);
      KSPSetOperators(solver1,DF_matrix1,DF_matrix1);
      KSPSetFromOptions(solver1);
      KSPSolve(solver1,res_new_rescaled,gamma1);

      PetscScalar* gamma_array1 = new PetscScalar[iter_count-1]();
      if(world_rank == 0 )
          VecGetValues(gamma,p_aar,col_inds,gamma_array);

      MPI_Bcast( gamma_array, p_aar, MPI_DOUBLE, 0, PETSC_COMM_WORLD); 

      VecMAXPY(dp1,iter_count-1,gamma_array1,DX);
      VecMAXPY(dp2,iter_count-1,gamma_array1,DF);
      VecWAXPY(dp,1.0,dp1,dp2);
      VecAXPY(x_new,-1.0,dp);

      delete [] j;
      delete [] gamma_array1;
      VecDestroy(&gamma1);
      MatDestroy(&DF_matrix1);

      if(iter_count == m_aar)
        anderson_mixing_computed = PETSC_TRUE;

      t_ls_final = MPI_Wtime();
      t_ls_tot += (t_ls_final-t_ls_init);
 
    }
    else if(iter_count%m_aar==0 && iter_count>=p_aar)
    {

      t_ls_init = MPI_Wtime();
      /*for( PetscInt col_index=0; col_index<p_aar; ++col_index )
      {
        VecGetArrayRead(DF[col_index],&df_col);   
        MatSetValuesLocal(DF_matrix,local_nrows,row_inds_local,1,&col_index,df_col,INSERT_VALUES);
        VecRestoreArrayRead(DF[col_index],&df_col);   
      }*/

      MatAssemblyBegin(DF_matrix,MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(DF_matrix,MAT_FINAL_ASSEMBLY);

      KSPSetOperators(solver,DF_matrix,DF_matrix);
      KSPSolve(solver,res_new_rescaled,gamma);

      if(world_rank == 0 )
          VecGetValues(gamma,p_aar,col_inds,gamma_array);

      MPI_Bcast( gamma_array, p_aar, MPI_DOUBLE, 0, PETSC_COMM_WORLD); 

      VecSet(dp1, 0.0);
      VecSet(dp2, 0.0);

      VecMAXPY(dp1,p_aar,gamma_array,DX);
      VecMAXPY(dp2,p_aar,gamma_array,DF);
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

  t_aar_final = MPI_Wtime();

  PetscPrintf(PETSC_COMM_WORLD, "Total time: %f\n", t_aar_final - t_aar_init);
  PetscPrintf(PETSC_COMM_WORLD, "LS time: %f\n", t_ls_tot);
  PetscPrintf(PETSC_COMM_WORLD, "Richardson time: %f\n", t_rich_tot);
  PetscPrintf(PETSC_COMM_WORLD, "Residual time: %f\n", t_res_tot);

  delete [] row_inds;
  delete [] col_inds;
  delete [] row_inds_local;
  delete [] df_col;
  MatDestroy(&DF_matrix);

  VecDestroy(&dp); 
  VecDestroy(&dp1);
  VecDestroy(&dp2);
  VecDestroy(&gamma);
  delete [] gamma_array;

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
