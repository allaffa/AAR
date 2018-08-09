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

int main( int argc, char **argv )
{
  int ierr; 
  PetscReal t0,t1;
  PetscReal norm_infA;
  PetscInt num_iterations = 50;
  PetscInt m_aar = 6; 
  PetscInt p_aar = 10; 
  PetscScalar omega;

  PetscInitialize(&argc,&argv,(char*)0,help);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  PetscViewer    view_out,view_in, view_matrix_out;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"chipcool0.dat",FILE_MODE_READ,&view_in);
  Mat A;
  MatCreate(PETSC_COMM_WORLD,&A);
  MatSetType(A,MATAIJ);
  MatLoad(A,view_in);
  PetscViewerDestroy(&view_in);
  MatNorm(A,NORM_INFINITY,&norm_infA);

  omega = 2/norm_infA;
 
  PetscInt M, N;
  MatGetSize(A,&M,&N);
  /*PetscViewerASCIIOpen(PETSC_COMM_WORLD,"matrix_output",&view_matrix_out);
  PetscViewerPushFormat(view_matrix_out, PETSC_VIEWER_ASCII_MATLAB);
  //MatView(A,view_matrix_out); THIS IS VERY SLOW!!!
  PetscViewerDestroy(&view_matrix_out);*/

  Vec x,b;
  VecCreate(PETSC_COMM_WORLD, &x);
  VecCreate(PETSC_COMM_WORLD, &b);
  VecSetSizes(x,PETSC_DECIDE,N);
  VecSetSizes(b,PETSC_DECIDE,M);
  VecSetFromOptions(x);
  VecSetFromOptions(b);
  PetscScalar    one = 1.0;
  VecSet(x,one);
  MatMult(A,x,b);

  PetscScalar b_norm;
  VecNorm(b,NORM_2,&b_norm);

  /*PetscViewerASCIIOpen(PETSC_COMM_WORLD,"vector_output",&view_out);
  PetscViewerPushFormat(view_out, PETSC_VIEWER_ASCII_MATLAB);
  VecView(b,view_out);
  PetscViewerDestroy(&view_out);  */

  t0=MPI_Wtime();

  PetscInt local_nrows, low, high;
  VecGetLocalSize(b,&local_nrows);
  VecGetOwnershipRange(b, &low, &high);

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

  Mat DF_matrix;
  MatCreate(PETSC_COMM_WORLD, &DF_matrix);
  MatSetSizes(DF_matrix,PETSC_DECIDE,p_aar,M,p_aar);
  MatSetType(DF_matrix,MATAIJ);
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
  PC prec;
  KSPSetTolerances(solver,1e-10,PETSC_DEFAULT,PETSC_DEFAULT,p_aar);
  KSPGetPC(solver,&prec);
  PCSetType(prec,PCNONE);

  Vec x_old, x_new, res_old, res_new, Ax, res_old_rescaled, res_new_rescaled;
  VecCreate(PETSC_COMM_WORLD, &x_old);
  VecCreate(PETSC_COMM_WORLD, &x_new);
  VecCreate(PETSC_COMM_WORLD, &res_old);
  VecCreate(PETSC_COMM_WORLD, &res_new);
  VecCreate(PETSC_COMM_WORLD, &Ax);
  VecCreate(PETSC_COMM_WORLD, &res_old_rescaled);
  VecCreate(PETSC_COMM_WORLD, &res_new_rescaled);
  VecSetSizes(x_old,PETSC_DECIDE,N);
  VecSetSizes(x_new,PETSC_DECIDE,N);
  VecSetSizes(res_old,PETSC_DECIDE,M);
  VecSetSizes(res_new,PETSC_DECIDE,M);
  VecSetSizes(Ax,PETSC_DECIDE,M);
  VecSetSizes(res_old_rescaled,PETSC_DECIDE,M);
  VecSetSizes(res_new_rescaled,PETSC_DECIDE,M);
  VecSetFromOptions(x_old);
  VecSetFromOptions(x_new);
  VecSetFromOptions(res_old);
  VecSetFromOptions(res_new);
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
  PetscInt iter_count;
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

  MatMult(A,x_old,Ax);
  VecWAXPY(res_old,-1.0,Ax,b);
  VecWAXPY(x_new,omega,res_old,x_old);
  VecNorm(res_old,NORM_2,&res_norm);
  MatMult(A,x_new,Ax);
  VecWAXPY(res_new,-1.0,Ax,b);
  VecNorm(res_new,NORM_2,&res_norm);
  VecWAXPY(x_new,omega,res_old,x_old);
  VecNorm(res_old,NORM_2,&res_norm);
  MatMult(A,x_new,Ax);
  VecWAXPY(res_new,-1.0,Ax,b);
  VecNorm(res_new,NORM_2,&res_norm);

  while( res_norm > 1e-9 && iter_count<= num_iterations )
  {

    PetscPrintf(PETSC_COMM_WORLD, "Residual norm at iteration %d: %f\n", iter_count, res_norm/b_norm);

    if((iter_count)%m_aar!=0)
    {
      //Computation of the residual
      VecSet(res_old_rescaled, 0.0);
      VecSet(res_new_rescaled, 0.0);
      VecAXPY(res_old_rescaled,omega,res_old);
      VecAXPY(res_new_rescaled,omega,res_new);
      VecWAXPY(DX[(iter_count-1)%p_aar],-1.0,x_old,x_new);
      VecWAXPY(DF[(iter_count-1)%p_aar],-1.0,res_old_rescaled,res_new_rescaled);

      VecCopy(x_new, x_old);
      VecCopy(res_new, res_old);
      VecNorm(res_old,NORM_2,&res_norm);
      VecWAXPY(x_new,omega,res_old,x_old);
    }
    else if(iter_count % m_aar==0 && iter_count<p_aar)
    {
      PetscInt *j = new PetscInt [iter_count-1]();

      for(PetscInt index = 0; index< iter_count-1; ++index)
        j[index]=index;

      Mat DF_matrix1;
      MatCreate(PETSC_COMM_WORLD, &DF_matrix1);
      MatSetSizes(DF_matrix1,PETSC_DECIDE,iter_count-1,M,iter_count-1);
      MatSetType(DF_matrix1,MATAIJ);
      MatSetUp(DF_matrix1);
      PetscInt row_low, row_high, col_low, col_high;
      MatGetOwnershipRange(DF_matrix1,&row_low,&row_high);
      MatGetOwnershipRangeColumn(DF_matrix1,&col_low,&col_high);
      ISLocalToGlobalMapping col_map1;
      ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,col_high-col_low,j,PETSC_COPY_VALUES,&col_map1);
      MatSetLocalToGlobalMapping(DF_matrix1,row_map,col_map1);

      const PetscScalar *df_col;

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
      KSPSetTolerances(solver1,1e-10,PETSC_DEFAULT,PETSC_DEFAULT,p_aar);
      KSPGetPC(solver1,&prec1);
      PCSetType(prec1,PCNONE);
      KSPSetOperators(solver1,DF_matrix1,DF_matrix1);
      KSPSetFromOptions(solver1);
      KSPSolve(solver1,res_new_rescaled,gamma1);

      PetscScalar* gamma_array1 = new PetscScalar[iter_count-1]();
      if(world_rank != 0 )
        for( PetscInt gamma_counter=0; gamma_counter<iter_count-1; ++gamma_counter )
           gamma_array1[gamma_counter]=0.0;

      else
        for( PetscInt gamma_counter=0; gamma_counter<iter_count-1; ++gamma_counter )
        {
           PetscScalar gamma_val;
           VecGetValues(gamma1,1,&gamma_counter,&gamma_val);
           gamma_array1[gamma_counter]=gamma_val;
        }

      MPI_Allreduce(MPI_IN_PLACE, gamma_array1, (iter_count-1), MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

      VecMAXPY(dp1,iter_count-1,gamma_array1,DX);
      VecMAXPY(dp2,iter_count-1,gamma_array1,DF);
      VecWAXPY(dp,1.0,dp1,dp2);
      PetscScalar dp_norm;
      VecNorm(dp,NORM_2,&dp_norm);
      VecAXPY(x_new,-1.0,dp);

      delete [] df_col;
      delete [] j;
      delete [] gamma_array1;
      VecDestroy(&gamma1);
      MatDestroy(&DF_matrix1);
 
    }
    else if(iter_count%m_aar==0 && iter_count>=p_aar)
    {

      const PetscScalar *df_col;

      for( PetscInt col_index=0; col_index<p_aar; ++col_index )
      {
        VecGetArrayRead(DF[col_index],&df_col);   
        MatSetValuesLocal(DF_matrix,local_nrows,row_inds_local,1,&col_index,df_col,INSERT_VALUES);
        VecRestoreArrayRead(DF[col_index],&df_col);   
      }

      MatAssemblyBegin(DF_matrix,MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(DF_matrix,MAT_FINAL_ASSEMBLY);

      KSPSetOperators(solver,DF_matrix,DF_matrix);
      KSPSetFromOptions(solver);
      KSPSolve(solver,res_new_rescaled,gamma);

      if(world_rank != 0 )
        for( PetscInt gamma_counter=0; gamma_counter<p_aar; ++gamma_counter )
            gamma_array[gamma_counter]=0.0;

      else
        for( PetscInt gamma_counter=0; gamma_counter<p_aar; ++gamma_counter )
        {
          PetscScalar gamma_val;
          VecGetValues(gamma,1,&gamma_counter,&gamma_val);
          gamma_array[gamma_counter]=gamma_val;
        }

      MPI_Allreduce(MPI_IN_PLACE, gamma_array, p_aar, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

      VecSet(dp1, 0.0);
      VecSet(dp2, 0.0);

      VecMAXPY(dp1,p_aar,gamma_array,DX);
      VecMAXPY(dp2,p_aar,gamma_array,DF);
      VecWAXPY(dp,1.0,dp1,dp2);
      PetscScalar dp_norm;
      VecNorm(dp,NORM_2,&dp_norm);
      VecAXPY(x_new,-1.0,dp);

      delete [] df_col;

    }
   
   //Computation of the residual
   MatMult(A,x_new,Ax);
   VecWAXPY(res_new,-1.0,Ax,b);
   VecNorm(res_new,NORM_2,&res_norm);

    iter_count ++;

  }

  delete [] row_inds;
  delete [] col_inds;
  delete [] row_inds_local;
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

  VecDestroy(&x);
  VecDestroy(&x_old);
  VecDestroy(&x_new);
  VecDestroy(&res_old);
  VecDestroy(&res_new);
  VecDestroy(&Ax);
  VecDestroy(&b);
  MatDestroy(&A);

  ierr = PetscFinalize();CHKERRQ(ierr);
 
  return 0;
}
 
