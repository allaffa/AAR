/*=============================================================================================
  | Alternating Anderson Richardson (AAR) code for real-valued systems
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
//#include "mkl_lapacke.h"
//#include "mkl.h"
#include "lapacke.h"
#include <iostream>

#define MAX_ORDER 10
#define M_PI 3.14159265358979323846

typedef struct
{
  PetscInt  numPoints_x;
  PetscInt  numPoints_y;
  PetscInt  numPoints_z;
  PetscInt  order; // half FD order
  PetscInt pc;     // 1=block-jacobi or 0=jacobi
  PetscReal coeffs[MAX_ORDER+1];
  
  char file[30];    
  
  PetscReal solver_tol;
  PetscInt m_aar;
  PetscInt p_aar;
  PetscReal beta_aar;
  PetscScalar Dinv_factor;
  
  DM da;
  Vec RHS;
  Vec Phi;

  Mat poissonOpr;   
  
  Vec xkprev;
  Vec xk;
  Vec fkprev;  
}AAR_OBJ;

void Setup_and_Initialize(AAR_OBJ* pAAR);
void ObjectInitialize(AAR_OBJ* pAAR);
void Read_parameters(AAR_OBJ* pAAR);
void Objects_Create(AAR_OBJ* pAAR);
void ComputeMatrixA(AAR_OBJ* pAAR);
void AAR(AAR_OBJ* pAAR); 
void Objects_Destroy(AAR_OBJ* pAAR);

int main( int argc, char **argv )
{
  int ierr; 
  AAR_OBJ aar;
  PetscReal t0,t1;
  
  PetscInitialize(&argc,&argv,(char*)0,help);

  t0=MPI_Wtime();

  Setup_and_Initialize(&aar);
  // Compute RHS and Matrix for the Poisson equation
  ComputeMatrixA(&aar);     // Matrix, A = paar->poissonOpr
  // NOTE: For a different problem, other than Poisson equation, provide the matrix through the variable "paar->poissonOpr" and right hand side through "paar->RHS". 
   
  t1=MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD,"\nTime spent in initialization = %.4f seconds.\n",t1-t0);

  PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");
  // -------------- AAR solver --------------------------
  AAR(&aar); // AAR with preconditioning PC=0={Jacobi}, PC=1={ICC(0) block-Jacobi}     
  PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");

  t1=MPI_Wtime();

  Objects_Destroy(&aar);  
  PetscPrintf(PETSC_COMM_WORLD,"Total wall time = %.4f seconds.\n\n",t1-t0);
  ierr = PetscFinalize();CHKERRQ(ierr);
 
  return 0;
}
 

//////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// AAR solver ////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void AAR(AAR_OBJ* pAAR)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  double t0,t1; 
  t0=MPI_Wtime();
  int max_iter=2000,m_aar=pAAR->m_aar,p_aar=pAAR->p_aar,iter=1,m,ctr,ctr_temp,cnt,ii,jj,kk;
  double beta_aar=pAAR->beta_aar,aar_tol=pAAR->solver_tol,relres=aar_tol+1.0;
  PetscScalar ***rhs,***phi_old,***phi_new,***phi_res,***Xold,***Fold;   
  PetscScalar *FtF_temp,*FtF;      // DF'*DF matrix in column major format, size m*m x 1
  PetscScalar *Ftf;      // DF'*phi_res, size mx1  
  PetscScalar **DX,**DF; // iterate history, size Np x m
  PetscScalar temp_sum; 
  int lprank;       // required for lapacke
  double *svec;
  double rhs_norm;  
  Vec Matvec_global,phi_old_global;
    
  PetscInt i,j,k,l,colidx,gxdim,gydim,gzdim,xcor,ycor,zcor,lxdim,lydim,lzdim;

  DMDAGetCorners(pAAR->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
  int Np = lxdim*lydim*lzdim; // no. of nodes in the current processor

  phi_new = new PetscScalar** [lzdim]();
  Xold = new PetscScalar** [lzdim]();
  Fold = new PetscScalar** [lzdim]();
  for(k=0;k<lzdim;k++)
    { 
      phi_new[k] = new PetscScalar* [lydim]();
      Xold[k] = new PetscScalar* [lydim]();
      Fold[k] = new PetscScalar* [lydim]();
      for(j=0;j<lydim;j++)
	{
	  phi_new[k][j] = new PetscScalar [lxdim](); 
	  Xold[k][j] = new PetscScalar [lxdim]();
	  Fold[k][j] = new PetscScalar [lxdim]();
	}
    }
  
  DX = new PetscScalar* [m_aar]();
  DF = new PetscScalar* [m_aar]();
  for(k=0;k<m_aar;k++)
    {
      DX[k] = new PetscScalar [Np]();
      DF[k] = new PetscScalar [Np]();
    }

  FtF = new PetscScalar [m_aar*m_aar]();
  Ftf = new PetscScalar [m_aar]();
  FtF_temp = new PetscScalar [m_aar*m_aar+m_aar+1]();
  svec = new double [m_aar]();

  VecDuplicate(pAAR->Phi,&Matvec_global); 
  VecDuplicate(pAAR->Phi,&phi_old_global);
  //VecSet(phi_old_global,0.0);  // initial guess
  VecCopy(pAAR->Phi,phi_old_global); // random initial guess
  VecNorm(pAAR->RHS, NORM_2, &rhs_norm); // rhs=pAAR->RHS

  // get rhs array
  DMDAVecGetArray(pAAR->da,pAAR->RHS,&rhs); // rhs is a global 3d array reference of RHS which is a global 1d array distributed among procs.
  
  // solving Ax=b, poissonOpr=A, rhs=b, D=diag(A)=diag(-1/4pi*lap)

  // Need to get the diagonal block of A
  Mat Dblock; // sequntial matrix contained in current processor
  MatGetDiagonalBlock(pAAR->poissonOpr,&Dblock);
  // Factorize the diagonal block of A (or) provide the solver info for solving with preconditoning matrix (in this case, (block) diagonal matrix)
  PC prec;
  PCCreate(PETSC_COMM_SELF,&prec);
  if(pAAR->pc==1)
    {
      PCSetType(prec,PCICC); //ICC(0), Block-Jacobi
      PetscPrintf(PETSC_COMM_WORLD,"AAR preconditioned with Block-Jacobi using ICC(0).\n");
    }
  else if(pAAR->pc==0)
    {
      PCSetType(prec,PCJACOBI); // AAJ
      PetscPrintf(PETSC_COMM_WORLD,"AAR preconditioned with Jacobi (AAJ).\n");
    }
    
  PCSetOperators(prec,Dblock,Dblock);

  // Create local vector to store ouput from PCApply
  Vec LocalVec1;
  VecCreateSeq(PETSC_COMM_SELF,Np,&LocalVec1);
  Vec LocalVec2;
  VecCreateSeq(PETSC_COMM_SELF,Np,&LocalVec2);

  PetscScalar *locarr1,*locarr2;

  while(relres>aar_tol && iter<=max_iter)
    {
      
      MatMult(pAAR->poissonOpr,phi_old_global,Matvec_global); // Matvec_global=A*phi_k  

      DMDAVecGetArray(pAAR->da,Matvec_global,&phi_res);
      DMDAVecGetArray(pAAR->da,phi_old_global,&phi_old);
      VecGetArray(LocalVec1,&locarr1);

      ctr=0;
      for(k=0;k<lzdim;k++)
	{
	  for(j=0;j<lydim;j++)
	    {
	      for(i=0;i<lxdim;i++)
		{
		  locarr1[ctr] = (rhs[k+zcor][j+ycor][i+xcor]-phi_res[k+zcor][j+ycor][i+xcor]); // (rhs-A*phi_k)
                  phi_res[k+zcor][j+ycor][i+xcor] = locarr1[ctr]; 
                  ctr+=1;
		}
	    }
	}

      VecRestoreArray(LocalVec1,&locarr1);
    
      PCApply(prec,LocalVec1,LocalVec2); // Vec2=inv(M)*Vec1=inv(M)*(rhs-A*phi_k), where M is preconditioner 
      VecGetArray(LocalVec2,&locarr2);       
      
      ctr=0;
      for(k=0;k<lzdim;k++)
	{
	  for(j=0;j<lydim;j++)
	    {
	      for(i=0;i<lxdim;i++)
		{
		  phi_new[k][j][i] = phi_old[k+zcor][j+ycor][i+xcor]+beta_aar*locarr2[ctr]; // phi_new = phi_old + beta*r_k, WJ update
                  ctr+=1;
		}
	    }
	}
      
      // ------- Store history -------
      if(iter>1)
	{
	  m = ((iter-2) % m_aar)+1-1;
	  ctr=0;
	  for(k=0;k<lzdim;k++)
	    {
	      for(j=0;j<lydim;j++)
		{
		  for(i=0;i<lxdim;i++)
		    {
		      DX[m][ctr] = phi_old[k+zcor][j+ycor][i+xcor]-Xold[k][j][i];  //DX[m][:] = phi_old-Xold
		      DF[m][ctr] = locarr2[ctr]-Fold[k][j][i];  //DF[m][:] = phi_res-Fold  
		      ctr+=1;
		    }
		}
	    }

	} // end if

      ctr=0;
      for(k=0;k<lzdim;k++)
	{
	  for(j=0;j<lydim;j++)
	    {
	      for(i=0;i<lxdim;i++)
		{
		  Xold[k][j][i] = phi_old[k+zcor][j+ycor][i+xcor];  
		  Fold[k][j][i] = locarr2[ctr];   
                  ctr+=1;            
		}
	    }
	}
      
      // -------- Anderson update --------
      if(iter % p_aar == 0 && iter > 1)
	{
	  for(j=0;j<m_aar;j++)
	    {	      
	      for(i=0;i<=j;i++)
		{  
		  temp_sum=0.0;
		  for(k=0;k<Np;k++)
		    {	      
		      temp_sum += DF[i][k]*DF[j][k]; // DF'*DF
		    }   
                  ctr=j*m_aar + i; // FtF(i,j) linear index is j*m_aar + i
		  FtF_temp[ctr]=temp_sum;   
                  // symmetric elements
                  ctr=i*m_aar + j; // FtF(j,i) linear index is i*m_aar + j
		  FtF_temp[ctr]=temp_sum;      
		}
	    }
          ctr_temp=m_aar*m_aar;

          ctr=0;
	  for(j=0;j<m_aar;j++)
	    {       
              temp_sum=0.0;
	      cnt=0;
	      for(kk=0;kk<lzdim;kk++)
		{
		  for(jj=0;jj<lydim;jj++)
		    {
		      for(ii=0;ii<lxdim;ii++)	
             		{ 
			  temp_sum += DF[j][cnt]*locarr2[cnt]; //DF'*phi_res
			  cnt+=1;
			}
		    }
		}
	      Ftf[ctr]=temp_sum; // (Ftf)_j element
	      FtF_temp[ctr_temp]=temp_sum;
	      ctr+=1;  
	      ctr_temp+=1;           
	    }
	  temp_sum=0.0;
	  for(k=0;k<lzdim;k++)
	    {
	      for(j=0;j<lydim;j++)
		{
		  for(i=0;i<lxdim;i++)
		    {
		      temp_sum += phi_res[k+zcor][j+ycor][i+xcor]*phi_res[k+zcor][j+ycor][i+xcor];
		    }
		}
	    }
	  FtF_temp[ctr_temp]=temp_sum;
 
          MPI_Allreduce(MPI_IN_PLACE,FtF_temp,(m_aar*m_aar+m_aar+1),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); 

	  ctr=0;
          ctr_temp=0;
	  for(j=0;j<m_aar;j++)
	    {       
	      for(i=0;i<m_aar;i++)
		{    
		  FtF[ctr]=FtF_temp[ctr_temp]; 
		  ctr+=1; 
		  ctr_temp+=1;            
		}
	    }
          ctr=0;
	  for(j=0;j<m_aar;j++)
	    {       
	      Ftf[ctr]=FtF_temp[ctr_temp];
	      ctr+=1;  
	      ctr_temp+=1;           
	    }

	  relres=FtF_temp[ctr_temp];
	  relres = sqrt(relres);
	  relres = relres/rhs_norm; // relative residual

	  //PetscPrintf(PETSC_COMM_WORLD,"Iteration=%d, Relres = %.14f \n",iter,relres);     
       
	  LAPACKE_dgelsd(LAPACK_COL_MAJOR,m_aar,m_aar,1,FtF,m_aar,Ftf,m_aar,svec,-1.0,&lprank); // Ftf gets replaced by Yk
	  
	  cnt=0;
	  for(kk=0;kk<lzdim;kk++)
	    {
	      for(jj=0;jj<lydim;jj++)
		{
		  for(ii=0;ii<lxdim;ii++)
		    {    
		      temp_sum=0.0;
		      for(j=0;j<m_aar;j++)
			{  
			  temp_sum += (DX[j][cnt] + beta_aar*DF[j][cnt])*Ftf[j];
			}           
		      phi_new[kk][jj][ii] = phi_new[kk][jj][ii]-temp_sum; // anderson update step
		      cnt+=1;
		    }
		}
	    }  
	} // end anderson update

      
      for(k=0;k<lzdim;k++)
	{
	  for(j=0;j<lydim;j++)
	    {
	      for(i=0;i<lxdim;i++)
		{
		  phi_old[k+zcor][j+ycor][i+xcor]=phi_new[k][j][i]; // phi_old = phi_new
		}
	    }
	}

      VecRestoreArray(LocalVec2,&locarr2);
      DMDAVecRestoreArray(pAAR->da,Matvec_global,&phi_res); 
      DMDAVecRestoreArray(pAAR->da,phi_old_global,&phi_old);     
      
      iter = iter+1;

    } // end while loop
  //VecView(phi_old_global,PETSC_VIEWER_STDOUT_WORLD);
  
  if(iter<max_iter)
    {
      PetscPrintf(PETSC_COMM_WORLD,"AAR converged!: Iterations=%d, Relres = %g \n",iter-1,relres);
    }else
    {
      PetscPrintf(PETSC_COMM_WORLD,"AAR exceeded max_iter!: Iterations=%d, Relres = %g \n",iter-1,relres);
    }

  t1=MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD,"Time taken by AAR = %.4f seconds.\n",t1-t0);  
    
  DMDAVecRestoreArray(pAAR->da,pAAR->RHS,&rhs);    

  VecDestroy(&phi_old_global);
  VecDestroy(&Matvec_global);

  PCDestroy(&prec);
  
  VecDestroy(&LocalVec1);
  VecDestroy(&LocalVec2);
  
  for(k=0;k<lzdim;k++)
    {
      for(j=0;j<lydim;j++)
	{
          delete [] phi_new[k][j];
          delete [] Xold[k][j];
          delete [] Fold[k][j];
        }
      delete [] phi_new[k];
      delete [] Xold[k];
      delete [] Fold[k];
    }
  delete [] phi_new;
  delete [] Xold;
  delete [] Fold;

  for(k=0;k<m_aar;k++)
    {
      delete [] DX[k];
      delete [] DF[k];
    }
  delete [] DX;
  delete [] DF;

  delete [] FtF;
  delete [] Ftf;
  delete [] FtF_temp;  
  delete [] svec;
}




///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Read Input file
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void Read_parameters(AAR_OBJ* pAAR)
{    
  FILE *fConfFile;
  PetscInt p,i;
  PetscReal Nr, Dr, val;
  char ConfigFile[100]="./";
  strcat(ConfigFile,pAAR->file);
  strcat(ConfigFile,".in");  
  if((fConfFile = fopen ((const char *)ConfigFile, "rb"))==NULL)
    {     
      PetscPrintf(PETSC_COMM_WORLD,"Couldn't open input file... Exiting...\n");
      exit(1);
    }         
  char str[60];

  pAAR->order = 3; // store half order
  pAAR->numPoints_x = 48; pAAR->numPoints_y = 48; pAAR->numPoints_z = 48;
  
  do 
    {
      fscanf(fConfFile,"%s",str);
      if(strcmp(str,"aar_pc:")==0)	
	{
	  fscanf(fConfFile,"%d", &pAAR->pc);
	}else if(strcmp(str,"m_aar:")==0)	
	{
	  fscanf(fConfFile,"%d", &pAAR->m_aar);
	}else if(strcmp(str,"p_aar:")==0)	
	{
	  fscanf(fConfFile,"%d", &pAAR->p_aar);
	}else if(strcmp(str,"beta_aar:")==0)	
	{
	  fscanf(fConfFile,"%lf", &pAAR->beta_aar);
	}
      else if(strcmp(str,"solver_tol:")==0)	
	{
	  fscanf(fConfFile,"%lf", &pAAR->solver_tol);
	}
    }while(!feof(fConfFile));
	  
  fclose(fConfFile);
  
  //coefficients of the laplacian
  pAAR->coeffs[0] = 0;
  for(p=1; p<=pAAR->order; p++)
    pAAR->coeffs[0]+= ((PetscReal)1.0/(p*p));
  pAAR->coeffs[0]*=((PetscReal)3.0);
 
  for(p=1;p<=pAAR->order;p++)
    {
      Nr=1;Dr=1;
      for(i=pAAR->order-p+1; i<=pAAR->order; i++)
	Nr*=i;
      for(i=pAAR->order+1; i<=pAAR->order+p; i++)
	Dr*=i;
      val = Nr/Dr;  
      pAAR->coeffs[p] = (PetscReal)(-1*pow(-1,p+1)*val/(p*p*(1)));
    }
  
  for(p=0;p<=pAAR->order;p++)
    {
      pAAR->coeffs[p] = pAAR->coeffs[p]/(2*M_PI); // so total (-1/4*pi) factor on fd coeffs
    }  
		
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD,"                           INPUT PARAMETERS                                \n");
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  //PetscPrintf(PETSC_COMM_WORLD,"FD_ORDER    : %d\n",2*pAAR->order);
  PetscPrintf(PETSC_COMM_WORLD,"aar_pc      : %d\n",pAAR->pc);
  PetscPrintf(PETSC_COMM_WORLD,"solver_tol  : %e \n",pAAR->solver_tol);
  PetscPrintf(PETSC_COMM_WORLD,"m_aar       : %d\n",pAAR->m_aar);
  PetscPrintf(PETSC_COMM_WORLD,"p_aar       : %d\n",pAAR->p_aar);
  PetscPrintf(PETSC_COMM_WORLD,"beta_aar    : %lf\n",pAAR->beta_aar);

  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");

  return;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Setup and finalize functions
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void Setup_and_Initialize(AAR_OBJ* pAAR)
{
  ObjectInitialize(pAAR);
  Read_parameters(pAAR);
  Objects_Create(pAAR);      
}

void ObjectInitialize(AAR_OBJ* pAAR)
{
  PetscOptionsGetString(PETSC_NULL,PETSC_NULL, "-name",pAAR->file,sizeof(pAAR->file),PETSC_NULL);
}

void Objects_Create(AAR_OBJ* pAAR)
{
  PetscInt n_x = pAAR->numPoints_x;
  PetscInt n_y = pAAR->numPoints_y;
  PetscInt n_z = pAAR->numPoints_z;
  PetscInt o = pAAR->order;
  double RHSsum;

  PetscInt gxdim,gydim,gzdim,xcor,ycor,zcor,lxdim,lydim,lzdim,nprocx,nprocy,nprocz;
  int i;
  Mat A;
  PetscMPIInt comm_size;
  MPI_Comm_size(PETSC_COMM_WORLD,&comm_size);

  DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,n_x,n_y,n_z,
	       PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,o,0,0,0,&pAAR->da);


  DMCreateGlobalVector(pAAR->da,&pAAR->RHS);
  VecDuplicate(pAAR->RHS,&pAAR->Phi);
  
  PetscRandom rnd;
  unsigned long seed;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  
  // RHS vector
  PetscRandomCreate(PETSC_COMM_WORLD,&rnd);
  PetscRandomSetFromOptions(rnd);
  seed=rank; //0;  
  PetscRandomSetSeed(rnd,seed);
  PetscRandomSeed(rnd);

  VecSetRandom(pAAR->RHS,rnd);
  VecSum(pAAR->RHS,&RHSsum);
  RHSsum=-RHSsum/(n_x*n_y*n_z);
  VecShift(pAAR->RHS,RHSsum); // make sum of RHS zero for periodic problem 
  //VecView(pAAR->RHS,PETSC_VIEWER_STDOUT_WORLD);
  
  PetscRandomDestroy(&rnd);

  // Initial guess
  PetscRandomCreate(PETSC_COMM_WORLD,&rnd);
  PetscRandomSetFromOptions(rnd);
  seed=1;  
  PetscRandomSetSeed(rnd,seed);
  PetscRandomSeed(rnd);
  
  VecSetRandom(pAAR->Phi,rnd);
  //VecView(pAAR->Phi,PETSC_VIEWER_STDOUT_WORLD);

  PetscRandomDestroy(&rnd);
  
  
  if(comm_size == 1 ) 
    {
      DMCreateMatrix(pAAR->da,&pAAR->poissonOpr);
      DMSetMatType(pAAR->da,MATSEQSBAIJ); // real symmetric
    }
  else 
    {
      DMCreateMatrix(pAAR->da,&pAAR->poissonOpr);
      DMSetMatType(pAAR->da,MATMPISBAIJ); // real symmetric
    }

}

/////////////////////////////////////////////////////////////////////////////////////////

// Destroy objects
void Objects_Destroy(AAR_OBJ* pAAR)
{
  DMDestroy(&pAAR->da);
  VecDestroy(&pAAR->RHS); 
  VecDestroy(&pAAR->Phi);
  MatDestroy(&pAAR->poissonOpr); 
    
  return;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Matrix creation
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void ComputeMatrixA(AAR_OBJ* pAAR)
{
  PetscInt i,j,k,l,colidx,gxdim,gydim,gzdim,xcor,ycor,zcor,lxdim,lydim,lzdim,nprocx,nprocy,nprocz;
  MatStencil row;
  MatStencil* col;
  PetscScalar* val;
  PetscInt o = pAAR->order;  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  PetscScalar Dinv_factor;
  Dinv_factor = (pAAR->coeffs[0]); // (-1/4pi)(Lap) = Diag term
  pAAR->Dinv_factor = 1/Dinv_factor;
  
  DMDAGetInfo(pAAR->da,0,&gxdim,&gydim,&gzdim,&nprocx,&nprocy,&nprocz,0,0,0,0,0,0);
  PetscPrintf(PETSC_COMM_WORLD,"nprocx: %d, nprocy: %d, nprocz: %d\n",nprocx,nprocy,nprocz); 
  DMDAGetCorners(pAAR->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);

  MatScale(pAAR->poissonOpr,0.0);

  PetscMalloc(sizeof(MatStencil)*(o*6+1),&col);
  PetscMalloc(sizeof(PetscScalar)*(o*6+1),&val);

  for(k=zcor; k<zcor+lzdim; k++)
    for(j=ycor; j<ycor+lydim; j++)
      for(i=xcor; i<xcor+lxdim; i++)
	{
	  row.k = k; row.j = j, row.i = i;
   
	  colidx=0; 
	  col[colidx].i=i ;col[colidx].j=j ;col[colidx].k=k ;
	  val[colidx++]=pAAR->coeffs[0] ;
	  for(l=1;l<=o;l++)
	    {
	      col[colidx].i=i ;col[colidx].j=j ;col[colidx].k=k-l ;
	      val[colidx++]=pAAR->coeffs[l];
	      col[colidx].i=i ;col[colidx].j=j ;col[colidx].k=k+l ;
	      val[colidx++]=pAAR->coeffs[l];
	      col[colidx].i=i ;col[colidx].j=j-l ;col[colidx].k=k ;
	      val[colidx++]=pAAR->coeffs[l];
	      col[colidx].i=i ;col[colidx].j=j+l ;col[colidx].k=k ;
	      val[colidx++]=pAAR->coeffs[l];
	      col[colidx].i=i-l ;col[colidx].j=j ;col[colidx].k=k ;
	      val[colidx++]=pAAR->coeffs[l];
	      col[colidx].i=i+l ;col[colidx].j=j ;col[colidx].k=k ;
	      val[colidx++]=pAAR->coeffs[l];
	    }
	  MatSetValuesStencil(pAAR->poissonOpr,1,&row,6*o+1,col,val,ADD_VALUES);
	}
  MatAssemblyBegin(pAAR->poissonOpr, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pAAR->poissonOpr, MAT_FINAL_ASSEMBLY); 
}
