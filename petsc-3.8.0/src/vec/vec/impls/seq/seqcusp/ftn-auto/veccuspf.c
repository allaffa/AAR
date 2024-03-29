#include "petscsys.h"
#include "petscfix.h"
#include "petsc/private/fortranimpl.h"
/* veccusp.c */
/* Fortran interface file */

/*
* This file was generated automatically by bfort from the C source
* file.  
 */

#ifdef PETSC_USE_POINTER_CONVERSION
#if defined(__cplusplus)
extern "C" { 
#endif 
extern void *PetscToPointer(void*);
extern int PetscFromPointer(void *);
extern void PetscRmPointer(void*);
#if defined(__cplusplus)
} 
#endif 

#else

#define PetscToPointer(a) (*(PetscFortranAddr *)(a))
#define PetscFromPointer(a) (PetscFortranAddr)(a)
#define PetscRmPointer(a)
#endif

#include <petscvec.h>
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define veccreateseqcusp_ VECCREATESEQCUSP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define veccreateseqcusp_ veccreateseqcusp
#endif


/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void PETSC_STDCALL  veccreateseqcusp_(MPI_Fint * comm,PetscInt *n,Vec *v, int *__ierr){
*__ierr = VecCreateSeqCUSP(
	MPI_Comm_f2c(*(comm)),*n,v);
}
#if defined(__cplusplus)
}
#endif
