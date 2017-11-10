#include "petscsys.h"
#include "petscfix.h"
#include "petsc/private/fortranimpl.h"
/* strumpack.c */
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

#include "petscmat.h"
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define matstrumpacksethssrelcomptol_ MATSTRUMPACKSETHSSRELCOMPTOL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define matstrumpacksethssrelcomptol_ matstrumpacksethssrelcomptol
#endif
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define matstrumpacksethssminsize_ MATSTRUMPACKSETHSSMINSIZE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define matstrumpacksethssminsize_ matstrumpacksethssminsize
#endif
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define matstrumpacksetcolperm_ MATSTRUMPACKSETCOLPERM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define matstrumpacksetcolperm_ matstrumpacksetcolperm
#endif


/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void PETSC_STDCALL  matstrumpacksethssrelcomptol_(Mat F,PetscReal *rctol, int *__ierr){
*__ierr = MatSTRUMPACKSetHSSRelCompTol(
	(Mat)PetscToPointer((F) ),*rctol);
}
PETSC_EXTERN void PETSC_STDCALL  matstrumpacksethssminsize_(Mat F,PetscInt *hssminsize, int *__ierr){
*__ierr = MatSTRUMPACKSetHSSMinSize(
	(Mat)PetscToPointer((F) ),*hssminsize);
}
PETSC_EXTERN void PETSC_STDCALL  matstrumpacksetcolperm_(Mat F,PetscBool *cperm, int *__ierr){
*__ierr = MatSTRUMPACKSetColPerm(
	(Mat)PetscToPointer((F) ),*cperm);
}
#if defined(__cplusplus)
}
#endif
