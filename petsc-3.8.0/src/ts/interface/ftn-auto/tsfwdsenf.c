#include "petscsys.h"
#include "petscfix.h"
#include "petsc/private/fortranimpl.h"
/* tsfwdsen.c */
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

#include "petscts.h"
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define tsforwardsetup_ TSFORWARDSETUP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define tsforwardsetup_ tsforwardsetup
#endif
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define tsforwardsetintegralgradients_ TSFORWARDSETINTEGRALGRADIENTS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define tsforwardsetintegralgradients_ tsforwardsetintegralgradients
#endif
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define tsforwardgetintegralgradients_ TSFORWARDGETINTEGRALGRADIENTS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define tsforwardgetintegralgradients_ tsforwardgetintegralgradients
#endif
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define tsforwardstep_ TSFORWARDSTEP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define tsforwardstep_ tsforwardstep
#endif
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define tsforwardsetsensitivities_ TSFORWARDSETSENSITIVITIES
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define tsforwardsetsensitivities_ tsforwardsetsensitivities
#endif
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define tsforwardgetsensitivities_ TSFORWARDGETSENSITIVITIES
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define tsforwardgetsensitivities_ tsforwardgetsensitivities
#endif


/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void PETSC_STDCALL  tsforwardsetup_(TS ts, int *__ierr){
*__ierr = TSForwardSetUp(
	(TS)PetscToPointer((ts) ));
}
PETSC_EXTERN void PETSC_STDCALL  tsforwardsetintegralgradients_(TS ts,PetscInt *numfwdint,Vec *vp,Vec *v, int *__ierr){
*__ierr = TSForwardSetIntegralGradients(
	(TS)PetscToPointer((ts) ),*numfwdint,vp,v);
}
PETSC_EXTERN void PETSC_STDCALL  tsforwardgetintegralgradients_(TS ts,PetscInt *numfwdint,Vec **v,Vec **vp, int *__ierr){
*__ierr = TSForwardGetIntegralGradients(
	(TS)PetscToPointer((ts) ),numfwdint,v,vp);
}
PETSC_EXTERN void PETSC_STDCALL  tsforwardstep_(TS ts, int *__ierr){
*__ierr = TSForwardStep(
	(TS)PetscToPointer((ts) ));
}
PETSC_EXTERN void PETSC_STDCALL  tsforwardsetsensitivities_(TS ts,PetscInt *nump,Vec *sp,PetscInt *num,Vec *s, int *__ierr){
*__ierr = TSForwardSetSensitivities(
	(TS)PetscToPointer((ts) ),*nump,sp,*num,s);
}
PETSC_EXTERN void PETSC_STDCALL  tsforwardgetsensitivities_(TS ts,PetscInt *nump,Vec **sp,PetscInt *num,Vec **s, int *__ierr){
*__ierr = TSForwardGetSensitivities(
	(TS)PetscToPointer((ts) ),nump,sp,num,s);
}
#if defined(__cplusplus)
}
#endif
