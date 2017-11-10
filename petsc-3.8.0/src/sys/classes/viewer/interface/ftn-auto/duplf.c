#include "petscsys.h"
#include "petscfix.h"
#include "petsc/private/fortranimpl.h"
/* dupl.c */
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

#include "petscviewer.h"
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscviewergetsubviewer_ PETSCVIEWERGETSUBVIEWER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscviewergetsubviewer_ petscviewergetsubviewer
#endif
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscviewerrestoresubviewer_ PETSCVIEWERRESTORESUBVIEWER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscviewerrestoresubviewer_ petscviewerrestoresubviewer
#endif


/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif
PETSC_EXTERN void PETSC_STDCALL  petscviewergetsubviewer_(PetscViewer viewer,MPI_Fint * comm,PetscViewer *outviewer, int *__ierr){
*__ierr = PetscViewerGetSubViewer(
	(PetscViewer)PetscToPointer((viewer) ),
	MPI_Comm_f2c(*(comm)),outviewer);
}
PETSC_EXTERN void PETSC_STDCALL  petscviewerrestoresubviewer_(PetscViewer viewer,MPI_Fint * comm,PetscViewer *outviewer, int *__ierr){
*__ierr = PetscViewerRestoreSubViewer(
	(PetscViewer)PetscToPointer((viewer) ),
	MPI_Comm_f2c(*(comm)),outviewer);
}
#if defined(__cplusplus)
}
#endif
