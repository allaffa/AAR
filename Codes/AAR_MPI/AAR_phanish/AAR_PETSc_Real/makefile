all:aar

PETSC_DIR = /sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/petsc-3.8.0
PETSC_INCLUDE = ${PETSC_DIR}/include
LAPACK_DIR = /sunhome/mlupopa/priv/PhD_project/Codes/Linear_Solver/AAR_MPI/lapack-3.7.1

CPPFLAGS = -I ./src -g -lpthread -I${PETSC_INCLUDE} -I${LAPACK_DIR}/LAPACKE/include
 
SOURCEC = ./AAR_Real.cc 
OBJSC = ./AAR_Real.o 
LIBBASE = ./AAR_Real

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

aar: ${OBJSC} chkopts
	${CLINKER} ${CPPFLAGS} -o ${LIBBASE} ${OBJSC} ${PETSC_LIB}
	${RM} ${OBJSC}



