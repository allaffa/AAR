
#requiresdefine   'PETSC_HAVE_FORTRAN'
ALL: lib
CPPFLAGS = ${NETCFD_INCLUDE} ${EXODUSII_INCLUDE}
CFLAGS   =
FFLAGS   =
SOURCEC  = plexfemf.c plexcreatef.c plexrefinef.c plextreef.c plexexodusiif.c plexnaturalf.c plexcgnsf.c plexindicesf.c plexfvmf.c plexf.c plexdistributef.c plexinterpolatef.c plexpartitionf.c plexorientf.c plexpointf.c plexgmshf.c plexgeometryf.c plexsubmeshf.c plexreorderf.c
SOURCEF  =
SOURCEH  = 
DIRS     =
LIBBASE  = libpetscdm
LOCDIR   = src/dm/impls/plex/ftn-auto/
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
