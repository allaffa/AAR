all:aar

CC = mpic++ 

CFLAGS = -O3 -funroll-loops -g -Wall 

SOURCEC = ./AAR.cpp 
OBJSC = ./AAR.o

%.o: %.cpp ${SOURCEH}
	${CC} -c -o $@ $< ${CFLAGS} 

aar:${OBJSC} 
	${CC} -o ./aar $^ ${CFLAGS} ${LIBS}

.PHONY: clean
clean:
	rm -f ./*.o *~ core core*
